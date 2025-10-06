"""
Enhanced training script with:
- Multi-task learning
- Domain generalization
- Self-supervised pretraining
- Advanced preprocessing
"""

import os
import sys
import json
import glob
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import mne
from mne import Epochs
from mne.io import read_raw_bdf
from sklearn.model_selection import train_test_split
from models.eegnet_transformer_enhanced import EEGNetTransformerEnhanced

# Import data loading functions from existing script
from scripts.train_eegnet_bids import prepare_data, find_bids_runs, load_epochs_from_bdf, build_event_id
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse


class EEGPreprocessor:
    """Enhanced EEG preprocessing pipeline"""
    def __init__(self, sfreq=128, l_freq=0.1, h_freq=50):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
    
    def preprocess(self, raw_eeg):
        """Apply comprehensive preprocessing pipeline"""
        # Filtering
        raw_eeg = mne.filter.filter_data(
            raw_eeg,
            self.sfreq,
            l_freq=self.l_freq,
            h_freq=self.h_freq,
            method='iir'
        )
        
        # Standardization (per channel)
        raw_eeg = (raw_eeg - np.mean(raw_eeg, axis=1, keepdims=True)) / \
                  (np.std(raw_eeg, axis=1, keepdims=True) + 1e-8)
        
        return raw_eeg


class EEGDataAugmentation:
    """Data augmentation techniques for EEG"""
    @staticmethod
    def gaussian_noise(x, std=0.1):
        return x + torch.randn_like(x) * std
    
    @staticmethod
    def time_mask(x, max_mask_size=0.1):
        mask_size = int(x.shape[-1] * max_mask_size)
        if mask_size > 0:
            start = torch.randint(0, x.shape[-1] - mask_size, (1,))
            x[:, :, start:start+mask_size] = 0
        return x
    
    @staticmethod
    def channel_dropout(x, p=0.1):
        mask = torch.bernoulli(torch.full((x.shape[1], 1), 1-p, device=x.device))
        return x * mask


def create_self_supervised_task(x, augmentation):
    """Create self-supervised learning tasks"""
    # Task 1: Predict clean signal from noisy input
    x_noisy = augmentation.gaussian_noise(x.clone())
    
    # Task 2: Predict missing segments
    x_masked = augmentation.time_mask(x.clone())
    
    # Task 3: Predict signals with dropped channels
    x_channel_drop = augmentation.channel_dropout(x.clone())
    
    return x_noisy, x_masked, x_channel_drop, x


class CustomLoss:
    """Combined loss for multiple objectives"""
    def __init__(self):
        self.cls_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()
        self.ssl_criterion = nn.MSELoss()
    
    def __call__(self, pred_cls=None, pred_reg=None, true_cls=None, true_reg=None, ssl_pred=None, ssl_target=None):
        loss = 0
        metrics = {}
        
        # Classification loss
        if pred_cls is not None and true_cls is not None:
            cls_loss = self.cls_criterion(pred_cls, true_cls)
            loss += cls_loss
            metrics['cls_loss'] = cls_loss.item()
        
        # Regression loss
        if pred_reg is not None and true_reg is not None:
            reg_loss = self.reg_criterion(pred_reg, true_reg)
            loss += reg_loss
            metrics['reg_loss'] = reg_loss.item()
        
        # Self-supervised loss
        if ssl_pred is not None and ssl_target is not None:
            ssl_loss = self.ssl_criterion(ssl_pred, ssl_target)
            loss += ssl_loss
            metrics['ssl_loss'] = ssl_loss.item()
        
        metrics['total_loss'] = loss.item() if isinstance(loss, torch.Tensor) else loss
        return loss, metrics


def train_epoch(model, train_loader, criterion, optimizer, device, augmentation, cls_weight, reg_weight):
    model.train()
    epoch_metrics = []
    
    for batch_idx, (data, cls_target, reg_target) in enumerate(train_loader):
        data, cls_target = data.to(device), cls_target.to(device)
        if reg_target is not None:
            reg_target = reg_target.to(device).unsqueeze(1)
        
        # Create self-supervised tasks
        x_noisy, x_masked, x_channel_drop, x_clean = create_self_supervised_task(
            data, augmentation
        )
        
        optimizer.zero_grad()
        
        # Main task forward pass
        if model.multi_task:
            cls_out, reg_out = model(data, ssl_task=False)
        else:
            cls_out, reg_out = model(data, ssl_task=False), None

        # Main task loss
        main_loss, main_metrics = criterion(
            pred_cls=cls_out, true_cls=cls_target,
            pred_reg=reg_out, true_reg=reg_target
        )
        loss = cls_weight * main_metrics.get('cls_loss', 0) + reg_weight * main_metrics.get('reg_loss', 0)
        metrics = {k: v for k, v in main_metrics.items() if k != 'total_loss'}

        # Self-supervised forward and loss
        ssl_loss_total = 0
        for i, x_aug in enumerate([x_noisy, x_masked, x_channel_drop]):
            ssl_pred = model(x_aug, ssl_task=True)
            ssl_loss, ssl_metrics = criterion(ssl_pred=ssl_pred, ssl_target=x_clean)
            ssl_loss_total += ssl_loss
            metrics[f'ssl_{i}_loss'] = ssl_metrics['ssl_loss']
        
        loss += (1 - cls_weight - reg_weight) * ssl_loss_total
        metrics['total_loss'] = loss.item()

        loss.backward()
        optimizer.step()
        
        epoch_metrics.append(metrics)
    
    # Average metrics over epoch
    avg_metrics = {}
    for key in epoch_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
    
    return avg_metrics


class EEGDataset(Dataset):
    def __init__(self, X, y, reaction_times=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        if reaction_times is not None:
            self.reaction_times = torch.FloatTensor(reaction_times)
        else:
            # If no reaction times available, use zeros
            self.reaction_times = torch.zeros(len(y))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.reaction_times[idx]


def validate(model, val_loader, criterion, device):
    model.eval()
    val_metrics = []
    
    with torch.no_grad():
        for data, cls_target, reg_target in val_loader:
            data = data.to(device)
            cls_target = cls_target.to(device)
            reg_target = reg_target.to(device).unsqueeze(1)
            
            if model.multi_task:
                cls_out, reg_out = model(data, ssl_task=False)
                loss, metrics = criterion(
                    pred_cls=cls_out, true_cls=cls_target,
                    pred_reg=reg_out, true_reg=reg_target
                )
            else:
                out = model(data, ssl_task=False)
                loss, metrics = criterion(pred_cls=out, true_cls=cls_target)
            
            val_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in val_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in val_metrics])
    
    return avg_metrics


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else
                         "cpu")
    
    # Initialize preprocessing and augmentation
    augmentation = EEGDataAugmentation()
    
    # Load data using existing pipeline 
    X, y, n_chans, n_samples, event_id = prepare__data(
        bids_roots=args.data_dirs,
        subjects=None,
        tasks=None,
        tmin=0.0,
        tmax=1.0,
        resample_hz=args.sfreq
    )
    
    # Reshape X from (trials, chans, samples, 1) to (trials, chans, samples)
    X = X.squeeze(-1)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create datasets
    train_dataset = EEGDataset(X_train, y_train)
    val_dataset = EEGDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model with correct dimensions
    model = EEGNetTransformerEnhanced(
        nb_classes=len(np.unique(y_train)),
        Chans=n_chans,
        Samples=n_samples,
        dropoutRate=args.dropout,
        n_transformer_layers=args.n_transformer_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        multi_task=args.multi_task
    ).to(device)
    
    # Initialize optimizer with learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize loss function
    criterion = CustomLoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, augmentation,
            args.cls_weight, args.reg_weight
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            torch.save(model.state_dict(), args.model_path)
            
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {val_metrics}")
        print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument("--nb_classes", type=int, default=2)
    parser.add_argument("--chans", type=int, default=129)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--n_transformer_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cls_weight", type=float, default=0.5)
    parser.add_argument("--reg_weight", type=float, default=0.3)
    
    # Data parameters
    parser.add_argument("--sfreq", type=float, default=128)
    parser.add_argument("--multi_task", action="store_true")
    
    # Paths
    parser.add_argument("--model_path", type=str, default="weights_enhanced.pt")
    parser.add_argument("--data_dirs", type=str, nargs='+', default=["R1_mini_L100_bdf"],
                       help="Path to one or more BIDS dataset directories")
    
    args = parser.parse_args()
    main(args)