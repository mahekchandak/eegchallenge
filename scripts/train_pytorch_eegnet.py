"""
PyTorch EEGNet training script for NeurIPS EEG Challenge
Trains on BIDS .bdf data and saves weights for submission
"""

import os
import sys
import json
import warnings
from typing import List, Tuple, Dict
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add models path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT_DIR)

from models.eegnet_pytorch import Model as EEGNetBaseline
from models.eegnet_transformer import EEGNetTransformer
from scripts.train_eegnet_bids import prepare_data  # Reuse data loading


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def convert_to_binary_labels(y, event_id):
    """
    Convert multi-class labels to binary classification for challenge
    Strategy: Use most frequent classes as binary targets
    """
    from collections import Counter
    
    # Count label frequencies
    label_counts = Counter(y)
    print(f"Original label distribution: {dict(label_counts)}")
    
    # Get the two most frequent classes
    most_common = label_counts.most_common(2)
    if len(most_common) < 2:
        print("Warning: Less than 2 classes found, using random binary split")
        # Create binary split if only one class
        unique_labels = np.unique(y)
        if len(unique_labels) == 1:
            # Randomly assign half to class 0, half to class 1
            binary_y = np.random.randint(0, 2, size=len(y))
        else:
            # Use first two unique labels
            binary_y = np.where(y == unique_labels[0], 0, 1)
    else:
        class_0, class_1 = most_common[0][0], most_common[1][0]
        print(f"Using classes {class_0} and {class_1} for binary classification")
        
        # Create binary labels
        binary_y = np.where(y == class_0, 0, 1)
        # Filter out samples that aren't in the two most common classes
        mask = np.isin(y, [class_0, class_1])
        binary_y = binary_y[mask]
    
    print(f"Binary label distribution: {Counter(binary_y)}")
    return binary_y, mask if 'mask' in locals() else np.ones(len(y), dtype=bool)


def train_pytorch_eegnet(X: np.ndarray, y: np.ndarray, n_chans: int, n_samples: int,
                        lr: float = 1e-3, batch_size: int = 32, epochs: int = 50,
                        device: str = 'cpu', save_path: str = None) -> Dict[str, float]:
    """
    Train PyTorch EEGNet model
    """
    
    # Convert data format: from (n_trials, n_chans, n_samples, 1) to (n_trials, n_chans, n_samples)
    if X.ndim == 4 and X.shape[-1] == 1:
        X = X.squeeze(-1)
    
    # Convert to binary classification for challenge
    y_binary, mask = convert_to_binary_labels(y, None)
    X = X[mask]  # Filter data to match binary labels
    
    print(f"After binary conversion: {X.shape[0]} samples, {len(np.unique(y_binary))} classes")
    
    # Create dataset and split
    dataset = EEGDataset(X, y_binary)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model - force binary classification for challenge
    nb_classes = 2  # Binary classification for NeurIPS challenge
    if os.environ.get("EEG_MODEL", "baseline").lower() == "transformer":
        model = EEGNetTransformer(nb_classes=nb_classes, Chans=n_chans, Samples=n_samples)
    else:
        model = EEGNetBaseline(nb_classes=nb_classes, Chans=n_chans, Samples=n_samples)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Training on {device} with {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return {
        "best_val_accuracy": best_val_acc / 100.0,
        "final_train_accuracy": train_accs[-1] / 100.0,
        "final_val_loss": val_losses[-1],
        "final_train_loss": train_losses[-1],
    }


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch EEGNet for NeurIPS Challenge")
    parser.add_argument("--bids_root", type=str, default=os.path.join(ROOT_DIR, "R1_mini_L100_bdf"))
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    parser.add_argument("--max_subjects", type=int, default=None, help="Max subjects to use (None = all)")
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=2.0)  # 2 seconds for challenge format
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_weights", type=str, default="weights_challenge_1.pt")
    parser.add_argument("--resample_hz", type=int, default=100)  # Challenge uses 100Hz
    args = parser.parse_args()
    
    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("Loading BIDS data...")
    if args.max_subjects is None:
        print("Using ALL subjects in the dataset")
    else:
        print(f"Using max {args.max_subjects} subjects")
        
    X, y, n_chans, n_samples, event_id = prepare_data(
        bids_root=args.bids_root,
        subjects=args.subjects,
        tasks=args.tasks,
        max_subjects=args.max_subjects,
        tmin=args.tmin,
        tmax=args.tmax,
        resample_hz=args.resample_hz,
    )
    
    print(f"Data shape: {X.shape}, Labels: {len(np.unique(y))} classes")
    print(f"Channels: {n_chans}, Samples: {n_samples}")
    print(f"Event mapping: {event_id}")
    
    # Train model
    metrics = train_pytorch_eegnet(
        X, y, n_chans, n_samples,
        lr=args.lr, 
        batch_size=args.batch_size, 
        epochs=args.epochs,
        device=str(device),
        save_path=args.save_weights
    )
    
    # Save metrics
    metrics_file = args.save_weights.replace('.pt', '_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nFinal Results:")
    print(json.dumps(metrics, indent=2))
    print(f"\nWeights saved to: {args.save_weights}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
