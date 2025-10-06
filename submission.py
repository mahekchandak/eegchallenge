"""
NeurIPS EEG Challenge Submission with Enhanced Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class EEGNet(nn.Module):
    """
    Enhanced EEGNet with Transformer layers and multi-task capability
    """
    def __init__(self, nb_classes=2, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=16, 
                 D=2, F2=32, norm_rate=0.25,
                 n_transformer_layers=2, 
                 d_model=128,
                 nhead=8,
                 regression=False,
                 multi_task=True):
        super().__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        self.regression = regression
        self.multi_task = multi_task
        
        # Enhanced spatial feature extraction
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise convolution with increased depth
        self.depthwise = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        # Additional spatial attention
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(F1 * D, F1 * D, 1),
            nn.Sigmoid()
        )
        
        # First feature processing block
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # Enhanced separable convolution
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)

        # Calculate feature dimensions for transformer
        with torch.no_grad():
            x = torch.randn(1, 1, Chans, Samples)
            x = self.feature_extraction(x)
            self.feature_dim = x.shape[1] * x.shape[2]
            self.seq_length = x.shape[3]

        # Transformer encoder for temporal dependencies
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropoutRate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=n_transformer_layers
        )
        
        # Project features to transformer dimension
        self.feature_projection = nn.Linear(self.feature_dim, d_model)
        
        # Task-specific heads
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Dropout(dropoutRate),
            nn.Linear(d_model // 2, nb_classes)
        )
        
        if multi_task or regression:
            self.regression_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ELU(),
                nn.Dropout(dropoutRate),
                nn.Linear(d_model // 2, 1)
            )
        
    def feature_extraction(self, x):
        # Initial temporal convolution
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Spatial feature extraction
        x = self.depthwise(x)
        x = self.bn2(x)
        
        # Apply spatial attention
        attn = self.spatial_attention(x)
        x = x * attn
        
        # First feature processing
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Enhanced feature processing
        x = self.separable(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        return x

    def forward(self, x):
        # Input shape: (batch, channels, time) -> (batch, 1, channels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Extract spatial-temporal features
        x = self.feature_extraction(x)
        
        # Reshape for transformer
        batch_size = x.shape[0]
        x = x.permute(0, 3, 1, 2)  # (batch, seq, channels, features)
        x = x.reshape(batch_size, self.seq_length, self.feature_dim)
        
        # Project to transformer dimension
        x = self.feature_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over sequence length
        x = torch.mean(x, dim=1)
        
        # Task-specific predictions
        if self.multi_task:
            class_output = self.classification_head(x)
            reg_output = self.regression_head(x)
            if not self.training:  # During inference
                class_output = F.softmax(class_output, dim=1)
            return class_output, reg_output
        else:
            if self.regression:
                return self.regression_head(x)
            else:
                x = self.classification_head(x)
                if not self.training:
                    x = F.softmax(x, dim=1)
                return x


class Submission:
    
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        
    def get_model_challenge_1(self):
        """
        Model for Challenge 1
        """
        model_challenge1 = EEGNet(
            nb_classes=2,
            Chans=129,
            Samples=int(2 * self.sfreq)
        ).to(self.device)
        
        try:
            weight_path = "weights_challenge_1.pt"
            checkpoint_1 = torch.load(weight_path, map_location=self.device)
            model_challenge1.load_state_dict(checkpoint_1)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"No weights found for Challenge 1: {e}")
        
        return model_challenge1
    
    def get_model_challenge_2(self):
        """
        Model for Challenge 2
        """
        model_challenge2 = EEGNet(
            nb_classes=1,
            Chans=129,
            Samples=int(2 * self.sfreq),
            regression=True
        ).to(self.device)
        
        try:
            weight_path = "weights_challenge_2.pt"
            checkpoint_2 = torch.load(weight_path, map_location=self.device)
            model_challenge2.load_state_dict(checkpoint_2)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"No weights found for Challenge 2: {e}")
        
        return model_challenge2


if __name__ == "__main__":
    # Test submission format
    SFREQ = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sub = Submission(SFREQ, DEVICE)
    
    # Test Challenge 1 model
    model_1 = sub.get_model_challenge_1()
    model_1.eval()
    
    # Test with dummy data
    x = torch.randn(1, 129, 200).to(DEVICE)
    with torch.inference_mode():
        y_pred = model_1(x)
        print(f"Challenge 1 - Input shape: {x.shape}, Output shape: {y_pred.shape}")
    
    # Test Challenge 2 model
    model_2 = sub.get_model_challenge_2()
    model_2.eval()
    
    with torch.inference_mode():
        y_pred = model_2(x)
        print(f"Challenge 2 - Input shape: {x.shape}, Output shape: {y_pred.shape}")
