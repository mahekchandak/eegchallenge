"""
Enhanced EEGNet + Transformer architecture for improved EEG processing
Includes:
- EEGNet backbone for spatial feature extraction
- Transformer encoder for temporal dependencies
- Multi-task learning heads
- Self-attention mechanisms
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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]


class EEGNetTransformerEnhanced(nn.Module):
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
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # Enhanced spatial feature extraction (EEGNet backbone)
        # Increased F1 and F2 for better feature extraction
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
        
        # SSL head for reconstruction
        self.ssl_decoder = nn.Sequential(
            nn.Linear(d_model, self.feature_dim),
            nn.ELU(),
        )
        self.ssl_deconv = nn.Sequential(
            # Reverse avgpool2
            nn.ConvTranspose2d(self.F2, self.F2, (1, 8), stride=(1, 8)),
            # Reverse separable conv
            nn.Conv2d(self.F2, self.F1 * self.D, (1, 16), padding='same'),
            nn.ELU(),
            # Reverse avgpool1
            nn.ConvTranspose2d(self.F1 * self.D, self.F1 * self.D, (1, 4), stride=(1, 4), output_padding=(0, 1)),
            nn.ELU(),
            # Reverse depthwise conv
            nn.ConvTranspose2d(self.F1 * self.D, self.F1, (self.Chans, 1), groups=self.F1),
            nn.ELU(),
            # Reverse conv1
            nn.Conv2d(self.F1, 1, (1, self.kernLength), padding='same')
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

    def forward(self, x, ssl_task=False):
        # Input shape: (batch, channels, time) -> (batch, 1, channels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Extract spatial-temporal features
        features = self.feature_extraction(x)
        
        # Reshape for transformer
        batch_size = features.shape[0]
        x = features.permute(0, 3, 1, 2)  # (batch, seq, channels, features)
        x = x.reshape(batch_size, self.seq_length, self.feature_dim)
        
        # Project to transformer dimension
        x = self.feature_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)

        if ssl_task:
            x = self.ssl_decoder(x)
            x = x.reshape(batch_size, self.seq_length, self.feature_dim, 1)
            x = x.permute(0, 2, 3, 1)
            x = self.ssl_deconv(x)
            return x.squeeze(1)

        
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