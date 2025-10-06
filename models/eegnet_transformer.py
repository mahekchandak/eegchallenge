"""
EEGNet + Transformer encoder with dual heads (classification + regression)
Designed for NeurIPS EEG Challenge Phase 1 (two challenges).
"""

import torch
import torch.nn as nn


class ConvStemEEGNet(nn.Module):
    def __init__(self, Chans: int, Samples: int, dropout: float = 0.5,
                 F1: int = 8, D: int = 2, kernLength: int = 64, F2: int = 16):
        super().__init__()
        self.Chans = Chans
        self.Samples = Samples

        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        # shape: (B, F2, 1, T')
        return x.squeeze(2)  # (B, F2, T')


class EEGNetTransformer(nn.Module):
    def __init__(self,
                 nb_classes: int = 2,
                 Chans: int = 129,
                 Samples: int = 200,
                 dropout: float = 0.5,
                 F1: int = 8,
                 D: int = 2,
                 F2: int = 16,
                 kernLength: int = 64,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 128,
                 regression: bool = True):
        super().__init__()
        self.nb_classes = nb_classes
        self.regression_enabled = regression

        self.stem = ConvStemEEGNet(Chans=Chans, Samples=Samples, dropout=dropout,
                                   F1=F1, D=D, kernLength=kernLength, F2=F2)

        # Project conv features (F2) to d_model for Transformer
        self.proj = nn.Conv1d(F2, d_model, kernel_size=1, bias=False)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Sequence pooling (mean)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Heads
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, nb_classes)
        )
        self.reg_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, 1)
        ) if regression else None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor, return_logits: bool = True):
        # x: (B, C, T)
        x = self.stem(x)              # (B, F2, T')
        x = self.proj(x)              # (B, d_model, T')
        x = x.transpose(1, 2)         # (B, T', d_model)
        x = self.encoder(x)           # (B, T', d_model)
        x = x.transpose(1, 2)         # (B, d_model, T')
        x = self.pool(x)              # (B, d_model, 1)

        cls_logits = self.cls_head(x) # (B, nb_classes)
        if not return_logits:
            cls_out = self.softmax(cls_logits)
        else:
            cls_out = cls_logits

        if self.reg_head is not None:
            reg_out = self.reg_head(x)    # (B, 1)
            return cls_out, reg_out
        return cls_out


class Model(nn.Module):
    """Wrapper to match submission expectations."""
    def __init__(self, nb_classes: int = 2, Chans: int = 129, Samples: int = 200):
        super().__init__()
        self.core = EEGNetTransformer(nb_classes=nb_classes, Chans=Chans, Samples=Samples)

    def forward(self, x):
        out = self.core(x)
        if isinstance(out, tuple):
            cls_out, _ = out
            return cls_out
        return out


