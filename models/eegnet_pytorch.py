"""
PyTorch implementation of EEGNet for NeurIPS EEG Challenge submission
Based on the original EEGNet architecture from arl-eegmodels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F


class EEGNet(nn.Module):
    """
    PyTorch implementation of EEGNet
    Adapted from: http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    """
    
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25):
        super(EEGNet, self).__init__()
        
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples
        
        # Block 1: Temporal convolution + Depthwise convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise convolution (spatial filtering)
        self.depthwise = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        
        # Activation and pooling
        self.activation1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # Block 2: Separable convolution
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # Classification
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(self._get_linear_input_size(), nb_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def _get_linear_input_size(self):
        """Calculate the input size for the linear layer"""
        # Simulate forward pass to get flattened size
        x = torch.randn(1, 1, self.Chans, self.Samples)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.separable(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        return x.size(1)
    
    def forward(self, x):
        # Input shape: (batch, channels, time) -> (batch, 1, channels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        
        return x


class Model(nn.Module):
    """
    Main model class for NeurIPS EEG Challenge submission
    """
    
    def __init__(self, nb_classes=2, Chans=129, Samples=200, dropoutRate=0.5):
        super(Model, self).__init__()
        
        # Adjust kernel length for 100Hz sampling rate (challenge uses 100Hz)
        # Original EEGNet assumes 128Hz, so we scale kernel length
        kernLength = int(64 * 100 / 128)  # Scale from 128Hz to 100Hz
        
        self.eegnet = EEGNet(
            nb_classes=nb_classes,
            Chans=Chans,
            Samples=Samples,
            dropoutRate=dropoutRate,
            kernLength=kernLength,
            F1=8,
            D=2,
            F2=16
        )
    
    def forward(self, x):
        """
        Forward pass for challenge evaluation
        Input: x shape (batch_size, 129, 200)
        Output: predictions shape (batch_size, nb_classes)
        """
        return self.eegnet(x)


def create_model(nb_classes=2, Chans=129, Samples=200):
    """Factory function to create model instance"""
    return Model(nb_classes=nb_classes, Chans=Chans, Samples=Samples)


if __name__ == "__main__":
    # Test the model
    model = create_model()
    x = torch.randn(2, 129, 200)  # batch_size=2, channels=129, time=200
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
