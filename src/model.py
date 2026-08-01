import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifierCNN(nn.Module):
    """
    Lightweight 3-Layer Convolutional Neural Network for Binary Classification
    Input: (B, 3, 256, 256)
    Output: Sigmoid probability [0.0 - 1.0] (0: Clean, 1: Garbage/Litter)
    """
    def __init__(self):
        super(BinaryClassifierCNN, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Feature dimension after 3 poolings of 256x256: 256 -> 128 -> 64 -> 32
        self.fc1 = nn.Linear(16 * 32 * 32, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def build_torch_model():
    return BinaryClassifierCNN()
