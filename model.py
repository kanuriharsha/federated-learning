# model.py
import torch
from torch import nn

class ImageClassifier(nn.Module):
    def __init__(self):  # <-- Fix: Use __init__ instead of init
        super(ImageClassifier, self).__init__()  # <-- Fix: Call __init__ properly
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)  # Adjusted the size to match your layers
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
