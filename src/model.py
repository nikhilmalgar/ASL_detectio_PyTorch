# src/model.py

import torch.nn as nn
import torch.nn.functional as F

class ASLClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(ASLClassifier, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # [B, 3, 64, 64] -> [B, 32, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [B, 32, 32, 32]

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> [B, 64, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [B, 64, 16, 16]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# -> [B, 128, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [B, 128, 8, 8]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                                # -> [B, 128*8*8]
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)                  # -> [B, num_classes]
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x
