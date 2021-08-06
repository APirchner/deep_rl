from typing import Tuple

import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, input_dim: Tuple[int], num_actions: int):
        super(QNet, self).__init__()
        c, h, w = input_dim
        assert h & (h - 1) == 0
        assert h == w

        flat_size = 64 * (((((h - 3) // 2 + 1) - 3) // 2 + 1) - 2) ** 2

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(in_features=flat_size, out_features=512),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(in_features=512, out_features=num_actions)
        )

    def forward(self, input):
        x = self.net(input)
        return x
