import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BassetNetwork(torch.nn.Module):

    def __init__(self):
        super(BassetNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 300, kernel_size=(4, 19), stride=1, padding=0),
            nn.BatchNorm2d(300),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(300, 200, kernel_size=(4, 11), stride=1, padding=0),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(200, 200, kernel_size=(4, 7), stride=1, padding=0),
            nn.BatchNorm2d(200),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=1))
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=200, out_features=1000),
            nn.ReLU(),
            nn.Dropout())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=164),
            nn.ReLU(),
            nn.Dropout())
        self.fc3 = nn.Linear(164, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, 320)  # reshape
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x)

