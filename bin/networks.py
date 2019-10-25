import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class BassetNetwork(torch.nn.Module):

    def __init__(self):
        super(BassetNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 300, kernel_size=(4, 19), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(300)
        self.conv2 = nn.Conv2d(300, 200, kernel_size=(4, 11), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(200)
        self.conv3 = nn.Conv2d(200, 200, kernel_size=(4, 7), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(200)
        self.dense1 = nn.Linear(in_features=200, out_features=50)
        self.dense1_bn = nn.BatchNorm1d(50)
        self.dense2 = nn.Linear(50, 10)


        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x)), 3))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x)), 4))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x)), 4))
        x = x.view(-1, 320)  # reshape
        x = F.relu(self.dense1_bn(self.dense1(x)))
        x = F.relu(self.dense2(x))
        return F.log_softmax(x)