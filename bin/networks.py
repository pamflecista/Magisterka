import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class BassetNetwork(torch.nn.Module):

    def __init__(self):
        super(BassetNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 300, kernel_size=19, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(300, 200, kernel_size=11, stride=1)


        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)