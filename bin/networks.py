import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import params


class BassetNetwork(torch.nn.Module):

    def __init__(self, seq_len):
        super(BassetNetwork, self).__init__()
        dropout = 0.3
        pooling_widths = [3, 4, 4]
        num_channels = [300, 200, 200]
        kernel_widths = [19, 11, 7]
        paddings = [int((w-1)/2) for w in kernel_widths]
        num_units = [1000, 4]
        self.params = {
            'input sequence length': seq_len,
            'dropout': dropout
        }

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_channels[0], kernel_size=(4, kernel_widths[0]), padding=(0, paddings[0])),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[0]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[0])

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(1, kernel_widths[1]), padding=(0, paddings[1])),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[1]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[1])

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=(1, kernel_widths[2]), padding=(0, paddings[2])),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[2]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[2])

        self.fc_input = 1 * seq_len * num_channels[-1]
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.fc_input, out_features=num_units[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout))

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=num_units[0], out_features=num_units[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class CustomNetwork(torch.nn.Module):

    def __init__(self, seq_len, num_channels=[300, 200, 200], kernel_widths=[19, 11, 7], pooling_widths=[3, 4, 4],
                 num_units=[2000, 4], dropout=params.dropout_value):
        super(CustomNetwork, self).__init__()
        paddings = [int((w-1)/2) for w in kernel_widths]
        self.seq_len = seq_len
        self.dropout = dropout
        self.params = {
            'input sequence length': seq_len,
            'convolutional layers': len(num_channels),
            'fully connected': len(num_units),
            'number of channels': num_channels,
            'kernels widths': kernel_widths,
            'pooling widths': pooling_widths,
            'units in fc': num_units,
            'dropout': dropout

        }

        conv_modules = []
        num_channels = [1] + num_channels
        for num, (input_channels, output_channels, kernel, padding, pooling) in \
                enumerate(zip(num_channels[:-1], num_channels[1:], kernel_widths, paddings, pooling_widths)):
            k = 4 if num == 0 else 1
            conv_modules += [
                nn.Conv2d(input_channels, output_channels, kernel_size=(k, kernel), padding=(0, padding)),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pooling), ceil_mode=True),
                nn.Dropout(p=params.conv_dropout_value)
            ]
            seq_len = math.ceil(seq_len / pooling)
        self.conv_layers = nn.Sequential(*conv_modules)

        fc_modules = []
        self.fc_input = 1 * seq_len * num_channels[-1]
        num_units = [self.fc_input] + num_units
        for input_units, output_units in zip(num_units[:-1], num_units[1:]):
            fc_modules += [
                nn.Linear(in_features=input_units, out_features=output_units),
                nn.ReLU(),
                nn.Dropout(p=self.dropout)
            ]
        self.fc_layers = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc_layers(x)
        return torch.sigmoid(x)


class TestNetwork(torch.nn.Module):

    def __init__(self, seq_len):
        super(TestNetwork, self).__init__()
        dropout = 0.3
        pooling_widths = [3, 4, 4]
        num_channels = [30, 20, 20]
        kernel_widths = [5, 3, 3]
        paddings = [int((w-1)/2) for w in kernel_widths]
        num_units = [100, 4]
        self.params = {
            'Input sequence length': seq_len,
            'Dropout': dropout
        }

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, num_channels[0], kernel_size=(4, kernel_widths[0]), padding=(0, paddings[0])),
            nn.BatchNorm2d(num_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[0]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[0])

        self.layer2 = nn.Sequential(
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=(1, kernel_widths[1]), padding=(0, paddings[1])),
            nn.BatchNorm2d(num_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[1]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[1])

        self.layer3 = nn.Sequential(
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=(1, kernel_widths[2]), padding=(0, paddings[2])),
            nn.BatchNorm2d(num_channels[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pooling_widths[2]), ceil_mode=True))
        seq_len = math.ceil(seq_len / pooling_widths[2])

        self.fc_input = 1 * seq_len * num_channels[-1]
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.fc_input, out_features=num_units[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout))

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=num_units[0], out_features=num_units[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fc_input)  # reshape
        x = self.fc1(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pooling, dropout):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.drop= nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class InceptionModule(nn.Module):

    def __init__(self, in_channels, f_1x1, f_3x3_r, f_3x3, f_7x7_r, f_7x7,
         f_11x11_r, f_11x11,  f_19x19_r, f_19x19, f_pp, dropout):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_1x1, kernel_size=1, stride=1, padding=0, dropout=dropout)
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_3x3_r, kernel_size=1, stride=1, padding=0,, dropout=0),
            ConvBlock(f_3x3_r, f_3x3, kernel_size=3, stride=1, padding=1, , dropout=dropout)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, f_7x7_r, kernel_size=1, stride=1, padding=0, , dropout=0),
            ConvBlock(f_7x7_r, f_7x7, kernel_size=7, stride=1, padding=3, , dropout=dropout)
        )

        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, f_11x11_r, kernel_size=1, stride=1, padding=0, , dropout=0),
            ConvBlock(f_11x11_r, f_11x11, kernel_size=11, stride=1, padding=5, , dropout=dropout)
        )

        self.branch5 = nn.Sequential(
            ConvBlock(in_channels, f_19x19_r, kernel_size=1, stride=1, padding=0, , dropout=0),
            ConvBlock(f_19x19_r, f_19x19, kernel_size=19, stride=1, padding=9, , dropout=dropout)
        )

        self.branch6 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            ConvBlock(in_channels, f_pp, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch5 = self.branch5(x)
        branch6 = self.branch6(x)

        return torch.cat([branch1, branch2, branch3, branch4, branch5, branch6], 1)

class InceptionExit(nn.Module):

    def __init__(self, in_channels, out_channels, input1, output1, output2, pooling, dropout):
        super(InceptionExit, self).__init__()
        self.fc_input = input1

        self.layer1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dropout=0)
        self.layer2 = nn.sequential(
            nn.Linear(in_features=input1, out_features=output1),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        self.layer3 = nn.sequential(
            nn.Linear(in_features=output1, out_features=output2),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        self.pool =  nn.MaxPool2d(kernel_size=pooling, ceil_mode=True),



    def forward(self, x):
        x = self.pool(x)
        x = self.layer1(x)
        x = x.view(-1, self.input1)
        x = self.layer2(x)
        x = self.layer3(x)

        return torch.sigmoid(x)


class PamflNet(nn.Module):
    def __init__(self):
        super(PamflNet, self).__init__()

        self.inception1 = InceptionModule(in_channels=1, f_1x1=64, f_3x3_r=32,
                                          f_3x3=64,
                                          f_7x7_r=32,
                                          f_7x7=64,
                                          f_11x11_r=32,
                                          f_11x11=64,
                                          f_19x19_r=32,
                                          f_19x19=64,
                                          f_pp=32,
                                          dropout=0.2)
        self.inception2 = InceptionModule(in_channels=352, f_1x1=64, f_3x3_r=32,
                                          f_3x3=64,
                                          f_7x7_r=32,
                                          f_7x7=64,
                                          f_11x11_r=32,
                                          f_11x11=64,
                                          f_19x19_r=32,
                                          f_19x19=64,
                                          f_pp=32,
                                          dropout=0.2)
        self.inception3 = InceptionModule(in_channels=352, f_1x1=64, f_3x3_r=32,
                                          f_3x3=64,
                                          f_7x7_r=32,
                                          f_7x7=64,
                                          f_11x11_r=32,
                                          f_11x11=64,
                                          f_19x19_r=32,
                                          f_19x19=64,
                                          f_pp=32,
                                          dropout=0.2)
        self.inexit = InceptionExit(in_channels=352, out_channels=256,
                                    pooling=4,
                                    input1=132000,
                                    output1=2000,
                                    output2=4,
                                    dropout=0.5)
        def forward(self, x):
            x = self.inception1(x)
            x = self.inception2(x)
            x = self.inception3(x)
            x = self.inexit(x)
            return x
