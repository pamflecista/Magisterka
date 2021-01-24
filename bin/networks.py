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
        paddings = [int((w - 1) / 2) for w in kernel_widths]
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
        paddings = [int((w - 1) / 2) for w in kernel_widths]
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
        paddings = [int((w - 1) / 2) for w in kernel_widths]
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


#####################################################################################################

class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class InceptionModule(torch.nn.Module):

    def __init__(self, in_channels, f_1x1, f_5x5_r, f_5x5,
                 f_11x11_r, f_11x11, f_19x19_r, f_19x19, f_pp, dropout):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_1x1, kernel_size=1, stride=1, padding=0, dropout=dropout)
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_5x5_r, kernel_size=1, stride=1, padding=0, dropout=0),
            ConvBlock(f_5x5_r, f_5x5, kernel_size=5, stride=1, padding=2, dropout=dropout)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, f_11x11_r, kernel_size=1, stride=1, padding=0, dropout=0),
            ConvBlock(f_11x11_r, f_11x11, kernel_size=11, stride=1, padding=5, dropout=dropout)
        )

        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, f_19x19_r, kernel_size=1, stride=1, padding=0, dropout=0),
            ConvBlock(f_19x19_r, f_19x19, kernel_size=19, stride=1, padding=9, dropout=dropout)
        )

        self.branch5 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1, ceil_mode=True),
            ConvBlock(in_channels, f_pp, kernel_size=1, stride=1, padding=0, dropout=0)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch5 = self.branch5(x)

        return torch.cat([branch1, branch2, branch3, branch4, branch5], 1)


class InceptionExit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, input1, output1, output2, pooling, dropout):
        super(InceptionExit, self).__init__()
        self.fc_input = input1

        self.layer1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dropout=0)
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=input1, out_features=output1),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=output1, out_features=output2),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        self.pool = nn.MaxPool2d(kernel_size=pooling, ceil_mode=True)

    def forward(self, x):
        x = self.pool(x)
        x = self.layer1(x)
        x = x.view(-1, self.fc_input)
        x = self.layer2(x)
        x = self.layer3(x)

        return torch.sigmoid(x)


class PamflNet(torch.nn.Module):
    def __init__(self, seq_len):
        super(PamflNet, self).__init__()

        self.params = {
            'input sequence length': seq_len,
            'convolutional layers': 3,
            'fully connected': 2,
            'number of channels': 'test',
            'kernels widths': 'test',
            'pooling widths': 'test',
            'units in fc': 'test',
            'dropout': 'test'

        }

        self.inception1 = InceptionModule(in_channels=1, f_1x1=8,
                                          f_5x5_r=4,
                                          f_5x5=8,
                                          f_11x11_r=4,
                                          f_11x11=8,
                                          f_19x19_r=4,
                                          f_19x19=8,
                                          f_pp=8,
                                          dropout=0.2)

        self.inception2 = InceptionModule(in_channels=40, f_1x1=8,
                                          f_5x5_r=4,
                                          f_5x5=8,
                                          f_11x11_r=4,
                                          f_11x11=8,
                                          f_19x19_r=4,
                                          f_19x19=8,
                                          f_pp=8,
                                          dropout=0.2)

        self.inexit = InceptionExit(in_channels=40, out_channels=16,
                                    pooling=4,
                                    output1=2000,
                                    output2=4,
                                    input1=8000,
                                    dropout=0.5)

    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inexit(x)
        return x


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class InceptionModule(torch.nn.Module):

    def __init__(self, in_channels, f_1x1, f_5x5_r, f_5x5,
                 f_11x11_r, f_11x11, f_19x19_r, f_19x19, f_pp, dropout, paddings, kernels):
        super(InceptionModule, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBlock(in_channels, f_1x1, kernel_size=kernels[0], stride=1, padding=paddings[0], dropout=dropout)
        )

        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, f_5x5_r, kernel_size=1, stride=1, padding=0, dropout=0),
            ConvBlock(f_5x5_r, f_5x5, kernel_size=kernels[1], stride=1, padding=paddings[1], dropout=dropout)
        )

        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, f_11x11_r, kernel_size=1, stride=1, padding=0, dropout=0),
            ConvBlock(f_11x11_r, f_11x11, kernel_size=kernels[2], stride=1, padding=paddings[2], dropout=dropout)
        )

        self.branch4 = nn.Sequential(
            ConvBlock(in_channels, f_19x19_r, kernel_size=1, stride=1, padding=0, dropout=0),
            ConvBlock(f_19x19_r, f_19x19, kernel_size=kernels[3], stride=1, padding=paddings[3], dropout=dropout)
        )

        self.branch5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernels[4], stride=1, padding=paddings[4], ceil_mode=True),
            ConvBlock(in_channels, f_pp, kernel_size=1, stride=1, padding=0, dropout=0)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch5 = self.branch5(x)

        return torch.cat([branch1, branch2, branch3, branch4, branch5], 1)


class InceptionExit(torch.nn.Module):

    def __init__(self, in_channels, out_channels, input1, output1, output2,  dropout):
        super(InceptionExit, self).__init__()
        self.fc_input = input1

        self.layer1 = ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dropout=0)
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=input1, out_features=output1),
            nn.ReLU(),
            nn.Dropout(p=dropout))
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=output1, out_features=output2),
            nn.ReLU(),
            nn.Dropout(p=dropout))

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1, self.fc_input)
        x = self.layer2(x)
        x = self.layer3(x)

        return torch.sigmoid(x)


class PamflNet(torch.nn.Module):
    def __init__(self, seq_len):
        super(PamflNet, self).__init__()

        self.params = {
            'input sequence length': seq_len,
            'convolutional layers': 3,
            'fully connected': 2,
            'number of channels': 'test',
            'kernels widths': 'test',
            'pooling widths': 'test',
            'units in fc': 'test',
            'dropout': 'test'

        }
        kernels1 = [(4, 1), (4, 5), (4, 11), (4, 19), (4, 3)]
        paddings1 = [(0, 0), (0, 2), (0, 5), (0, 9), (0, 1)]
        f_in1 = [1, 1, 1]
        f_out1 = [300, 300, 300, 300, 300]
        f_in2 = [100, 100, 100]
        f_out2 = [200, 200, 200, 200, 200]
        kernels2 = [(1, 1), (1, 5), (1, 11), (1, 19), (1, 3)]
        paddings2 = [(0, 0), (0, 2), (0, 5), (0, 9), (0, 1)]

        f_in3 = [100, 100, 100]
        f_out3 = [200, 200, 200, 200, 200]
        kernels3 = [(1, 1), (1, 5), (1, 11), (1, 19), (1, 3)]
        paddings3 = [(0, 0), (0, 2), (0, 5), (0, 9), (0, 1)]
        self.inception1 = InceptionModule(in_channels=1, f_1x1=f_out1[0],
                                          f_5x5_r=f_in1[0],
                                          f_5x5=f_out1[1],
                                          f_11x11_r=f_in1[1],
                                          f_11x11=f_out1[2],
                                          f_19x19_r=f_in1[2],
                                          f_19x19=f_out1[3],
                                          f_pp=f_out1[4],
                                          dropout=0.2,
                                          paddings=paddings1,
                                          kernels=kernels1)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 4), ceil_mode=True)


        self.inception2 = InceptionModule(in_channels=sum(f_out1), f_1x1=f_out2[0],
                                          f_5x5_r=f_in2[0],
                                          f_5x5=f_out2[1],
                                          f_11x11_r=f_in2[1],
                                          f_11x11=f_out2[2],
                                          f_19x19_r=f_in2[2],
                                          f_19x19=f_out2[3],
                                          f_pp=f_out2[4],
                                          dropout=0.2,
                                          paddings=paddings2,
                                          kernels=kernels2)

        self.inception3 = InceptionModule(in_channels=sum(f_out2), f_1x1=f_out3[0],
                                          f_5x5_r=f_in3[0],
                                          f_5x5=f_out3[1],
                                          f_11x11_r=f_in3[1],
                                          f_11x11=f_out3[2],
                                          f_19x19_r=f_in3[2],
                                          f_19x19=f_out3[3],
                                          f_pp=f_out3[4],
                                          dropout=0.2,
                                          paddings=paddings2,
                                          kernels=kernels2)

        self.inexit = InceptionExit(in_channels=sum(f_out2), out_channels=200,
                                    output1=2000,
                                    output2=4,
                                    input1=25000,
                                    dropout=0.2)

    def forward(self, x):
        x = self.inception1(x)
        x = self.pool1(x)
        x = self.inception2(x)
        x = self.pool1(x)
        x = self.inexit(x)
        return x
