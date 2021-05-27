# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


def CNN_block(in_channels: int, out_channels: int, kernel_size=3, padding=None, batch_norm=True, max_pool_size=None):
    if padding is not None:
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
    else:
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.ReLU(inplace=True)
        )
    if batch_norm:
        block.add_module('bn', nn.BatchNorm2d(out_channels))
    if max_pool_size is not None:
        block.add_module('max_pool', nn.MaxPool2d(max_pool_size))
    return block


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            CNN_block(6, 64, max_pool_size=(1, 8), padding=(1, 2)),
            CNN_block(64, 64, max_pool_size=(1, 8), padding=(1, 4)),
            CNN_block(64, 64, max_pool_size=(1, 4), padding=(1, 2)),
        )
        self.rnn = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.prob = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
            nn.Sigmoid()
        )
        self.divide = nn.ModuleList([nn.Sequential(
            CNN_block(1, 12, padding=(1, 1), max_pool_size=(1, 8)),
            CNN_block(12, 6, padding=(1, 1), max_pool_size=(4, 4)),
            CNN_block(6, 1, padding=(1, 1))
        ) for i in range(6)])
        self.output = nn.ModuleList([nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ) for i in range(6)])

    def forward(self, x):
        # x: [batch, 25, 513, 6]
        x = self.cnn(x)
        # x: [batch, 64, 25, 2]
        x.transpose_(1, 2)
        x = x.flatten(start_dim=2)
        # x: [batch, 25, 128]
        x.transpose_(0, 1)
        x, _ = self.rnn(x)
        # x: [25, batch, 128]
        x.transpose_(0, 1)
        # x: [batch, 25, 128]

        prob = self.prob(x)
        # prob: [batch, 25, 6]
        prob = prob.max(dim=1).values
        # prob: [batch, 6]

        x.unsqueeze_(1)
        # x: [batch, 1, 25, 128]
        coords = list(map(lambda m: m(x), self.divide))
        # coords : 6 * [batch, 1, 6, 4]
        for i in range(6):
            coords[i] = coords[i].flatten(start_dim=2)
            coords[i].squeeze_(1)
        # coords : 6 * [batch, 24]
        coords = list(map(lambda i: self.output[i](coords[i]), range(6)))
        # coords : 6 * [batch, 1, 1]
        coord = torch.cat(coords, dim=1)
        # coord: [batch, 6, 1]
        coord.squeeze_(-1)
        # coord: [batch, 6]
        return coord, prob
