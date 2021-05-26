# -*- coding: utf-8 -*-
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
        self.fc = nn.Sequential(
            nn.Linear(25 * 128, 429),
            nn.ReLU(),
            nn.Linear(429, 6 * 1)
        )
        self.prob = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [batch, 25, 513, 6]
        x = self.cnn(x)
        # x: [batch, 64, 25, 2]
        x = x.transpose(1, 2)
        x = x.flatten(start_dim=2)
        # x: [batch, 25, 128]
        x = x.transpose(0, 1)
        x, _ = self.rnn(x)
        # x: [25, batch, 128]
        x = x.transpose(0, 1)

        prob = self.prob(x)
        # prob: [batch, 25, 6]
        prob = prob.max(dim=1).values
        # prob: [batch, 6]

        coord = x.view(x.shape[0], -1)
        coord = self.fc(coord)
        # x: [batch, 6]
        return coord, prob
