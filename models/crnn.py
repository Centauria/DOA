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
            nn.Linear(128, 429),
            nn.Linear(429, 2)
        )

    def forward(self, x):
        # x: [batch, 25, 513, 6]
        x = self.cnn(x)
        x = x.view(x.shape[0], -1, 128)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
