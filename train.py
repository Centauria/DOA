# _*_ coding: UTF-8 _*_
import time
import argparse
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.distributed as dist
from torch.autograd import Variable
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataset import GenDOA
from models.crnn import CRNN

plt.switch_backend('agg')

seed = np.random.randint(0, 1000)
np.random.seed(seed)
torch.random.manual_seed(seed)


def train():
    parser = argparse.ArgumentParser(prog='train', description="""Script to train a DOA estimator""")
    parser.add_argument("--datasets", "-d", required=True, help="Directory where datasets are", type=str)
    parser.add_argument("--outputs", "-o", required=True, help="Directory to write results", type=str)
    parser.add_argument("--batchsize", "-b", type=int, default=256, help="Choose a batchsize")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--loss", "-lo", type=str, choices=["cartesian", "categorical", "polar"], required=True,
                        help="Choose loss representation")

    args = parser.parse_args()
    assert os.path.exists(args.datasets)

    epochs = args.epochs
    batchsize = args.batchsize
    loss = args.loss
    foldername = '{}_batch{}'.format(args.loss, batchsize)
    outpath = os.path.join(args.outputs, foldername)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    savepath = os.path.join(outpath, 'best_model.{epoch:02d}-{val_loss:.6f}.h5')

    # read data
    # feature shape:(nBatch, nchunk, 25, 513, 6)
    train_data = GenDOA(args.datasets, loss_type=args.loss)  # 返回feature, loc  loc->azimuth
    train_data_entries, val_data_entries = train_test_split(train_data, test_size=0.3, random_state=11)
    train_loader = torch.utils.data.DataLoader(train_data_entries, batch_size=batchsize, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data_entries, batch_size=batchsize, shuffle=True, drop_last=True)

    # initialize model
    crnn = CRNN()

    # scheduler
    Divide_criterion = nn.L1Loss(reduction='sum')
    Prob_criterion = nn.BCEloss(weight=None, size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(params=crnn.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # train
    for epoch in range(epochs):
        loss1 = 0.0
        loss2 = 0.0
        scheduler.step()

        # data entries
        # feature shape:(nBatch, 1, 25, 513, 6)
        for i in range(train_loader[0].shape[1]):
            train_feature = train_loader[0][:, i, :, :, :]
            train_label = train_loader[1][:, i, :]
            # label!=0, set prob=1
            # label=0, set prob=0
            prob_train_label = torch.empty(train_label.shape)

            val_feature = val_loader[0][:, i, :, :, :]
            val_label = val_loader[1][:, i, :]
            prob_val_label = torch.empty(val_label.shape)

            train_feature, train_label, val_feature, val_label = Variable(train_feature), Variable(train_label), \
                                                                 Variable(val_feature), Variable(val_label)


            # forward, backward, update weights
            optimizer.zero_grad()
            predictions = crnn(train_feature)
            coords, prob = zip(*predictions)
            loss1 = Divide_criterion(coords, train_label)
            loss2 = Prob_criterion(prob, prob_train_label)
            # compute loss
            loss =

            loss.backward()
            optimizer.step()




if __name__ == "__main__":
    train()