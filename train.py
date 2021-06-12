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

from itertools import permutations
from scipy.special import perm

plt.switch_backend('agg')

seed = np.random.randint(0, 1000)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)


def train():
    parser = argparse.ArgumentParser(prog='train', description="""Script to train a DOA estimator""")
    parser.add_argument("--datasets", "-d", required=True, help="Directory where datasets are", type=str)
    parser.add_argument("--outputs", "-o", required=True, help="Directory to write results", type=str)
    parser.add_argument("--batchsize", "-b", type=int, default=256, help="Choose a batchsize")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--loss", "-lo", type=str, choices=["cartesian", "polar"], required=True,
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

    # read data input
    # feature shape:(Batch, 6, 25, 513)
    # loc shape:(Batch, 6)
    # prob shape :(Batch, 6)
    train_data = GenDOA(args.datasets, loss_type=args.loss)  # feature, loc, prob
    train_data_entries, val_data_entries = train_test_split(train_data, test_size=0.3, random_state=11)
    train_loader = torch.utils.data.DataLoader(train_data_entries, batch_size=batchsize, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data_entries, batch_size=batchsize, shuffle=True, drop_last=True)

    # initialize model
    crnn = CRNN()

    # scheduler
    Divide_criterion = nn.MSELoss(reduction='mean')
    Prob_criterion = nn.BCEloss(weight=None, size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(params=crnn.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # train
    for epoch in range(epochs):
        scheduler.step()

        # batch data entries
        # feature shape:(nBatch, 6, 25, 513)
        # loc shape:(nBatch, 6)
        # prob shape:(nBatch, 6)
        for i, train_data in enumerate(train_loader):
            feature, loc, prob = train_data
            # feature: (batchsize, 6, 25, 513)
            # loc: (batchsize, 6)
            # prob: (batchsize, 6)
            feature, loc, prob = Variable(feature), Variable(loc), Variable(prob)

            # forward, backward, update weights
            optimizer.zero_grad()
            predictions = crnn(feature)
            doa, target = zip(*predictions)
            # doa_loss = Divide_criterion(doa, loc)
            # target_loss = Prob_criterion(target, prob)
            # compute loss
            # loss = doa_loss + target_loss

            # PIT loss
            # get PIT doa, target
            # doa: predict doa of speakers  [nBatch, 6]
            # target: predict prob of speakers  [nBatch, 6]
            if doa.shape[0] == target.shape[0]:
                nBatch = doa.shape[0]
            active_speakers_number = (target==1).sum(dim=1)  # active number of each batch  [nBatch]
            count = perm(6, active_speakers_number)  # [nBatch]
            active_doa = torch.empty((nBatch, 6))
            for iBatch in range(nBatch):
                cnt = int(count[iBatch])  # int(A(6, n))
                pad_loc = torch.repeat_interleave(loc[iBatch].unsqueeze(dim=0), repeats=cnt, dim=0)  # [cnt, 6]
                pad_prob = torch.repeat_interleave(prob[iBatch].unsqueeze(dim=0), repeats=cnt, dim=0)  # [cnt, 6]
                true_active_doa = torch.empty((cnt, 6))  # [cnt, 6]
                true_active_target = torch.repeat_interleave(target[iBatch].unsqueeze(dim=0), repeats=cnt, dim=0)  # [cnt, 6]
                pit_doa_loss = []
                pit_target_loss = []
                ip = 0
                for p in permutations(doa[iBatch], int(active_speakers_number[iBatch])):  # A(6,active_speakers_number[iBatch]){doa[iBatch]}=cnt
                    p = torch.tensor(list(p))  # p:one sequence in A(6, n), len(p)=n
                    for i, probility in enumerate(target[iBatch]):
                        index = 0
                        if probility == 1:  # fill (target != 0) with sequence in permutations
                            true_active_doa[ip][i] = p[index]
                        else:
                            true_active_doa[ip][i] = 0
                    pit_doa = Divide_criterion(Variable(true_active_doa[ip]).float(), Variable(pad_loc[ip]).float())
                    pit_target = Prob_criterion(Variable(true_active_target[ip]).float(), Variable(pad_prob[ip]).float())
                    pit_doa_loss.append(pit_doa)  # list
                    pit_target_loss.append(pit_target)   # len(pit_target_loss)=len(pit_doa_loss)=cnt
                    ip += 1
                pit_loss = torch.add(torch.tensor(pit_doa_loss), torch.tensor(pit_target_loss))   # tensorlenth(pit_loss)=cnt for iBatch
                pit_index = torch.argmin(pit_loss)  # active_doa index for  iBatch
                active_doa[iBatch] = true_active_doa[pit_index]

            loss1 = Divide_criterion(Variable(active_doa).float(), Variable(loc).float())
            loss2 = Prob_criterion(Variable(target).float(), Variable(prob).float())
            loss = torch.add(loss1, loss2)  # adjust alpha

            loss.backward()
            optimizer.step()

    # print loss

    # valid


if __name__ == "__main__":
    train()