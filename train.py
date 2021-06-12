# _*_ coding: UTF-8 _*_
import time
import argparse
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.distributed as dist
from torch.autograd import Variable
from datetime import datetime

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = np.random.randint(0, 1000)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)


def PIT(doa, target, loc, prob):
    """
    get PIT doa
    :param doa: predict doa of speakers  [nBatch, 6]
    :param target: predict prob of speakers  [nBatch, 6]
    :param loc: label of speakers  [nBatch, 6]
    :param prob: probability of speakers  [nBatch, 6]
    :return: active doa after pit  [nBatch, 6]
    """
    Divide_criterion = nn.MSELoss(reduction='mean')
    Prob_criterion = nn.BCEloss(weight=None, size_average=None, reduce=None, reduction='mean')

    if doa.shape[0] == target.shape[0]:
        nBatch = doa.shape[0]
    active_speakers_number = (target == 1).sum(dim=1)  # active number of each batch  [nBatch]
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
        for p in permutations(doa[iBatch], int(
                active_speakers_number[iBatch])):  # A(6,active_speakers_number[iBatch]){doa[iBatch]}=cnt
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
            pit_target_loss.append(pit_target)  # len(pit_target_loss)=len(pit_doa_loss)=cnt
            ip += 1
        pit_loss = torch.add(torch.tensor(pit_doa_loss),
                             torch.tensor(pit_target_loss))  # tensorlenth(pit_loss)=cnt for iBatch
        pit_index = torch.argmin(pit_loss)  # active_doa index for  iBatch
        active_doa[iBatch] = true_active_doa[pit_index]
    return active_doa


def train():
    parser = argparse.ArgumentParser(prog='train', description="""Script to train a DOA estimator""")
    parser.add_argument("--train_datasets", "-t", required=True, help="Directory where train datasets are", type=str)
    parser.add_argument("--val_datasets", "-v", required=True, help="Directory where valid datasets are", type=str)
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
    train_datasets = args.train_datatsets
    val_datasets = args.val_datasets
    foldername = '{}_batch{}'.format(args.loss, batchsize)
    outpath = os.path.join(args.outputs, foldername)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    savepath = os.path.join(outpath, 'best_model.{epoch:02d}-{val_loss:.6f}.h5')

    # read data input
    # feature shape:(Batch, 6, 25, 513)
    # loc shape:(Batch, 6)
    # prob shape :(Batch, 6)
    train_data = GenDOA(train_datasets, loss_type=args.loss)  # feature, loc, prob
    val_data  = GenDOA(val_datasets, loss_type=args.loss)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batchsize, shuffle=True, drop_last=True)

    # initialize model
    crnn = CRNN().to(device)

    train_loss = []
    valid_loss = []
    min_valid_loss = np.inf

    # scheduler
    Divide_criterion = nn.MSELoss(reduction='mean')
    Prob_criterion = nn.BCEloss(weight=None, size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(params=crnn.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # train
    for epoch in range(epochs):
        total_train_loss = []
        crnn.train()
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

            # forward, backward, update weights
            optimizer.zero_grad()
            predictions = crnn(Variable(feature))
            doa, target = zip(*predictions)
            # doa_loss = Divide_criterion(doa, loc)
            # target_loss = Prob_criterion(target, prob)
            # compute loss
            # loss = doa_loss + target_loss

            active_doa = PIT(doa, target, loc, prob)

            loss1 = Divide_criterion(Variable(active_doa).float(), Variable(loc).float())
            loss2 = Prob_criterion(Variable(target).float(), Variable(prob).float())
            loss = torch.add(loss1, loss2)  # adjust alpha

            loss.backward()
            optimizer.step()
            total_train_loss.append(loss.item())

        # record train_loss
        train_loss.append(np.mean(total_train_loss))

        # validate
        total_valid_loss = []
        crnn.eval()
        for i, val_data in enumerate(val_loader):
            feature, loc, prob = val_data
            with torch.no_grad():
                prediction = crnn(Variable(feature))  # rnn output
                doa, target = zip(*prediction)
            active_doa = PIT(doa, target, loc, prob)
            loss1 = Divide_criterion(Variable(active_doa).float(), Variable(loc).float())
            loss2 = Prob_criterion(Variable(target).float(), Variable(prob).float())
            loss = torch.add(loss1, loss2)
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))

        if (valid_loss[-1] < min_valid_loss):
            torch.save({'epoch': i, 'model': crnn, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, './LSTM.model')
            #         torch.save(optimizer, './crnn.optim')
            min_valid_loss = valid_loss[-1]

        log_string = ('iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, '
                      'best_valid_loss: {:0.6f}, lr: {:0.7f}').format((epoch + 1), epochs,
                                                                      train_loss[-1],
                                                                      valid_loss[-1],
                                                                      min_valid_loss,
                                                                      optimizer.param_groups[0]['lr'])
        scheduler.step()
        print(str(datetime.datetime.now() + datetime.timedelta(hours=8)) + ': ')
        print(log_string)


if __name__ == "__main__":
    train()