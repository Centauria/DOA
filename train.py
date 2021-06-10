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
from scipy.special import comb,perm

plt.switch_backend('agg')

seed = np.random.randint(0, 1000)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)


def PIT_loss(doa, target, n, train_label, prob_train_label):
    """
    compute PIT_loss = min_{A(6. n)}(MSE + alpha*CE)
    :param doa: (nBatch, 6)  predict DOA of speakers # (x1, x2, x3, x4, x5, x6)
    :param target: (nBatch, 6)  predict prob of speakers # (0, 0, 1, 0, 1, 1)
    :param n: number of true speakers
    :param train_label: true DOA of speakers, to compute MSE(doa, train_label)  train_label:(nBatch, 6)——>（nBatch, count ,6）
    :param prob_train_label: true prob of speakers, to compute CE(target, prob_train_label)  prob_train_label: (nBatch, 6)——>（nBatch, count ,6）
    :return: loss  float
    """
    alpha =
    loss = []
    if doa.shape[0] == target.shape[0]:
        nBatch = doa.shape[0]

    count = int(perm(6, n))
    true_doa = torch.empty((nBatch, count, 6))
    for iBatch in range(nBatch):
        for cnt in range(count):
            for p in permutations(doa[iBatch], n):
                p = torch.tensor(list(p)) # p: one sequence in A(6,n), len(p)=n
                for i, prob in enumerate(target):
                    index = 0  # true_doa_index in p, default max_index=n?
                    if prob != 0:  # ==1?
                        true_doa[iBatch][cnt][i] = p[index]
                        index += 1
                    else:
                        true_doa[iBatch][cnt][i] = 0






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

    # read data
    # feature shape:(nBatch, nchunk, 25, 513, 6)
    # loc shape:(nBatch, nchunk, 6)
    train_data = GenDOA(args.datasets, loss_type=args.loss)  # feature, loc
    train_data_entries, val_data_entries = train_test_split(train_data, test_size=0.3, random_state=11)
    train_loader = torch.utils.data.DataLoader(train_data_entries, batch_size=batchsize, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data_entries, batch_size=batchsize, shuffle=True, drop_last=True)

    # initialize model
    crnn = CRNN()

    # scheduler
    Divide_criterion = nn.MSEloss(reduction='mean')
    Prob_criterion = nn.BCEloss(weight=None, size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(params=crnn.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # train
    for epoch in range(epochs):
        scheduler.step()

        # data entries
        # feature shape:(nBatch, 1, 25, 513, 6)
        # loc shape:(nBatch, 1, 6)
        feature_train, loc_train = train_loader
        feature_val, loc_val = val_loader
        for iChunk in range(feature_train.shape[1]):
            train_feature = feature_train[:, iChunk, :, :, :]
            train_label = loc_train[:, iChunk, :]
            # label!=0, set prob=1
            # label=0, set prob=0
            prob_train_label = torch.empty(train_label.shape)
            for i in range(6):
                if train_label[:, iChunk, i] != 0:
                    prob_train_label[:, iChunk, i] = 1
                else:
                    prob_train_label[:, iChunk, i] = 0
            val_feature = feature_val[:, iChunk, :, :, :]
            val_label = loc_val[:, iChunk, :]
            prob_val_label = torch.empty(val_label.shape)
            for i in range(6):
                if prob_val_label[:, iChunk, i] != 0:
                    prob_val_label[:, iChunk, i] = 1
                else:
                    prob_val_label[:, iChunk, i] = 0

            train_feature, train_label, val_feature, val_label = Variable(train_feature), Variable(train_label), \
                                                                 Variable(val_feature), Variable(val_label)
            prob_train_label, prob_val_label = Variable(prob_train_label), Variable(prob_val_label)

            # forward, backward, update weights
            optimizer.zero_grad()
            predictions = crnn(train_feature)
            coord, prob = zip(*predictions)
            doa_loss = Divide_criterion(coord, train_label)
            target_loss = Prob_criterion(prob, prob_train_label)
            # compute loss
            loss =

            loss.backward()
            optimizer.step()




if __name__ == "__main__":
    train()