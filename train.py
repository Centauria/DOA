# _*_ coding: UTF-8 _*_
import argparse
import os
from datetime import datetime
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from scipy.special import perm

from dataset import GenDOA, DataLoaderX
from models.crnn import CRNN

plt.switch_backend('agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = np.random.randint(0, 1000)
np.random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)


def PIT(doa, target, loc, prob, divide_criterion, prob_criterion):
    """
    get PIT doa
    :param doa: predict doa of speakers  [nBatch, 6]
    :param target: predict prob of speakers  [nBatch, 6]
    :param loc: label of speakers  [nBatch, 6]
    :param prob: probability of speakers  [nBatch, 6]
    :param divide_criterion: criterion for divide network
    :param prob_criterion: criterion for probability network
    :return: active doa after pit  [nBatch, 6]
    """

    doa = doa.to(device)
    target = target.to(device)
    loc = loc.to(device)
    prob = prob.to(device)
    nBatch = doa.shape[0]
    active_speakers_number = (target == 1).sum(dim=1)  # active number of each batch  [nBatch]
    count = perm(6, active_speakers_number.cpu())  # [nBatch]
    active_doa = torch.empty((nBatch, 6), device=device)
    for iBatch in range(nBatch):
        cnt = int(count[iBatch])  # int(A(6, n))
        pad_loc = torch.repeat_interleave(loc[iBatch].unsqueeze(dim=0), repeats=cnt, dim=0)  # [cnt, 6]
        pad_prob = torch.repeat_interleave(prob[iBatch].unsqueeze(dim=0), repeats=cnt, dim=0)  # [cnt, 6]
        true_active_doa = torch.empty((cnt, 6), device=device)  # [cnt, 6]
        true_active_target = torch.repeat_interleave(target[iBatch].unsqueeze(dim=0), repeats=cnt, dim=0)  # [cnt, 6]
        # pad_loc.to(device)
        # pad_prob.to(device)
        # true_active_target.to(device)
        pit_doa_loss = []
        pit_target_loss = []
        ip = 0
        for p in permutations(doa[iBatch], int(active_speakers_number[iBatch])):
            # A(6,active_speakers_number[iBatch]){doa[iBatch]}=cnt
            p = torch.tensor(list(p))  # p:one sequence in A(6, n), len(p)=n
            index = 0
            for i, probability in enumerate(target[iBatch]):
                if probability == 1:  # fill (target != 0) with sequence in permutations
                    true_active_doa[ip][i] = p[index]
                    index += 1
                else:
                    true_active_doa[ip][i] = 0
            pit_doa = divide_criterion(true_active_doa[ip].float(), pad_loc[ip].float())
            pit_target = prob_criterion(true_active_target[ip].float(), pad_prob[ip].float())
            pit_doa_loss.append(pit_doa)  # list
            pit_target_loss.append(pit_target)  # len(pit_target_loss)=len(pit_doa_loss)=cnt
            ip += 1
        pit_loss = torch.add(torch.tensor(pit_doa_loss),
                             torch.tensor(pit_target_loss))  # tensor length(pit_loss)=cnt for iBatch
        pit_index = torch.argmin(pit_loss)  # active_doa index for  iBatch
        active_doa[iBatch] = true_active_doa[pit_index]
    return active_doa


def train():
    parser = argparse.ArgumentParser(prog='train', description="""Script to train a DOA estimator""")
    parser.add_argument("--dataset", "-d", required=True, help="Directory where dataset are", type=str)
    parser.add_argument("--outputs", "-o", required=True, help="Directory to write results", type=str)
    parser.add_argument("--batch_size", "-b", type=int, default=256, help="Choose batch size")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--loader-workers", "-w", type=int, default=8, help="Threads used in Dataloader")
    parser.add_argument("--loss", "-lo", type=str, choices=["cartesian", "polar", "xpolar"], default="xpolar",
                        help="Choose loss representation")

    args = parser.parse_args()
    assert os.path.exists(args.dataset)

    epochs = args.epochs
    workers = args.loader_workers
    batch_size = args.batch_size
    loss_type = args.loss
    dataset = args.dataset
    foldername = f'{loss_type}_batch{batch_size}'
    outpath = os.path.join(args.outputs, foldername)
    os.makedirs(outpath, exist_ok=True)
    savepath = os.path.join(outpath, 'best_model.{epoch:02d}-{val_loss:.6f}.h5')

    # read data input
    # feature shape:(Batch, 6, 25, 513)
    # loc shape:(Batch, 6)
    # prob shape :(Batch, 6)
    train_data = GenDOA(dataset, split='train', loss_type=loss_type)
    val_data = GenDOA(dataset, split='test', loss_type=loss_type)
    train_loader = DataLoaderX(
        train_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True
    )
    val_loader = DataLoaderX(
        val_data, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True
    )

    # initialize model
    model = CRNN().to(device)

    train_loss = []
    valid_loss = []
    min_valid_loss = np.inf

    # scheduler
    divide_criterion = nn.MSELoss(reduction='mean')
    prob_criterion = nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # train
    for epoch in range(epochs):
        total_train_loss = []
        model.train()
        lr = optimizer.param_groups[0]['lr']
        print(f'# epoch {epoch}, lr={lr}')

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
            doa, target = model(feature.to(device))

            active_doa = PIT(doa, target, loc, prob, divide_criterion, prob_criterion)

            loss1 = divide_criterion(active_doa.float(), loc.float().to(device))
            loss2 = prob_criterion(target.float(), prob.float().to(device))
            print(f'epoch {epoch} step {i}: loss1={loss1}, loss2={loss2}')
            loss = torch.add(loss1, loss2)  # adjust alpha

            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss.append(loss.item())

        # record train_loss
        train_loss.append(np.mean(total_train_loss))

        # validate
        total_valid_loss = []
        model.eval()
        for i, val_data in enumerate(val_loader):
            feature, loc, prob = val_data
            with torch.no_grad():
                doa, target = model(feature.to(device))
            active_doa = PIT(doa, target, loc, prob, divide_criterion, prob_criterion)
            loss1 = divide_criterion(active_doa.float(), loc.float().to(device))
            loss2 = prob_criterion(target.float(), prob.float().to(device))
            loss = torch.add(loss1, loss2)
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))

        if valid_loss[-1] < min_valid_loss:
            torch.save({'epoch': epoch, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss}, os.path.join(outpath, 'LSTM.model'))
            #         torch.save(optimizer, './crnn.optim')
            min_valid_loss = valid_loss[-1]

        log_string = 'iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, ' \
                     'best_valid_loss: {:0.6f}, lr: {:0.7f}' \
            .format(epoch + 1, epochs, train_loss[-1], valid_loss[-1], min_valid_loss, lr)
        scheduler.step()
        print(str() + f'{datetime.now()}: {log_string}')


if __name__ == "__main__":
    train()
