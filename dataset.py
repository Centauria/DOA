# -*- coding: utf-8 -*-
import json
import os

import librosa
import numpy as np
import ruamel_yaml as yaml
import torch
import torch.utils.data
from intervaltree import IntervalTree
from tqdm.auto import tqdm

import util


class GenDOA(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split='train', feature_path=None, loss_type='xpolar', pad_strategy='zero'):
        self.__path = dataset_path
        self.__split = split
        meta_path = os.path.join(dataset_path, 'meta', split, 'records')
        if os.path.isdir(meta_path):
            self.__meta_path = meta_path
        else:
            raise FileNotFoundError(f'Meta folder {meta_path} does not exist')
        records = os.listdir(meta_path)
        self.file_indexes = list(map(lambda s: os.path.splitext(s)[0], records))
        feature_path = os.path.join(dataset_path, 'features', split) if feature_path is None else feature_path
        if os.path.isdir(feature_path):
            self.__feature_path = feature_path
        else:
            raise FileNotFoundError(f'Feature folder {feature_path} does not exist')
        wav_path = os.path.join(dataset_path, 'wav', split)
        if os.path.isdir(wav_path):
            self.__wav_path = wav_path
        else:
            raise FileNotFoundError(f'Wave folder {wav_path} does not exist')
        room_path = os.path.join(dataset_path, 'meta', split, 'rooms')
        if os.path.isdir(room_path):
            self.__room_path = room_path
        else:
            raise FileNotFoundError(f'Room folder {room_path} does not exist')
        assert loss_type in ('categorical', 'cartesian', 'polar', 'xpolar')
        self.__loss_type = loss_type
        assert pad_strategy in ('zero', 'minus', 'select', 'generate')
        self.__pad_strategy = pad_strategy
        self.indexes = IntervalTree()
        n = 0
        for name in tqdm(self.file_indexes, leave=True):
            slices, _ = self.slice_tracks(name)
            s_num = len(slices)
            self.indexes[n:n + s_num] = name
            n += s_num

    def __getitem__(self, item):
        index = sorted(self.indexes[item])
        if isinstance(item, int):
            itv = index[0]
            feature, loc, prob = self.get(itv.data, item - itv.begin)
            feature = torch.tensor(feature).permute(2, 0, 1)
        elif isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            item = range(start, stop, step)
            item_info = map(lambda i: next(filter(lambda x: x.begin <= i < x.end, index)), item)
            item_info = map(lambda x: (x[0].data, x[1] - x[0].begin), zip(item_info, item))
            feature, loc, prob = list(zip(*map(lambda x: self.get(*x), item_info)))
            loc = torch.stack(loc)
            prob = torch.stack(prob)
            feature = torch.tensor(feature).permute(0, 3, 1, 2)
        else:
            raise ValueError('Index must be int or slice')
        return feature, loc, prob

    def __len__(self):
        return self.indexes.end() - self.indexes.begin()

    def get(self, filename, num):
        feature = self.feature(filename)[num]
        loc, prob = self.slice_tracks(filename)
        loc, prob = loc[num], prob[num]
        locs = self.locations(filename)
        loc = list(map(lambda x: locs[x], loc))
        room = self.room(filename)
        if self.__loss_type == 'cartesian':
            loc = util.cartesian(room['mic_array_location'], loc)
        elif self.__loss_type == 'polar':
            loc = util.polar(room['mic_array_location'], loc)
        elif self.__loss_type == 'xpolar':
            loc = util.x_linear_polar(room['mic_array_location'], loc)
        else:
            raise ValueError(f'Loss type {self.__loss_type} not defined')
        loc = torch.tensor(loc)
        prob = torch.ones_like(loc)
        if loc.shape[-1] < 6:
            pad = torch.zeros(6 - loc.shape[-1])
            loc = torch.cat((loc, pad), dim=-1)
            prob = torch.cat((prob, pad), dim=-1)
        return feature, loc, prob

    def wave(self, filename):
        y, sr = librosa.load(os.path.join(self.__wav_path, filename + '.wav'), sr=None, mono=False)
        return y, sr

    def feature(self, filename):
        return np.load(os.path.join(self.__feature_path, filename + '.npy'))

    def meta(self, filename):
        with open(os.path.join(self.__meta_path, filename + '.json')) as f:
            meta = json.load(f)
        return meta

    def room(self, filename):
        a, b, c, _ = filename.split('-')
        room_filename = '-'.join((a, b, c))
        with open(os.path.join(self.__room_path, room_filename + '.yml')) as f:
            room = yaml.safe_load(f)
        return room

    def locations(self, filename):
        meta = self.meta(filename)
        room = self.room(filename)
        return dict(zip(meta.keys(), room['sources_location']))

    def slice_tracks(self, filename, hop=25, sample_rate=16000, fft_size=1024, overlap_size=512):
        hop_size = fft_size - overlap_size
        time_per_frame = fft_size / sample_rate
        hop_time = hop * hop_size / sample_rate
        track_info = self.meta(filename)
        ts = []
        ei = event_intervals(track_info)
        mt = max_end_time(track_info)
        t = 0
        while t <= mt:
            ts.append(ei[t:t + hop_time + time_per_frame])
            t += hop_time
        lo = [list(set(map(lambda x: x.data, ss))) for ss in ts]
        lengths = list(map(len, lo))
        probs = [[1] * n + [0] * (6 - n) for n in lengths]
        return lo, probs


def max_end_time(track_info):
    end_time = 0
    for speaker, wave_clips in track_info.items():
        for wc in wave_clips:
            end_time = wc['end_time'] if wc['end_time'] > end_time else end_time
    return end_time


def event_intervals(track_info):
    tracks = IntervalTree()
    for k, v in track_info.items():
        for wc in v:
            tracks[wc['start_time']:wc['end_time']] = k
    return tracks


def pad_location(location, length=6, strategy='zero'):
    loc_len = len(location)
    if loc_len < length:
        pad_len = length - loc_len
        if strategy == 'zero':
            pad = [[0, 0, 0]] * pad_len
        elif strategy == 'minus':
            pad = [[-1, -1, -1]] * pad_len
        elif strategy == 'select':
            # TODO: select locations in dataset
            pad = []
        elif strategy == 'generate':
            # TODO: generate locations from scratch
            pad = []
        else:
            raise ValueError('Argument "strategy" must be in (zero, minus, select, generate)')
        locs = location + pad
    else:
        locs = location
    return locs
