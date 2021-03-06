# -*- coding: utf-8 -*-
import json
import os
import random

import librosa
import numpy as np
import ruamel_yaml as yaml
import torch
import torch.nn.functional as F
import torch.utils.data
import zmq
from intervaltree import IntervalTree
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import util


class GenDOA(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split='train',
                 output_dim=6, feature_path=None,
                 loss_type='xpolar', pad_strategy='zero',
                 freeze=True, frozen_path=None,
                 remote_source=None):
        self.__path = dataset_path
        self.__split = split
        self.__output_dim = output_dim
        meta_path = os.path.join(dataset_path, 'meta', split, 'records')
        if os.path.isdir(meta_path):
            self.__meta_path = meta_path
        else:
            raise FileNotFoundError(f'Meta folder {meta_path} does not exist')
        records = os.listdir(meta_path)
        self.file_indexes = list(map(lambda s: os.path.splitext(s)[0], records))
        feature_path = feature_path or os.path.join(dataset_path, 'features', split)
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
        self.__freeze = freeze
        if freeze:
            frozen_path = frozen_path or os.path.join(feature_path, '.sliced')
            self.__feature_slices_path = os.path.join(frozen_path, 'features')
            os.makedirs(self.__feature_slices_path, exist_ok=True)
            self.__loc_slices_path = os.path.join(frozen_path, 'loc')
            os.makedirs(self.__loc_slices_path, exist_ok=True)
            self.__prob_slices_path = os.path.join(frozen_path, 'prob')
            os.makedirs(self.__prob_slices_path, exist_ok=True)
        self.__data_source = ['local']
        if remote_source is not None:
            self.__data_source.extend(remote_source)

    def __getitem__(self, item):
        if len(self.__data_source) == 1:
            source = self.__data_source[0]
        else:
            source = random.choice(self.__data_source)
        return self.getitem(item, source)

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
        loc = torch.tensor(loc, dtype=torch.float)
        prob = torch.ones_like(loc)
        loc = F.pad(loc, (0, self.__output_dim - loc.shape[-1]))
        prob = F.pad(prob, (0, self.__output_dim - prob.shape[-1]))
        feature = torch.tensor(feature)
        return feature, loc, prob

    def getitem(self, item, source='local'):
        if source == 'local':
            index = sorted(self.indexes[item])
            if isinstance(item, int):
                itv = index[0]
                if self.__freeze:
                    item_a, item_b = divmod(item, 1000)
                    frozen_file = os.path.join(str(item_a), f'{item_b}.pt')
                    if os.path.exists(frozen_file):
                        feature = torch.load(os.path.join(self.__feature_slices_path, frozen_file))
                        loc = torch.load(os.path.join(self.__loc_slices_path, frozen_file))
                        prob = torch.load(os.path.join(self.__prob_slices_path, frozen_file))
                    else:
                        feature, loc, prob = self.get(itv.data, item - itv.begin)
                        for path in (self.__feature_slices_path, self.__loc_slices_path, self.__prob_slices_path):
                            os.makedirs(os.path.join(path, str(item_a)), exist_ok=True)
                        torch.save(feature, os.path.join(self.__feature_slices_path, frozen_file))
                        torch.save(loc, os.path.join(self.__loc_slices_path, frozen_file))
                        torch.save(prob, os.path.join(self.__prob_slices_path, frozen_file))
                else:
                    feature, loc, prob = self.get(itv.data, item - itv.begin)

            elif isinstance(item, slice):
                start, stop, step = item.indices(len(self))
                item = range(start, stop, step)
                item_info = map(lambda i: next(filter(lambda x: x.begin <= i < x.end, index)), item)
                item_info = list(map(lambda x: (x[0].data, x[1] - x[0].begin), zip(item_info, item)))

                if self.__freeze:
                    features, locs, probs = [], [], []
                    for n in item:
                        n_a, n_b = divmod(n, 1000)
                        frozen_file = os.path.join(str(n_a), f'{n_b}.pt')
                        if os.path.exists(frozen_file):
                            feature = torch.load(os.path.join(self.__feature_slices_path, frozen_file))
                            loc = torch.load(os.path.join(self.__loc_slices_path, frozen_file))
                            prob = torch.load(os.path.join(self.__prob_slices_path, frozen_file))
                        else:
                            feature, loc, prob = self.get(*item_info[n])
                            for path in (self.__feature_slices_path, self.__loc_slices_path, self.__prob_slices_path):
                                os.makedirs(os.path.join(path, str(n_a)), exist_ok=True)
                            torch.save(feature, os.path.join(self.__feature_slices_path, frozen_file))
                            torch.save(loc, os.path.join(self.__loc_slices_path, frozen_file))
                            torch.save(prob, os.path.join(self.__prob_slices_path, frozen_file))
                        features.append(feature)
                        locs.append(loc)
                        probs.append(prob)
                    feature, loc, prob = features, locs, probs
                else:
                    feature, loc, prob = list(zip(*map(lambda x: self.get(*x), item_info)))
                loc = torch.stack(loc)
                prob = torch.stack(prob)
                feature = torch.stack(feature)
            else:
                raise ValueError('Index must be int or slice')
        else:
            context = zmq.Context()
            sock = context.socket(zmq.REQ)
            sock.connect(f'tcp://{source}')
            sock.send_pyobj((self.__split, item))
            feature, loc, prob = sock.recv_pyobj()
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
        probs = [[1] * n + [0] * (self.__output_dim - n) for n in lengths]
        return lo, probs


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


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
