# -*- coding: utf-8 -*-
import json
import os

import librosa
import numpy as np
import ruamel_yaml as yaml
import torch
import torch.utils.data

import util


class GenDOA(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split='train', feature_path=None, loss_type='cartesian'):
        self.__path = dataset_path
        self.__split = split
        meta_path = os.path.join(dataset_path, 'meta', split, 'records')
        if os.path.isdir(meta_path):
            self.__meta_path = meta_path
        else:
            raise FileNotFoundError(f'Meta folder {meta_path} does not exist')
        records = os.listdir(meta_path)
        self.indexes = list(map(lambda s: os.path.splitext(s)[0], records))
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
        assert loss_type in ('categorical', 'cartesian', 'polar')
        self.__loss_type = loss_type

    def __getitem__(self, item):
        name = self.indexes[item]
        if type(name) == list:
            room = list(map(self.room, name))
            if self.__loss_type == 'cartesian':
                loc = list(map(lambda r: util.cartesian(r['mic_array_location'], r['sources_location']), room))
            elif self.__loss_type == 'polar':
                loc = list(map(lambda r: util.polar(r['mic_array_location'], r['sources_location']), room))
            else:
                raise ValueError(f'Loss type {self.__loss_type} not defined')
            feature = list(map(self.feature, name))
            feature = np.concatenate(feature, axis=0)
        else:
            room = self.room(name)
            if self.__loss_type == 'cartesian':
                loc = util.cartesian(room['mic_array_location'], room['sources_location'])
            elif self.__loss_type == 'polar':
                loc = util.polar(room['mic_array_location'], room['sources_location'])
            else:
                raise ValueError(f'Loss type {self.__loss_type} not defined')
            feature = self.feature(name)
        loc = torch.tensor(loc)
        feature = torch.tensor(feature).permute(0, 3, 1, 2)
        # TODO: 输出每一个时刻的DOA信息
        return feature, loc

    def __len__(self):
        return len(self.indexes)

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
