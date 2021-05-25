# -*- coding: utf-8 -*-
import json
import os

import librosa
import numpy as np
import ruamel_yaml as yaml
from torch.utils.data import Dataset


class GenDOA(Dataset):
    def __init__(self, dataset_path, split='train', feature_path=None):
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

    def __getitem__(self, item):
        name = self.indexes[item]
        if type(name) == list:
            meta = []
            feature = []
            for n in name:
                meta_filename = os.path.join(self.__meta_path, n + '.json')
                with open(meta_filename) as f:
                    meta.append(json.load(f))
                feature.append(self.feature(n))
            feature = np.concatenate(feature, axis=0)
        else:
            meta_filename = os.path.join(self.__meta_path, name + '.json')
            with open(meta_filename) as f:
                meta = json.load(f)
            feature = self.feature(name)
        return feature, meta

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
