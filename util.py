# -*- coding: utf-8 -*-
import numpy as np


def cartesian(mic_array_location, sources_location):
    mic_array_location = np.asarray(mic_array_location)
    sources_location = np.asarray(sources_location)
    mic_center = np.average(mic_array_location, axis=1)
    xyz = sources_location - mic_center
    return xyz


def polar(mic_array_location, sources_location):
    xyz = cartesian(mic_array_location, sources_location)
    r = np.linalg.norm(xyz, axis=1)
    inclination = np.arccos(xyz[:, 2] / r)
    azimuth = np.arccos(xyz[:, 0] / np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2))
    return np.asarray((r, inclination, azimuth)).T
