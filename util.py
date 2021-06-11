# -*- coding: utf-8 -*-
import numpy as np


def cartesian(mic_array_location, sources_location):
    if len(sources_location) == 0:
        return np.array([])
    mic_array_location = np.asarray(mic_array_location)
    sources_location = np.asarray(sources_location)
    mic_center = np.average(mic_array_location, axis=1)
    xyz = sources_location - mic_center
    return xyz


def polar(mic_array_location, sources_location):
    if len(sources_location) == 0:
        return np.array([])
    xyz = cartesian(mic_array_location, sources_location)
    r = np.linalg.norm(xyz, axis=1)
    inclination = np.arccos(xyz[:, 2] / r)
    azimuth = np.arccos(xyz[:, 0] / np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2))
    return np.asarray((r, inclination, azimuth)).T


def x_linear_polar(mic_array_location, sources_location):
    if len(sources_location) == 0:
        return np.array([])
    xyz = cartesian(mic_array_location, sources_location)
    r = np.linalg.norm(xyz, axis=1)
    alpha = np.arccos(xyz[:, 0] / r)
    return np.asarray(alpha)
