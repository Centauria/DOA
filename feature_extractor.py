# -*- coding: utf-8 -*-
import argparse
import os
from functools import partial
from multiprocessing import Pool
from time import time

import librosa
import numpy as np

overlapping = 512
chunk_size = 1024
num_frames = 25


def get_feature_stft(filename, mapping=None):
    x, fs = librosa.load(filename, sr=None, mono=False)
    assert fs == 16000

    if mapping is not None:
        newx = np.empty_like(x)
        mapping = np.insert(mapping, 0, 0)
        for i in range(len(mapping)):
            sign = np.sign(mapping[i]) if i > 0 else 1
            newx[i, :] = sign * x[abs(mapping[i]), :]
        x = newx

    # Compute the STFT
    lFrame = chunk_size  # size of the STFT window, in samples
    nOverlap = overlapping
    hop = lFrame - nOverlap

    x_f = np.asarray(list(map(
        partial(
            librosa.stft,
            n_fft=lFrame, hop_length=hop,
            window=np.sin(np.arange(0.5, lFrame + 0.5) / lFrame * np.pi),
            pad_mode='constant'
        ), x
    )))
    x_f = x_f.swapaxes(1, 2)

    # The neural network can only process buffers of 25 frames
    x_f = np.expand_dims(x_f, axis=0)
    lBuffer = num_frames
    splits = np.dsplit(x_f, range(lBuffer, x_f.shape[2], lBuffer))
    splits[-1] = np.pad(splits[-1], ((0, 0), (0, 0), (0, lBuffer - splits[-1].shape[2]), (0, 0)), 'constant')
    x_f = np.vstack(splits)
    return x_f


def get_feature_intensity(filename):
    x_f = get_feature_stft(filename, mapping=None)
    (lBatch, nChannel, lSentence, nBand) = x_f.shape
    h_nFeat = nChannel - 1
    nFeat = h_nFeat * 2
    inputFeat_f = np.empty((lBatch, nFeat, lSentence, nBand), dtype=np.float32)

    for nBatch, sig_f in enumerate(x_f):  # Process all examples in the batch
        # Compute the intensity vector in each TF bin
        f0_conj = sig_f[0].conj()
        intensityVect = np.transpose(f0_conj * sig_f[1:], (1, 2, 0))

        # Normalize it in each TF bin
        coeffNorm = (np.abs(sig_f[0]) ** 2 + np.sum(np.abs(sig_f[1:]) ** 2 / h_nFeat, axis=0))[:, :, np.newaxis]
        inputFeat_f[nBatch, :h_nFeat, :, :] = np.divide(
            intensityVect.real, coeffNorm,
            out=np.zeros_like(intensityVect, dtype=np.float32), where=(coeffNorm != 0)
        ).transpose(2, 0, 1)
        inputFeat_f[nBatch, h_nFeat:, :, :] = np.divide(
            intensityVect.imag, coeffNorm,
            out=np.zeros_like(intensityVect, dtype=np.float32), where=(coeffNorm != 0)
        ).transpose(2, 0, 1)

    return inputFeat_f


def get_feature_lps(filename):
    x_f = get_feature_stft(filename, mapping=None)
    x_f = np.abs(x_f)
    lps = 20 * np.log10(x_f, where=(x_f > 0), out=-5 * np.ones_like(x_f, dtype=float))
    return lps


def save_feature(filepath, savepath, method='stft', mapping=None):
    if method == 'stft':
        feature = get_feature_stft(filepath, mapping=mapping)
    elif method == 'intensity':
        feature = get_feature_intensity(filepath)
    elif method == 'lps':
        feature = get_feature_lps(filepath)
    else:
        raise ValueError('method must be in [stft, intensity, lps]')

    np.save(savepath, feature)
    print(f'Saved feature: {savepath}  dim={feature.shape}')


def main():
    parser = argparse.ArgumentParser(prog='feature_extractor',
                                     description="""Script to convert ambisonic audio to intensity vectors""")
    parser.add_argument('--feature-type', '-f', type=str, required=True, choices=('stft', 'intensity', 'lps'),
                        help='Type of feature')
    parser.add_argument('--audiodir', '-d', type=str, required=True,
                        help='Directory where audio files are located')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory where feature files are written to')
    parser.add_argument('--nthreads', '-n', type=int, default=1,
                        help='Number of threads to use')

    args = parser.parse_args()
    audiodir = args.audiodir
    nthreads = args.nthreads
    outpath = args.output

    os.makedirs(outpath, exist_ok=True)

    ts = time()
    # Convert all houses
    # # Create a pool to communicate with the worker threads
    with Pool(nthreads) as pool:
        for subdir, dirs, files in os.walk(audiodir):
            for d in dirs:
                os.makedirs(os.path.join(outpath, subdir.replace(audiodir, '').lstrip('/'), d), exist_ok=True)
            for f in files:
                if f.endswith('.wav'):
                    filename = os.path.join(subdir, f)
                    savepath = os.path.join(
                        outpath,
                        subdir.replace(audiodir, '').lstrip('/'),
                        f.replace('.wav', '.npy')
                    )
                    pool.apply_async(
                        partial(save_feature, method=args.feature_type),
                        args=(filename, savepath)
                    )
        pool.close()
        pool.join()

    print('Took {} seconds.'.format(time() - ts))


if __name__ == "__main__":
    main()
