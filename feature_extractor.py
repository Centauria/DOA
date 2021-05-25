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


def getNormalizedIntensity(x_f):
    """
    Compute the input feature needed by the localization neural network.

    Parameters
    ----------
    x_f: nd-array
        STFT of the HOA signal.
        Shape: (nBatch, nChannel, lSentence, nBand)

    Returns
    -------
    inputFeat_f: nd-array
        Shape: (nBatch, lSentence, nBand, nFeat)
    """
    (lBatch, nChannel, lSentence, nBand) = x_f.shape
    nFeat = 6
    inputFeat_f = np.empty((lBatch, lSentence, nBand, nFeat), dtype=np.float32)

    for nBatch, sig_f in enumerate(x_f):  # Process all examples in the batch
        # Compute the intensity vector in each TF bin
        intensityVect = np.empty((lSentence, nBand, 3), dtype=complex)
        intensityVect[:, :, 0] = sig_f[0].conj() * sig_f[1]
        intensityVect[:, :, 1] = sig_f[0].conj() * sig_f[2]
        intensityVect[:, :, 2] = sig_f[0].conj() * sig_f[3]

        # Normalize it in each TF bin
        coeffNorm = (abs(sig_f[0]) ** 2 + np.sum(abs(sig_f[1:]) ** 2 / 3, axis=0))[:, :, np.newaxis]
        inputFeat_f[nBatch, :, :, :nFeat // 2] = np.divide(
            intensityVect.real, coeffNorm,
            out=np.zeros_like(intensityVect, dtype=np.float32), where=(coeffNorm != 0)
        )
        inputFeat_f[nBatch, :, :, nFeat // 2:] = np.divide(
            intensityVect.imag, coeffNorm,
            out=np.zeros_like(intensityVect, dtype=np.float32), where=(coeffNorm != 0)
        )

    return inputFeat_f


def save_feature(filepath, savepath, mapping=None):
    x, fs = librosa.load(filepath, sr=None, mono=False)
    assert fs == 16000

    if mapping is not None:
        newx = np.empty_like(x)
        mapping = np.insert(mapping, 0, 0)
        for i in range(len(mapping)):
            sign = np.sign(mapping[i]) if i > 0 else 1
            newx[i, :] = sign * x[abs(mapping[i]), :]
        x = newx

    # Compute the STFT
    nChannel, nSmp = x.shape
    lFrame = 1024  # size of the STFT window, in samples
    nOverlap = lFrame // 2
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
    lBuffer = 25
    splits = np.dsplit(x_f, range(lBuffer, x_f.shape[2], lBuffer))
    splits[-1] = np.pad(splits[-1], ((0, 0), (0, 0), (0, lBuffer - splits[-1].shape[2]), (0, 0)), 'constant')
    x_f = np.vstack(splits)

    # Get the input feature for the neural network
    inputFeat_f = getNormalizedIntensity(x_f)
    np.save(savepath, inputFeat_f)
    print(f'Saved feature: {savepath}  dim={inputFeat_f.shape}')


def main():
    parser = argparse.ArgumentParser(prog='feature_extractor',
                                     description="""Script to convert ambisonic audio to intensity vectors""")
    parser.add_argument("--audiodir", "-d", help="Directory where audio files are located",
                        type=str, required=True)
    parser.add_argument("--output", "-o", help="Directory where feature files are written to",
                        type=str, required=True)
    parser.add_argument("--nthreads", "-n", type=int, default=1, help="Number of threads to use")

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
                    pool.apply_async(save_feature, args=(filename, savepath))
        pool.close()
        pool.join()

    print('Took {}'.format(time() - ts))


if __name__ == "__main__":
    main()
