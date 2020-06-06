"""
    This script extracts pitch (log-freq.) and mfcc features from audio files.
"""

import argparse
import multiprocessing
import numpy as np
import os
import tqdm
import glob
import librosa


def extract_features(params):
    sr = 22050
    fn_audio = params['fn_audio']
    hop_size = params['hop_size']

    cur_audio, _ = librosa.load(fn_audio, sr=sr)

    f_pitch = librosa.core.iirt(cur_audio, win_length=2*hop_size, hop_length=hop_size)

    f_mfcc = librosa.feature.mfcc(cur_audio, sr, S=None, n_mfcc=20,
                                  n_fft=2*hop_size, hop_length=hop_size)

    fn_cur_features = os.path.splitext(os.path.split(fn_audio)[1])

    cur_features = {'f_pitch': f_pitch, 'params': params,
                    'f_mfcc': f_mfcc}

    np.savez_compressed(os.path.join(path_features, fn_cur_features[0] + '.npz'), **cur_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', help='path to the audio sources')
    parser.add_argument('-d', '--dest', help='path to the output destination')
    args = parser.parse_args()

    path_audio = args.script
    path_features = args.dest

    # check if output folder exists
    if not os.path.exists(path_features):
        os.makedirs(path_features)

    hop_size = 2205

    fn_audios = glob.glob(os.path.join(path_audio, '*.wav'))

    params = [{'fn_audio': fn_audio,
               'hop_size': hop_size} for fn_audio in fn_audios]

    mode = 'PRODUCTION'

    if mode == 'DEVELOP':
        # Process single file and display results
        extract_features(params[0])
        extract_features(params[1])
        extract_features(params[2])
        extract_features(params[3])
        extract_features(params[4])

    if mode == 'PRODUCTION':
        # Batch processing
        # Init threading pool
        n_workers = 4
        pool = multiprocessing.Pool(n_workers)
        success = list(tqdm.tqdm(pool.imap_unordered(extract_features, params),
                       total=len(fn_audios)))

        pool.close()
