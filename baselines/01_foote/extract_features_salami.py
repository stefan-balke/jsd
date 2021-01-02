"""
    This script extracts mfcc features from audio files in parallel.
"""

import argparse
import multiprocessing
import numpy as np
import pandas as pd
import os
import tqdm
import glob
import librosa


def extract_features(params):
    sr = 22050
    fn_audio = params['fn_audio']
    hop_size = params['hop_size']
    file_id = params['file_id']
    path_annos = params['path_annos']

    # load corresponding annotation file
    path_anno = os.path.join(path_annos, file_id + '.txt')

    try:
        anno = pd.read_csv(path_anno, header=None, sep='\t')
    except FileNotFoundError:
        tqdm.tqdm.write('{}: No annotation file available. Skipping.'.format(file_id))
        return 0

    cur_audio, _ = librosa.load(fn_audio, sr=sr)

    f_mfcc = librosa.feature.mfcc(cur_audio, sr, S=None, n_mfcc=20,
                                  n_fft=2*hop_size, hop_length=hop_size)

    cur_features = {'params': params,
                    'f_mfcc': f_mfcc}

    np.savez_compressed(os.path.join(path_features, file_id + '.npz'), **cur_features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', help='path to the audio sources')
    parser.add_argument('-d', '--dest', help='path to the output destination')
    parser.add_argument('-a', '--annos', help='path to the annotations')
    args = parser.parse_args()

    path_audio = args.src
    path_features = args.dest
    path_annos = args.annos

    # check if output folder exists
    if not os.path.exists(path_features):
        os.makedirs(path_features)

    hop_size = 2205

    fn_audios = glob.glob(os.path.join(path_audio, '*.mp3'))
    fn_audios += glob.glob(os.path.join(path_audio, '*.ogg'))
    fn_audios += glob.glob(os.path.join(path_audio, '*.wav'))
    file_ids = []

    for cur_path_audio in tqdm.tqdm(fn_audios):
        # extract names of the track
        file_ids.append(os.path.splitext(os.path.basename(cur_path_audio))[0])

    params = [{'fn_audio': fn_audio,
               'file_id': file_id,
               'path_annos': path_annos,
               'hop_size': hop_size} for (fn_audio, file_id) in zip(fn_audios, file_ids)]

    mode = 'PRODUCTION'

    if mode == 'DEVELOP':
        # Process single file and display results
        extract_features(params[1])

    if mode == 'PRODUCTION':
        # Batch processing
        # Init threading pool
        n_workers = 4
        pool = multiprocessing.Pool(n_workers)
        success = list(tqdm.tqdm(pool.imap_unordered(extract_features, params),
                       total=len(fn_audios)))

        pool.close()
