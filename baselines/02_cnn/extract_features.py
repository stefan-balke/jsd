"""
    This script extracts mel features from audio files.
    python extract_features.py\
    -s data/salami_audios/\
    -d data/salami_features/\
    -a data/salami_annotations/\
    -t data/salami_targets/
"""

import argparse
import multiprocessing
import numpy as np
import os
import tqdm
import glob
import librosa
import librosa.display
import utils
import pandas as pd


def extract_features(params):
    """Extracts mel features

    Parameters
    ----------
    params : dict
        dict containing all parameters

    """
    # extract params
    input_ds = params['input_ds']
    fn_audio = params['fn_audio']
    win_size = params['win_size']
    hop_size = params['hop_size']
    sr = params['sr']
    feature_rate = sr / hop_size
    file_id = params['file_id']
    path_features = params['path_features']
    path_annos = params['path_annos']
    path_targets = params['path_targets']

    try:
        if input_ds == 'salami':
            # load corresponding annotation file
            path_anno = os.path.join(path_annos, file_id + '.txt')
            anno = pd.read_csv(path_anno, header=None, sep='\t')
        elif input_ds == 'jsd':
            path_anno = os.path.join(path_annos, file_id + '.csv')
            anno = pd.read_csv(path_anno, sep=';')

    except FileNotFoundError:
        tqdm.tqdm.write('{}: No annotation file available. Skipping.'.format(file_id))
        return 0

    # load audio
    cur_audio, _ = librosa.load(fn_audio, sr=sr)

    # extract features with librosa
    f_mel = librosa.feature.melspectrogram(y=cur_audio, sr=sr, S=None, n_fft=win_size,
                                           hop_length=hop_size, power=2.0, fmin=80, fmax=16000, n_mels=80)

    # save features as npz
    cur_features = {'f_mel': f_mel, 'params': params}
    np.savez_compressed(os.path.join(path_features, file_id + '.npz'), **cur_features)

    # init target vector
    targets = np.zeros(f_mel.shape[1])

    if input_ds == 'salami':
        # filter out first silence boundary (it's the beginning of the song)
        if anno[1][0].lower() == 'silence':
            anno = anno.drop(0)

        if anno[1][anno.index[-1]].lower() == 'end':
            anno = anno.drop(anno.index[-1])

        # sometimes the last boundaries are beyond the song duration (mp3 problem?)
        if np.floor(anno[0][anno.index[-1]] * feature_rate).astype('int') >= targets.shape[0]:
            anno = anno.drop(anno.index[-1])
    elif input_ds == 'jsd':
        # filter out first silence boundary (it's the beginning of the song)
        if anno.head(1)['label'].lower() == 'silence':
            anno = anno.drop(0)

        if anno.tail(1)['label'].lower() == 'silence':
            anno = anno.drop(anno.tail(1).index)

    # from seconds to frame indices
    target_idcs = np.floor(anno[0].values * feature_rate).astype('int')

    # set boundaries
    try:
        targets[target_idcs] = 1
    except IndexError as e:
        print('{} could not assign targets. Skipping.'.format(file_id))
        print(e)

    cur_targets = {'target': targets, 'params': params}
    np.savez_compressed(os.path.join(path_targets, file_id + '.npz'), **cur_targets)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='name of the input dataset (jsd or salami)')
    parser.add_argument('-s', '--src', help='path to the audio sources')
    parser.add_argument('-d', '--dest', help='path to the output destination',
                        default='data/salami_features')
    parser.add_argument('-a', '--annos', help='path to the annotations')
    parser.add_argument('-t', '--targets', help='path to the target output destination',
                        default='data/salami_targets')
    args = parser.parse_args()

    path_audio = args.src
    path_features = args.dest
    path_annos = args.annos
    path_targets = args.targets

    # check if output folder exists
    os.makedirs(path_features, exist_ok=True)
    os.makedirs(path_targets, exist_ok=True)

    # create list of all track files, every track is in one folder
    fn_audios = glob.glob(os.path.join(path_audio, '*.mp3'))
    fn_audios += glob.glob(os.path.join(path_audio, '*.ogg'))
    fn_audios += glob.glob(os.path.join(path_audio, '*.wav'))
    file_ids = []

    for cur_path_audio in tqdm.tqdm(fn_audios):
        # extract names of the track
        file_ids.append(os.path.splitext(os.path.basename(cur_path_audio))[0])

    # stft parameters
    config = utils.load_config()
    hop_size = config['hop_size']
    win_size = config['win_size']
    sr = config['fs']

    params = [{'input_ds': args.input,
               'fn_audio': fn_audio,
               'file_id': file_id,
               'win_size': win_size,
               'hop_size': hop_size,
               'path_features': path_features,
               'path_annos': path_annos,
               'path_targets': path_targets,
               'sr': sr} for (fn_audio, file_id) in zip(fn_audios, file_ids)]

    mode = 'PROD'

    if mode == 'DEV':
        extract_features(params[1])

    if mode == 'PROD':
        # Batch processing
        # Init threading pool
        n_workers = 16
        pool = multiprocessing.Pool(n_workers)
        success = list(tqdm.tqdm(pool.imap_unordered(extract_features, params),
                       total=len(fn_audios)))

        pool.close()
