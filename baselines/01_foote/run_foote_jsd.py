"""
    This script analyses the audiofeatures with foote for different parameter settings
    and evaluates the data with the annotations as reference. The results are stored as
    csv files.
"""

import sys
import os
import glob
import pandas as pd
import librosa
import foote
import tqdm
import yaml
import mir_eval
import numpy as np

# hacky relative import
sys.path.append(os.path.join('..', '..'))
import jsd_utils


def analysis(features, params, kernel_size):
    """Analysis of the audiofeature file with the current parameter settings.
    Calculates the SSM, NC and peaks of NC.

    Parameters
    ----------
    features : np.array
        Audiofeatures of the track extracted by extract_features_audio.py
    params : tuple
        Parameter tuples (smoothing_factor, downsampling_factor)
    kernel_size : int
        Kernel size of the gaussian kernel for Foote

    Returns
    -------
    ssm_f_cens : np.array
        SSM computed with CENS features
    ssm_f_mfcc : np.array
        SSM computed with MFCC features
    nc_cens : np.array
        NC computed for SSM based on CENS features
    nc_mfcc : np.array
        NC computed for SSM based on MFCC features
    boundaries_cens : np.array
        Array containing the peaks of the NC based on CENS features
    boundaries_mfcc : np.array
        Array containing the peaks of the NC based on MFCC features

    """

    # read features and abolish first two bands
    f_mfcc = features['f_mfcc'][2:]

    # normalize MFCC
    cur_f_mfcc = librosa.util.normalize(f_mfcc, norm=2, axis=0)

    # smooth MFCC
    cur_f_mfcc = foote.smooth_features(cur_f_mfcc, win_len_smooth=params[0])

    # downsample by params[1]
    cur_f_mfcc = cur_f_mfcc[:, ::params[1]]

    # compute the SSM
    ssm_f_mfcc = foote.compute_ssm(cur_f_mfcc)
    ssm_f_mfcc /= ssm_f_mfcc.max()

    # Compute gaussian kernel
    G = foote.compute_kernel_checkerboard_gaussian(kernel_size / 2)

    # Compute the novelty curve
    nc_mfcc = foote.compute_novelty_SSM(ssm_f_mfcc, G, exclude=True)
    nc_mfcc = np.abs(nc_mfcc) / nc_mfcc.max()

    return ssm_f_mfcc, nc_mfcc


def main():
    PATH_DATA = os.path.join('..', '..', 'data')
    path_annotations = os.path.join(PATH_DATA, 'annotations_csv')
    path_annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))
    jsd_track_db = jsd_utils.load_jsd(path_annotation_files)

    path_output = 'evaluation'
    path_data = 'data'
    path_features = os.path.join(path_data, 'jsd_features')
    feature_rate = 10

    # parameter tuning result on training set:
    # 0.5s = (9, 4)
    # 3.0s = (36, 4)
    params = [{'kernel_size': 80, 'cens': (9, 4)},
              {'kernel_size': 80, 'cens': (36, 4)}]
    thresholds = np.linspace(0, 1, 21)

    # make sure the folders exist before trying to save things
    if not os.path.isdir(os.path.join(path_data, path_output)):
        os.mkdir(os.path.join(path_data, path_output))

    with open('../../splits/jsd_fold-0.yml') as fh:
        split = yaml.load(fh, Loader=yaml.FullLoader)

    eval_output = []

    # analyze and evaluate the dataset with different kernel sizes
    for cur_params in params:
        cur_kernel_size = cur_params['kernel_size']
        cur_cens = cur_params['cens']
        print('--> {}'.format(cur_kernel_size))

        # loop over all tracks
        for cur_track_name in tqdm.tqdm(jsd_track_db['track_name'].unique()):
            cur_jsd_track = jsd_track_db[jsd_track_db['track_name'] == cur_track_name]
            cur_boundaries_ref = jsd_utils.get_boundaries(cur_jsd_track, musical_only=True)

            # get path to audiofeature file
            cur_path_features = os.path.join(path_features, cur_track_name + '.npz')

            # load features
            features = np.load(cur_path_features)

            # analyze the audio features
            (_, nc_mfcc) = analysis(features, cur_cens, cur_kernel_size)

            for cur_threshold in thresholds:
                # Compute the peaks of the NCs
                boundaries_mfcc = foote.peak_picking(nc_mfcc, abs_thresh=np.tile(cur_threshold, len(nc_mfcc)))
                # boundaries_mfcc = foote.detect_peaks(nc_mfcc, threshold=cur_threshold, fps=feature_rate / cur_cens[1])
                boundaries_mfcc = np.asarray(boundaries_mfcc)
                boundaries_mfcc = np.sort(boundaries_mfcc)

                # convert frame indices to seconds
                boundaries_mfcc = boundaries_mfcc / (feature_rate / cur_cens[1])

                cur_eval_row = {}
                cur_eval_row['track_name'] = cur_track_name
                cur_eval_row['cens'] = cur_cens
                cur_eval_row['kernel_size'] = cur_kernel_size
                cur_eval_row['threshold'] = cur_threshold

                # evaluate the boundaries for 0.5 seconds
                cur_eval_row['F_mfcc_05'], cur_eval_row['P_mfcc_05'], cur_eval_row['R_mfcc_05'] = \
                    mir_eval.onset.f_measure(boundaries_mfcc, cur_boundaries_ref, window=0.5)

                # evaluate the boundaries for 3.0 seconds
                cur_eval_row['F_mfcc_3'], cur_eval_row['P_mfcc_3'], cur_eval_row['R_mfcc_3'] = \
                    mir_eval.onset.f_measure(boundaries_mfcc, cur_boundaries_ref, window=3.0)

                # add dataframe of one track to dataframe of all songs
                eval_output.append(cur_eval_row)

    eval_output = pd.DataFrame(eval_output)
    eval_means = []

    for cur_key in split.keys():
        # get all tracks for the current split
        cur_tracks = split[cur_key]
        cur_eval_group = eval_output[eval_output['track_name'].isin(cur_tracks)]

        for cur_name, cur_group in cur_eval_group.groupby(by=['cens', 'kernel_size', 'threshold']):
            cur_means = cur_group.mean()
            cur_means['split'] = cur_key
            cur_means['cens'] = cur_name[0]
            eval_means.append(cur_means)

    eval_means = pd.DataFrame(eval_means)
    eval_means['threshold'] = eval_means['threshold'].round(2)

    # save dataframe as csv
    eval_means.to_csv(os.path.join(path_data, path_output, 'eval_means_jsd.csv'), sep=';')

    # get best thresholds on validation set
    best_thresholds = []
    for cur_name, cur_group in eval_means[eval_means['split'] == 'val'].groupby('cens'):
        cur_idx_05 = cur_group['F_mfcc_05'].idxmax()
        cur_idx_3 = cur_group['F_mfcc_3'].idxmax()

        cur_threshold = dict()
        cur_threshold['window'] = 0.5
        cur_threshold['cens'] = cur_group['cens'].unique()[0]
        cur_threshold['threshold'] = eval_means.iloc[cur_idx_05]['threshold']
        best_thresholds.append(cur_threshold)

        cur_threshold = dict()
        cur_threshold['window'] = 3
        cur_threshold['cens'] = cur_group['cens'].unique()[0]
        cur_threshold['threshold'] = eval_means.iloc[cur_idx_3]['threshold']
        best_thresholds.append(cur_threshold)

    for cur_params in best_thresholds:
        print(cur_params['window'])
        print(eval_means[(eval_means['split'] == 'test') & (eval_means['cens'] == cur_params['cens']) & (eval_means['threshold'] == cur_params['threshold'])])


if __name__ == '__main__':
    main()
