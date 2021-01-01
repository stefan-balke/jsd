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
import mir_eval
import numpy as np

# hacky relative import
sys.path.append(os.path.join('..', '..'))
import utils


def analysis(features, params_cens, kernel_size, threshold):
    """Analysis of the audiofeature file with the current parameter settings.
    Calculates the SSM, NC and peaks of NC for CENS and MFCC.

    Parameters
    ----------
    features : np.array
        Audiofeatures of the track extracted by extract_features_audio.py
    params_cens : tuple
        Parameter tuples (smoothing_factor, downsampling_factor)
    kernel_size : int
        Kernel size of the gaussian kernel for Foote
    threshold : float
        Threshold for the peak picker.


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

    # read features
    f_pitch = features['f_pitch']
    f_mfcc = features['f_mfcc']

    # calculate CENS features, includes smoothing
    cur_f_cens = librosa.feature.chroma_cens(C=f_pitch, win_len_smooth=params_cens[0])

    # downsample by params_cens[1]
    cur_f_cens = cur_f_cens[:, ::params_cens[1]]

    # smooth MFCC
    cur_f_mfcc = foote.smooth_features(f_mfcc, win_len_smooth=params_cens[0])

    # normalize MFCC
    cur_f_mfcc = librosa.util.normalize(cur_f_mfcc, norm=2, axis=0)

    # downsample by params_cens[1]
    cur_f_mfcc = cur_f_mfcc[2:, ::params_cens[1]]

    # compute the SSMs
    ssm_f_cens = foote.compute_ssm(cur_f_cens)
    ssm_f_cens /= ssm_f_cens.max()
    ssm_f_mfcc = foote.compute_ssm(cur_f_mfcc)
    ssm_f_mfcc /= ssm_f_mfcc.max()

    # Compute gaussian kernel
    G = foote.compute_kernel_checkerboard_gaussian(kernel_size)

    # Compute the novelty curves
    nc_cens = foote.compute_novelty_SSM(ssm_f_cens, G, exclude=True)
    nc_cens = np.abs(nc_cens) / nc_cens.max()
    nc_mfcc = foote.compute_novelty_SSM(ssm_f_mfcc, G, exclude=True)
    nc_mfcc = np.abs(nc_mfcc) / nc_mfcc.max()

    # Compute the peaks of the NCs
    boundaries_cens = np.sort(np.asarray(foote.peak_picking(nc_cens, abs_thresh=np.tile(threshold, len(nc_cens)))))
    boundaries_mfcc = np.sort(np.asarray(foote.peak_picking(nc_mfcc, abs_thresh=np.tile(threshold, len(nc_mfcc)))))
    # fps = 10 / params_cens[1]
    # fps = 10
    # boundaries_cens = np.sort(np.asarray(foote.detect_peaks(nc_cens, threshold=threshold, fps=fps)))
    # boundaries_mfcc = np.sort(np.asarray(foote.detect_peaks(nc_mfcc, threshold=threshold, fps=fps)))

    return ssm_f_cens, ssm_f_mfcc, nc_cens, nc_mfcc, boundaries_cens, boundaries_mfcc


def main():
    PATH_DATA = os.path.join('..', '..', 'data')
    path_annotations = os.path.join(PATH_DATA, 'annotations_csv')
    path_annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))
    jsd_track_db = utils.load_jsd(path_annotation_files)

    path_output = 'evaluation'
    path_data = 'data'
    path_features = os.path.join(path_data, 'jsd_features')
    feature_rate = 10
    # params_cens = [(9, 2), (9, 4), (21, 5)]
    params_cens = [(9, 2), ]
    thresholds = np.linspace(0, 1, 21)

    # make sure the folders exist before trying to save things
    if not os.path.isdir(os.path.join(path_data, path_output)):
        os.mkdir(os.path.join(path_data, path_output))

    eval_output_05 = []
    eval_output_3 = []

    # analyse and evaluate the dataset with different kernel sizes
    for cur_kernel_size in [40, ]:  # [20, 30, 40, 50]:  # [60, 70, 80, 90]:
        print('--> {}'.format(cur_kernel_size))

        # loop over all tracks
        for cur_track_name in tqdm.tqdm(jsd_track_db['track_name'].unique()):
            cur_jsd_track = jsd_track_db[jsd_track_db['track_name'] == cur_track_name]
            cur_boundaries_ref = utils.get_boundaries(cur_jsd_track)

            # get path to audiofeature file
            cur_path_features = os.path.join(path_features, cur_track_name + '.npz')

            # load features
            features = np.load(cur_path_features)

            # loop over all parameter settings
            for cur_params_cens in params_cens:
                for cur_threshold in thresholds:
                    # analyse the audiofeatures
                    (_, _, _, _, boundaries_cens, boundaries_mfcc) = analysis(features,
                                                                              cur_params_cens,
                                                                              cur_kernel_size,
                                                                              cur_threshold)

                    # convert frame indices to seconds
                    boundaries_cens = boundaries_cens / (feature_rate / cur_params_cens[1])
                    boundaries_mfcc = boundaries_mfcc / (feature_rate / cur_params_cens[1])

                    # evaluate the boundaries for 0.5 seconds
                    cur_eval_row_05 = {}
                    cur_eval_row_05['F_cens'], cur_eval_row_05['P_cens'], cur_eval_row_05['R_cens'] = \
                        mir_eval.onset.f_measure(boundaries_cens, cur_boundaries_ref, window=0.5)
                    cur_eval_row_05['F_mfcc'], cur_eval_row_05['P_mfcc'], cur_eval_row_05['R_mfcc'] = \
                        mir_eval.onset.f_measure(boundaries_mfcc, cur_boundaries_ref, window=0.5)
                    cur_eval_row_05['param'] = cur_params_cens
                    cur_eval_row_05['kernel_size'] = cur_kernel_size
                    cur_eval_row_05['threshold'] = cur_threshold

                    # evaluate the boundaries for 3.0 seconds
                    cur_eval_row_3 = {}
                    cur_eval_row_3['F_cens'], cur_eval_row_3['P_cens'], cur_eval_row_3['R_cens'] = \
                        mir_eval.onset.f_measure(boundaries_cens, cur_boundaries_ref, window=3.0)
                    cur_eval_row_3['F_mfcc'], cur_eval_row_3['P_mfcc'], cur_eval_row_3['R_mfcc'] = \
                        mir_eval.onset.f_measure(boundaries_mfcc, cur_boundaries_ref, window=3.0)
                    cur_eval_row_3['param'] = cur_params_cens
                    cur_eval_row_3['kernel_size'] = cur_kernel_size
                    cur_eval_row_3['threshold'] = cur_threshold

                    # add dataframe of one track to dataframe of all songs
                    eval_output_05.append(cur_eval_row_05)
                    eval_output_3.append(cur_eval_row_3)

    # save dataframe as csv
    eval_output_05 = pd.DataFrame(eval_output_05)
    eval_output_05.to_csv(os.path.join(path_data, path_output, 'evaluation_winlength-{}.csv'.format('05')),
                          sep=';')

    eval_output_3 = pd.DataFrame(eval_output_3)
    eval_output_3.to_csv(os.path.join(path_data, path_output, 'evaluation_winlength-{}.csv'.format('3')),
                         sep=';')


if __name__ == '__main__':
    main()
