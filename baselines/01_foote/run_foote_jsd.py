"""
    This script analyses the audiofeatures with foote for different parametersettings
    and evaluates the data with the annotations as reference. The results are stored as
    csv files.
"""

import sys
import os
import glob
import pandas as pd
import tqdm
import yaml
import mir_eval
import numpy as np
import foote_utils
from scipy import signal

# hacky relative import
sys.path.append(os.path.join('..', '..'))
import jsd_utils


def main():
    PATH_DATA_JSD = os.path.join('..', '..', 'data')
    path_annotations = os.path.join(PATH_DATA_JSD, 'annotations_csv')
    path_annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))
    jsd_track_db = jsd_utils.load_jsd(path_annotation_files)

    path_output = 'foote_jsd_website'
    path_data = os.path.join('..', 'data')
    path_eval = os.path.join(path_data, path_output)
    path_features = os.path.join(path_data, 'foote_jsd_features')
    feature_rate = 10

    # parameter tuning result on training set:
    # 0.5s = (9, 4)
    # 3.0s = (36, 4)
    params = [{'kernel_size': 40, 'wl_ds': (9, 4)},
              {'kernel_size': 80, 'wl_ds': (36, 4)}]
    thresholds = np.linspace(0, 1, 21)

    # make sure the folders exist before trying to save things
    if not os.path.isdir(os.path.join(path_data, path_output)):
        os.mkdir(os.path.join(path_data, path_output))

    # call main experiment
    eval_output = foote_utils.foote_experiment(jsd_track_db, params, thresholds, feature_rate,
                                               path_features, path_eval, save_ncs=True)

    # evaluate on different splits
    with open('../../splits/jsd_fold-0.yml') as fh:
        split = yaml.load(fh, Loader=yaml.FullLoader)

    # statistics on evaluation results
    eval_means = []

    for cur_key in split.keys():
        # get all tracks for the current split
        cur_tracks = split[cur_key]
        cur_eval_group = eval_output[eval_output['track_name'].isin(cur_tracks)]

        for cur_name, cur_group in cur_eval_group.groupby(by=['wl_ds', 'kernel_size', 'threshold']):
            cur_means = cur_group.mean()
            cur_means['split'] = cur_key
            cur_means['wl_ds'] = cur_name[0]
            eval_means.append(cur_means)

    eval_means = pd.DataFrame(eval_means)
    eval_means['threshold'] = eval_means['threshold'].round(2)

    # save dataframe as csv
    eval_means.to_csv(os.path.join(path_eval, 'eval_means_jsd.csv'), sep=';')

    # get best thresholds on validation set
    best_thresholds = []
    for cur_name, cur_group in eval_means[eval_means['split'] == 'val'].groupby('wl_ds'):
        cur_idx_05 = cur_group['F_mfcc_05'].idxmax()
        cur_idx_3 = cur_group['F_mfcc_3'].idxmax()

        cur_threshold = dict()
        cur_threshold['window'] = 0.5
        cur_threshold['wl_ds'] = cur_group['wl_ds'].unique()[0]
        cur_threshold['threshold'] = eval_means.iloc[cur_idx_05]['threshold']
        best_thresholds.append(cur_threshold)

        cur_threshold = dict()
        cur_threshold['window'] = 3
        cur_threshold['wl_ds'] = cur_group['wl_ds'].unique()[0]
        cur_threshold['threshold'] = eval_means.iloc[cur_idx_3]['threshold']
        best_thresholds.append(cur_threshold)

    for cur_params in best_thresholds:
        print('Best parameter set for window {}'.format(cur_params['window']))
        print(eval_means[(eval_means['split'] == 'test') & (eval_means['wl_ds'] == cur_params['wl_ds']) & (eval_means['threshold'] == cur_params['threshold'])].round(decimals=3))


if __name__ == '__main__':
    main()
