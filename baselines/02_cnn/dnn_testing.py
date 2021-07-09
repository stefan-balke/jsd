"""
    Loop over the test set and calculate accuracy.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import data_streamer as ds
from tensorflow.keras.models import model_from_json
import tqdm
import utils
import mir_eval
import pescador
import yaml
import glob

# hacky relative import
sys.path.append(os.path.join('..', '..'))
import jsd_utils
sys.path.append(os.path.join('..'))
import salami_utils


def filter_boundary_estimates(boundaries_est, boundaries_ref, boundaries_ref_nm, bndry_window=3.0):
    # import info for the evaluation region
    time_first_musical_boundary = np.min(boundaries_ref)
    time_last_musical_boundary = np.max(boundaries_ref)

    # keep boundaries for visualization
    est_boundaries_exclude = []

    # filter out boundary candidates around non-musical boundaries
    cur_est_boundaries_exclude_idcs = []

    for cur_boundary_nm in boundaries_ref_nm:
        # check if estimate is in near nm-boundary
        cur_excludes = np.abs(boundaries_est - cur_boundary_nm)
        cur_excludes = np.where(cur_excludes <= bndry_window)
        cur_excludes = cur_excludes[0].tolist()

        cur_est_boundaries_exclude_idcs.extend(cur_excludes)

    # add boundaries outside evaluation window [t_first - window, t_last + window]
    cur_est_boundaries_exclude_idcs.extend(np.where(boundaries_est < (time_first_musical_boundary - bndry_window))[0].tolist())
    cur_est_boundaries_exclude_idcs.extend(np.where(boundaries_est > (time_last_musical_boundary + bndry_window))[0].tolist())

    cur_est_boundaries_exclude_idcs = np.asarray(cur_est_boundaries_exclude_idcs)
    cur_est_boundaries_exclude_idcs = np.unique(cur_est_boundaries_exclude_idcs)

    if cur_est_boundaries_exclude_idcs.shape[0] > 1:
        cur_est_boundaries_exclude = boundaries_est[cur_est_boundaries_exclude_idcs]
    else:
        cur_est_boundaries_exclude = []

    est_boundaries_exclude.append(cur_est_boundaries_exclude)

    try:
        return np.delete(boundaries_est, cur_est_boundaries_exclude_idcs), cur_est_boundaries_exclude
    except IndexError:
        print('Problem with boundary exclusion.')
        return boundaries_est, cur_est_boundaries_exclude


def predict(pathes_X, pathes_y, path_model, path_weights, config):

    # Model reconstruction from JSON file
    with open(path_model, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(path_weights)

    predictions = []
    gts = []
    songs = []

    for cur_song_idx, cur_path_X in enumerate(tqdm.tqdm(pathes_X)):
        # streamer with just a single song
        stream = ds.stream_generator(cur_path_X, 'f_mel', pathes_y[cur_song_idx], 'target', patch_width=config['input_shape'][0],
                                     flatten_X=False, shuffle=False, add_dimension=True, target_smear=False,
                                     class_balance=False, subsampling=config['subsampling'])
        mini_batch_generator = pescador.buffer_stream(stream, 8)

        # predict for each patch
        cur_predictions = []
        cur_gts = []
        cur_song = os.path.splitext(os.path.basename(cur_path_X))[0]

        for cur_batch in mini_batch_generator:
            cur_predictions.extend(model.predict_on_batch(cur_batch['X']).squeeze())
            cur_gts.extend(cur_batch['y'].squeeze())

        predictions.append(cur_predictions)
        gts.append(cur_gts)
        songs.append(cur_song)

    return predictions, gts, songs


def evaluate(songs, predictions, gts, window, feature_rate, threshold, musical_only=True):
    Fs = []
    Ps = []
    Rs = []

    for cur_song_id, cur_song in enumerate(songs):
        cur_nc = np.asarray(predictions[cur_song_id])

        # get ground-truth
        cur_track = gts[gts['track_name'] == songs[cur_song_id]]

        # Compute the peaks of the NCs
        # cur_boundaries = utils.detect_peaks(cur_nc, fps=feature_rate, threshold=threshold)
        # prominence = 0.01
        from scipy import signal
        cur_boundaries = signal.find_peaks(cur_nc, height=0, prominence=prominence)[0]
        cur_boundaries = np.asarray(cur_boundaries)
        cur_boundaries = np.sort(cur_boundaries)
        cur_boundaries = cur_boundaries / feature_rate

        if musical_only:
            # get reference boundaries for the current song
            # (not restricted to JSD...only calling a helper function)
            cur_boundaries_ref = jsd_utils.get_boundaries(cur_track, musical_only=True)
            # get the non-musical boundaries only
            cur_boundaries_ref_nm = jsd_utils.get_boundaries(cur_track, musical_only=False)
            cur_boundaries_ref_nm = np.setdiff1d(cur_boundaries_ref_nm, cur_boundaries_ref)

            try:
                assert len(cur_boundaries_ref) > 0
            except AssertionError:
                print('Detected track with no musical boundaries: {}. Setting P, R, and F to 0.'.format(songs[cur_song_id]))
                Fs.append(0)
                Rs.append(0)
                Ps.append(0)
                continue

            # filter boundary estimates
            cur_boundaries_est, cur_est_boundaries_exclude = filter_boundary_estimates(cur_boundaries, cur_boundaries_ref, cur_boundaries_ref_nm, bndry_window=3.0)

            # debugging visualization
            debug = False
            if debug:
                import matplotlib.pyplot as plt
                os.makedirs('debug_plots', exist_ok=True)
                scaler = feature_rate
                fig, ax = plt.subplots(nrows=1, figsize=(20, 4))
                fig.suptitle('{}, Threshold: {:.2f}'.format(cur_song, threshold))
                ax.plot(cur_nc, label='NC')
                ax.set_xlim(0, len(cur_nc))
                ax.set_ylim(0, 1)

                ax.vlines(np.round(cur_boundaries_ref * scaler), 0.8, 1, color='g', label='Ref.')
                ax.vlines(np.round(cur_boundaries_ref_nm * scaler), 0.7, 1, color='b', label='Ref. NM')

                ax.vlines(np.round(cur_boundaries * scaler), 0, 0.8, color='r', label='Est.')
                ax.vlines(np.round(cur_boundaries_est * scaler), 0, 0.3, color='k', label='Est. Eval')
                if len(cur_est_boundaries_exclude) > 0:
                    ax.vlines(np.round(cur_est_boundaries_exclude * scaler), 0, 0.5, color='y', label='Excluded')

                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                plt.savefig('debug_plots/{}.png'.format(cur_song))
                plt.close()
        else:
            # consider all boundaries
            cur_boundaries_ref = jsd_utils.get_boundaries(cur_track, musical_only=False)
            cur_boundaries_est = cur_boundaries

        F, P, R = mir_eval.onset.f_measure(cur_boundaries_ref,
                                           cur_boundaries_est,
                                           window=window)

        # print('#{}: F: {}, P: {}, R: {}'.format(cur_song_id, F, P, R))
        Fs.append(F)
        Rs.append(R)
        Ps.append(P)

    return np.mean(Fs), np.mean(Ps), np.mean(Rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Testing')
    parser.add_argument('--path_features', type=str, nargs='*', default=['../data/cnn_salami_features', ])
    parser.add_argument('--path_targets', type=str, nargs='*', default=['../data/cnn_salami_targets', ])
    parser.add_argument('--path_results', type=str)
    parser.add_argument('--test_splits', type=str, nargs='*', default=['../data/salami_split.yml', ], help='Path to test splits (can be more than one).')
    parser.add_argument('--eval_only', action='store_true', default=False, help='Use pre-computed novelty curves for evaluation.')
    parser.add_argument('--musical_only', action='store_true', default=False, help='Only consider musical segments for evaluation.')
    parser.add_argument('--bagging', type=int, default=1, help='Number of bagged networks.')

    args = parser.parse_args()

    config = utils.load_config(os.path.join(args.path_results, 'config.yml'))

    # collect pathes
    PATH_X = args.path_features
    PATH_y = args.path_targets

    # collect split data
    splits = []
    for cur_path in args.test_splits:
        print('Loading test split: {}'.format(cur_path))

        # load test split
        with open(cur_path) as fh:
            splits.append(yaml.load(fh, Loader=yaml.FullLoader))

    # prepare path to data for test set
    pathes_test_X = []
    for cur_ds_id, cur_path_X in enumerate(PATH_X):
        for cur_fn in splits[cur_ds_id]['test']:
            pathes_test_X.append(os.path.join(cur_path_X, '{}.npz'.format(cur_fn)))

    pathes_test_y = []
    for cur_ds_id, cur_path_y in enumerate(PATH_y):
        for cur_fn in splits[cur_ds_id]['test']:
            pathes_test_y.append(os.path.join(cur_path_y, '{}.npz'.format(cur_fn)))

    # load pre-computed thresholds for peak picking
    try:
        fn_split = 'peak_picking_thresholds.yml'
        with open(os.path.join(args.path_results, fn_split), 'rb') as fp:
            thresholds = yaml.load(fp, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print('No peak_picking_thresholds.yml found. Please run "optimize_peak_picking.py" first.')
        sys.exit()

    # load annotations
    PATH_DATA_SALAMI = os.path.join('..', 'data')
    path_annotations = os.path.join(PATH_DATA_SALAMI, 'salami_annotations')
    track_durs = pd.read_csv(os.path.join(PATH_DATA_SALAMI, 'salami_track_durations.csv'))
    track_durs = track_durs.astype(str)
    salami_track_db = salami_utils.load_salami(track_durs, path_annotations)

    PATH_DATA_JSD = os.path.join('..', '..', 'data')
    path_annotations = os.path.join(PATH_DATA_JSD, 'annotations_csv')
    path_annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))
    jsd_track_db = jsd_utils.load_jsd(path_annotation_files)

    annotations = pd.concat([salami_track_db, jsd_track_db])
    bags = []

    for cur_bag_idx in range(args.bagging):
        path_model = os.path.join(args.path_results, 'architecture-{}.json'.format(cur_bag_idx))
        path_weights = os.path.join(args.path_results, 'weights-{}.h5'.format(cur_bag_idx))
        path_pred = os.path.join(args.path_results, 'pred-test_{}_bag-{}.npz'.format(
            os.path.splitext(os.path.split(args.test_splits[0])[1])[0],
            cur_bag_idx))
        data = dict()

        if args.eval_only:
            saved_data = np.load(path_pred, allow_pickle=True)
            data['predictions'] = saved_data['predictions']
            data['gts'] = saved_data['gts']
            data['songs'] = saved_data['songs']
        else:
            predictions, gts, songs = predict(pathes_test_X, pathes_test_y,
                                              path_model, path_weights, config=config)
            np.savez_compressed(path_pred, predictions=predictions, gts=gts, songs=songs)
            data['predictions'] = predictions
            data['gts'] = gts
            data['songs'] = songs

        bags.append(data)

        try:
            assert np.all(bags[0]['songs'] == bags[cur_bag_idx]['songs'])
        except AssertionError:
            print('Song order is not identical. Evaluation is corrupt!')

    # model bagging
    predictions = []
    songs = bags[0]['songs']

    # collect data for model bagging
    if args.bagging > 1:
        for cur_song_idx in range(len(songs)):
            cur_predictions = []

            for cur_bag in bags:
                cur_predictions.append(cur_bag['predictions'][cur_song_idx])

            # average predictions
            cur_prediction = np.mean(np.asarray(cur_predictions), axis=0)
            predictions.append(cur_prediction)
    else:
        predictions = bags[0]['predictions']

    # evaluation
    # Evaluate with 0.5 s tolerance
    fps = config['fs'] / (config['hop_size'] * config['subsampling'])
    F_05, P_05, R_05 = evaluate(songs, predictions, annotations, window=0.5,
                                feature_rate=fps, threshold=thresholds['thresh_05'], 
                                musical_only=args.musical_only)
    print('0.5 s window: P={0:.3f}, R={1:.3f}, F={2:.3f}'.format(P_05, R_05, F_05))

    # Evaluate with 3.0 s tolerance
    F_3, P_3, R_3 = evaluate(songs, predictions, annotations, window=3.0,
                             feature_rate=fps, threshold=thresholds['thresh_3'],
                             musical_only=args.musical_only)
    print('3.0 s window: P={0:.3f}, R={1:.3f}, F={2:.3f}'.format(P_3, R_3, F_3))
