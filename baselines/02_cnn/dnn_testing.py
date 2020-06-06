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


def predict(path_data, path_inputs, path_targets, predict_files,
            path_model, path_weights, config):

    # Model reconstruction from JSON file
    with open(path_model, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(path_weights)

    predictions = []
    gts = []
    songs = []

    for cur_path_song in tqdm.tqdm(predict_files):
        # streamer with just a single song
        cur_path_features = os.path.join(path_data, path_inputs, '{}.npz'.format(cur_path_song))
        cur_path_targets = os.path.join(path_data, path_targets, '{}.npz'.format(cur_path_song))

        stream = ds.stream_generator(cur_path_features, 'f_mel', cur_path_targets, 'target', patch_width=config['input_shape'][0],
                                     flatten_X=False, shuffle=False, add_dimension=True, target_smear=False,
                                     class_balance=False, subsampling=config['subsampling'])
        mini_batch_generator = pescador.buffer_stream(stream, 8)

        # predict for each patch
        cur_predictions = []
        cur_gts = []
        cur_song = cur_path_song

        for cur_batch in mini_batch_generator:
            cur_predictions.extend(model.predict_on_batch(cur_batch['X']).squeeze())
            cur_gts.extend(cur_batch['y'].squeeze())

        predictions.append(cur_predictions)
        gts.append(cur_gts)
        songs.append(cur_song)

    return predictions, gts, songs


def evaluate(songs, predictions, gts, window, fps, threshold):

    Fs = []
    Ps = []
    Rs = []

    for cur_song_id, cur_song in enumerate(songs):
        cur_pred = predictions[cur_song_id]
        cur_gt = gts[cur_song_id]

        estimated_onsets = utils.detect_peaks(np.asarray(cur_pred), fps=fps, threshold=threshold)
        reference_onsets = np.where(np.asarray(cur_gt) == 1.0)[0] / fps

        F, P, R = mir_eval.onset.f_measure(reference_onsets,
                                           estimated_onsets,
                                           window=window)

        # print('#{}: F: {}, P: {}, R: {}'.format(cur_song_id, F, P, R))
        Fs.append(F)
        Rs.append(R)
        Ps.append(P)

    return np.mean(Fs), np.mean(Ps), np.mean(Rs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Testing')
    parser.add_argument('--path_data', type=str, default='data')
    parser.add_argument('--path_inputs', type=str, default='salami_features')
    parser.add_argument('--path_targets', type=str, default='salami_targets')
    parser.add_argument('--path_results', type=str)
    parser.add_argument('--path_split', type=str, default='data/salami_split.yml', help='Path to split yml.')
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--bagging', type=int, default=1, help='Number of networks to train.')

    args = parser.parse_args()

    config = utils.load_config(os.path.join(args.path_results, 'config.yml'))
    path_test = os.path.join(args.path_data, 'salami_test.txt')
    cur_split_name = os.path.splitext(os.path.basename(args.path_split))[0]

    bags = []

    for cur_bag_idx in range(args.bagging):
        # load pre-computed thresholds for peak picking
        try:
            fn_split = 'peak_picking_thresholds-{}-{}.yml'.format(cur_split_name, cur_bag_idx)
            with open(os.path.join(args.path_results, fn_split), 'rb') as fp:
                thresholds = yaml.load(fp, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print('No peak_picking_thresholds.yml found. Please run "optimize_peak_picking.py" first.')
            sys.exit()

        path_model = os.path.join(args.path_results, 'architecture-{}.json'.format(cur_bag_idx))
        path_weights = os.path.join(args.path_results, 'weights-{}.h5'.format(cur_bag_idx))
        path_pred = os.path.join(args.path_results, 'pred-test-{}.npz'.format(cur_bag_idx))
        data = dict()

        if args.eval_only:
            saved_data = np.load(path_pred)
            data['predictions'] = saved_data['predictions']
            data['gts'] = saved_data['gts']
            data['songs'] = saved_data['songs']
        else:
            predictions, gts, songs = predict(args.path_data, args.path_inputs, args.path_targets,
                                              path_test, path_model, path_weights, config=config)
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
    gts = bags[0]['gts']

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
    F_05, P_05, R_05 = evaluate(songs, predictions, gts, window=0.5, fps=fps, threshold=thresholds['thresh_05'])
    print('0.5 s window: F={0:.3f}, P={1:.3f}, R={2:.3f}'.format(F_05, P_05, R_05))

    # Evaluate with 3.0 s tolerance
    F_3, P_3, R_3 = evaluate(songs, predictions, gts, window=3.0, fps=fps, threshold=thresholds['thresh_3'])
    print('3.0 s window: F={0:.3f}, P={1:.3f}, R={2:.3f}'.format(F_3, P_3, R_3))
