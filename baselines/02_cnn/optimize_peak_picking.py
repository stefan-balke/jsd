"""
    Optimize Peak Picking thresholds on the validation data.
"""
import os
import argparse
import numpy as np
import utils
import yaml
import tqdm

from dnn_testing import predict, evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Testing')
    parser.add_argument('--path_data', type=str, default='data')
    parser.add_argument('--path_results', type=str)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--bagging', type=int, default=1, help='Number of bagged networks.')

    args = parser.parse_args()

    config = utils.load_config(os.path.join(args.path_results, 'config.yml'))
    feature_rate = config['fs'] / (config['hop_size'] * config['subsampling'])

    # collect pathes
    PATH_X = [os.path.join(args.path_data, cur_path) for cur_path in config['input_data']]
    PATH_y = [os.path.join(args.path_data, cur_path) for cur_path in config['target_data']]

    # collect split data
    splits = []
    for cur_path in config['split_data']:
        with open(cur_path) as fh:
            splits.append(yaml.load(fh, Loader=yaml.FullLoader))

    # prepare path to data for validation set
    pathes_val_X = []
    for cur_ds_id, cur_path_X in enumerate(PATH_X):
        for cur_fn in splits[cur_ds_id]['val']:
            pathes_val_X.append(os.path.join(cur_path_X, '{}.npz'.format(cur_fn)))

    pathes_val_y = []
    for cur_ds_id, cur_path_y in enumerate(PATH_y):
        for cur_fn in splits[cur_ds_id]['val']:
            pathes_val_y.append(os.path.join(cur_path_y, '{}.npz'.format(cur_fn)))

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
        print('Model {}:'.format(cur_bag_idx))

        path_model = os.path.join(args.path_results, 'architecture-{}.json'.format(cur_bag_idx))
        path_weights = os.path.join(args.path_results, 'weights-{}.h5'.format(cur_bag_idx))
        path_pred = os.path.join(args.path_results, 'pred-valid-{}.npz'.format(cur_bag_idx))
        data = dict()

        if args.eval_only:
            saved_data = np.load(path_pred, allow_pickle=True)
            data['predictions'] = saved_data['predictions']
            data['gts'] = saved_data['gts']
            data['songs'] = saved_data['songs']
        else:
            predictions, gts, songs = predict(pathes_val_X, pathes_val_y,
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

    # Calculate optimal threshold parameter on validation set
    # perform threshold sweep
    thresholds = np.arange(0.05, 0.95, 0.05)
    F_05_max = 0
    F_3_max = 0

    for cur_threshold in tqdm.tqdm(thresholds):
        # Evaluate with 0.5 s tolerance
        F_05, _, _ = evaluate(songs, predictions, annotations, window=0.5, feature_rate=feature_rate, threshold=cur_threshold)

        # Evaluate with 3.0 s tolerance
        F_3, _, _ = evaluate(songs, predictions, annotations, window=3.0, feature_rate=feature_rate, threshold=cur_threshold)

        if F_05_max < F_05:
            thresh_05 = cur_threshold
            F_05_max = F_05

        if F_3_max < F_3:
            thresh_3 = cur_threshold
            F_3_max = F_3

    # save thresholds
    data = {'thresh_05': float(thresh_05), 'thresh_3': float(thresh_3)}
    file_name = os.path.join(args.path_results, 'peak_picking_thresholds.yml')

    with open(file_name, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
