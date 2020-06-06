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
    parser.add_argument('--path_inputs', type=str, default='salami_features')
    parser.add_argument('--path_targets', type=str, default='salami_targets')
    parser.add_argument('--path_results', type=str)
    parser.add_argument('--path_split', type=str, default='data/salami_split.yml', help='Path to split yml.')
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--bagging', type=int, default=1, help='Number of networks to train.')

    args = parser.parse_args()

    config = utils.load_config(os.path.join(args.path_results, 'config.yml'))
    fps = config['fs'] / (config['hop_size'] * config['subsampling'])
    bags = []
    cur_split_name = os.path.splitext(os.path.basename(args.path_split))[0]

    predict_files = None
    with open(args.path_split) as fh:
        predict_files = yaml.load(fh, Loader=yaml.FullLoader)

    predict_files = predict_files['val']

    for cur_bag_idx in range(args.bagging):
        print('Model {}:'.format(cur_bag_idx))

        path_model = os.path.join(args.path_results, 'architecture-{}.json'.format(cur_bag_idx))
        path_weights = os.path.join(args.path_results, 'weights-{}.h5'.format(cur_bag_idx))
        path_pred = os.path.join(args.path_results, 'pred-valid-{}.npz'.format(cur_bag_idx))
        fps = config['fs'] / (config['hop_size'] * config['subsampling'])
        data = dict()

        if args.eval_only:
            saved_data = np.load(path_pred, allow_pickle=True)
            data['predictions'] = saved_data['predictions']
            data['gts'] = saved_data['gts']
            data['songs'] = saved_data['songs']
        else:
            predictions, gts, songs = predict(args.path_data, args.path_inputs, args.path_targets,
                                              predict_files, path_model, path_weights, config=config)
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

    # Calculate optimal threshold parameter on validation set
    # perform threshold sweep
    thresholds = np.arange(0.05, 0.95, 0.05)
    F_05_max = 0
    F_3_max = 0

    for cur_threshold in tqdm.tqdm(thresholds):
        # Evaluate with 0.5 s tolerance
        F_05, _, _ = evaluate(songs, predictions, gts, window=0.5, fps=fps, threshold=cur_threshold)

        # Evaluate with 3.0 s tolerance
        F_3, _, _ = evaluate(songs, predictions, gts, window=3.0, fps=fps, threshold=cur_threshold)

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
