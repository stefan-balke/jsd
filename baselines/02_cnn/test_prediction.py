"""
    Loop over the test set and calculate accuracy.
"""
import os
import argparse
import numpy as np
import pandas as pd
import data_streamer as ds
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import utils


def main(path_data, path_inputs, path_targets, path_test,
         path_model, path_weights):

    dataset_test = pd.read_csv(path_test)['id'].tolist()
    dataset_test = [1346, ]

    # Model reconstruction from JSON file
    with open(path_model, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(path_weights)

    for cur_path_song in dataset_test:
        # streamer with just a single song
        path_features = os.path.join(path_data, path_inputs, '{}.npz'.format(cur_path_song))
        path_targets = os.path.join(path_data, path_targets, '{}.npz'.format(cur_path_song))

        stream = ds.stream_generator(path_features, 'f_mel', path_targets, 'target', patch_width=116,
                                     flatten_X=False, shuffle=False, add_dimension=True, target_smear=False,
                                     class_balance=False, subsampling=6)

        # predict for each patch
        predictions = []
        gt = []

        for cur_patch in stream:
            cur_X = cur_patch['X']
            cur_X = cur_X[None, ...]
            predictions.append(model.predict(cur_X))
            gt.append(cur_patch['y'])

        predictions = np.squeeze(predictions)
        gt = np.squeeze(gt)

        fps = 44100 / (1024 * 6)
        peaks = (utils.detect_peaks(predictions, fps=fps) * fps).astype('int')

        fig, ax = plt.subplots(1, figsize=(20, 3))
        ax.plot(predictions)
        ax.vlines(peaks, colors='r', linestyles='dotted', ymin=0, ymax=1.1)
        ax.vlines(np.where(gt == 1.0), colors='g', ymin=0.9, ymax=1.1)
        ax.set_ylim([0, 1.1])
        plt.savefig('{}.pdf'.format(dataset_test[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Testing')
    parser.add_argument('--path_data', type=str, default='data')
    parser.add_argument('--path_inputs', type=str, default='salami_features')
    parser.add_argument('--path_targets', type=str, default='salami_targets')
    parser.add_argument('--path_model', type=str, default='results/cnn_ismir2014_architecture.json')
    parser.add_argument('--path_weights', type=str, default='results/cnn_ismir2014_weights.h5')
    args = parser.parse_args()

    path_test = os.path.join(args.path_data, 'salami_test.txt')

    main(args.path_data, args.path_inputs, args.path_targets,
         path_test, args.path_model, args.path_weights)
