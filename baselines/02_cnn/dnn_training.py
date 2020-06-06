"""
Example Call:
    `python dnn_training.py data one_song_overfit_set/test_features_pinknoise
     one_song_overfit_set/test_targetvector fold-00_train.csv`
"""

import shutil
import tensorflow.keras as K
import data_streamer as ds
import os
import pandas as pd
import argparse
import pescador
import networks
import utils
import datetime
import yaml


def train(args, random_state, bag_idx, path_network, path_results):
    """Main model training."""

    config = utils.load_config(args.config)
    fps = config['fs'] / (config['hop_size'] * config['subsampling'])

    # collect pathes
    PATH_X = [os.path.join(args.path_data, cur_path) for cur_path in config['input_data']]
    PATH_y = [os.path.join(args.path_data, cur_path) for cur_path in config['target_data']]

    # collect split data
    splits = []
    for cur_path in config['split_data']:
        with open(cur_path) as fh:
            splits.append(yaml.load(fh, Loader=yaml.FullLoader))

    # prepare path to data for training set
    pathes_train_X = []
    for cur_ds_id, cur_path_X in enumerate(PATH_X):
        for cur_fn in splits[cur_ds_id]['train']:
            pathes_train_X.append(os.path.join(cur_path_X, '{}.npz'.format(cur_fn)))

    pathes_train_y = []
    for cur_ds_id, cur_path_y in enumerate(PATH_y):
        for cur_fn in splits[cur_ds_id]['train']:
            pathes_train_y.append(os.path.join(cur_path_y, '{}.npz'.format(cur_fn)))

    streamer_train = ds.data_streamer(pathes_train_X, 'f_mel',
                                      pathes_train_y, 'target',
                                      config['input_shape'][0], mini_batch_size=args.mini_batch_size,
                                      rate=512, mode='with_replacement', add_dimension=True,
                                      class_balance=config['class_balance'], random_state=random_state,
                                      subsampling=config['subsampling'], shuffle=True,
                                      target_smear=int(config['target_smear'] * fps))

    # prepare path to data for validation set
    pathes_val_X = []
    for cur_ds_id, cur_path_X in enumerate(PATH_X):
        for cur_fn in splits[cur_ds_id]['val']:
            pathes_val_X.append(os.path.join(cur_path_X, '{}.npz'.format(cur_fn)))

    pathes_val_y = []
    for cur_ds_id, cur_path_y in enumerate(PATH_y):
        for cur_fn in splits[cur_ds_id]['val']:
            pathes_val_y.append(os.path.join(cur_path_y, '{}.npz'.format(cur_fn)))

    streamer_val = ds.data_streamer(pathes_val_X, 'f_mel',
                                    pathes_val_y, 'target',
                                    config['input_shape'][0], mini_batch_size=args.mini_batch_size,
                                    rate=None, mode='with_replacement', add_dimension=True,
                                    class_balance=False, random_state=random_state, shuffle=False,
                                    subsampling=config['subsampling'], target_smear=False)

    # get keras model
    model = networks.get_network(args.net)(config['input_shape'])
    model.summary()

    fn_architecture = 'architecture-{}.json'.format(bag_idx)
    fn_weights = 'weights-{}.h5'.format(bag_idx)
    fn_hist = 'hist_train-{}.csv'.format(bag_idx)

    # save model architecture
    with open(os.path.join(path_results, fn_architecture), 'w') as fp:
        fp.write(model.to_json())

    optimizer = K.optimizers.Adam(lr=0.001)

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    cb = list()
    cb.append(K.callbacks.ModelCheckpoint(os.path.join(path_results, fn_weights),
                                          save_best_only=True,
                                          verbose=1,
                                          monitor='loss'))
    cb.append(K.callbacks.ReduceLROnPlateau(patience=5,
                                            verbose=1,
                                            monitor='loss'))

    hist = model.fit_generator(pescador.tuples(streamer_train, 'X', 'y', 'w'),
                               epochs=args.n_epochs,
                               steps_per_epoch=args.epoch_size,
                               callbacks=cb,
                               validation_data=pescador.tuples(streamer_val, 'X', 'y', 'w'),
                               validation_steps=1024)

    model.save_weights(os.path.join(path_results, fn_weights), overwrite=True)

    # save loss history as csv
    hist = pd.DataFrame(hist.history)
    hist.to_csv(os.path.join(path_results, fn_hist))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN Training')
    parser.add_argument('--path_data', type=str, default='data')
    parser.add_argument('--path_results', type=str, default='results', help='Optional results path.')
    parser.add_argument('--bagging', type=int, default=1, help='Number of networks to train.')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--epoch_size', type=int, default=2000, help='Size of epochs.')
    parser.add_argument('--mini_batch_size', type=int, default=128, help='Mini batch size.')
    parser.add_argument('--net', type=str, default='cnn_ismir2014', help='Network name.')
    parser.add_argument('--config', type=str, default='configs/config_jsd_short.yml', help='Path to the config.')

    args = parser.parse_args()

    random_states = [4711, 1848, 1234, 42, 1984]

    # create results folders
    path_network = '{}-{}'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), args.net)
    path_results = os.path.join(args.path_results, path_network)
    os.makedirs(path_results, exist_ok=True)

    # copy config file
    shutil.copyfile(args.config, os.path.join(path_results, 'config.yml'))

    for cur_idx in range(args.bagging):
        # start training
        train(args, random_states[cur_idx], cur_idx, path_network, path_results)
