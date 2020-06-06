"""
    Tests the data streamer and plot target smearing and sample weighting.
"""
import os
import numpy as np
import data_streamer as ds
import matplotlib.pyplot as plt


if __name__ == '__main__':

    fn_test = '1304.npz'
    path_data = 'data'
    path_features = os.path.join(path_data, 'salami_features', fn_test)
    path_targets = os.path.join(path_data, 'salami_targets', fn_test)

    subsamling = 1
    target_smear = 1.5
    fps = 44100 / (1024 * subsamling)

    # streamer with just a single song
    stream = ds.stream_generator(path_features, 'f_mel', path_targets, 'target', patch_width=1,
                                 flatten_X=False, shuffle=False, add_dimension=True, target_smear=int(target_smear*fps),
                                 class_balance=False, subsampling=subsamling)

    X = []
    y = []
    w = []

    # get song from streamer
    for cur_frame in stream:
        X.append(np.squeeze(cur_frame['X']))
        y.append(cur_frame['y'])
        w.append(cur_frame['w'])

    X = np.asarray(X).T
    y = np.asarray(y).T
    w = np.asarray(w).T

    # plotting
    fig, ax = plt.subplots(3, sharex=True)
    x_max_s = 61
    x_max = x_max_s * fps
    ax[0].imshow(X, origin='lower', aspect='auto')
    # ax[0].vlines(np.where(y == 1), 0, 80, colors='r')
    ax[0].set_xlim(0, x_max)
    ax[1].plot(y)
    ax[1].set_xlim(0, x_max)
    ax[2].plot(w)
    ax[2].set_xlim(0, x_max)
    plt.suptitle('Song: {}, Subsampling: {}, Target Smearing: {} s'.format(os.path.splitext(fn_test)[0],
                                                                           subsamling, target_smear))
    plt.xlabel('Time (s)')
    plt.xticks((np.arange(0, x_max_s, 10) * fps).astype('int'), np.arange(0, x_max_s, 10))
    plt.savefig('test_data_streamer_{}.pdf'.format(subsamling))
