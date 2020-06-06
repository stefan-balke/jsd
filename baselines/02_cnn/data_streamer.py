import numpy as np
import pescador
import os
import utils
import librosa

RANDOM_STATE = 20171010
ACTIVE_STREAMS = 100
np.random.seed(RANDOM_STATE)


def get_patch_X(t, n_t, data_X):
    """Get a time-frequency patch from a feature matrix with `K` rows

    Parameters
    ----------
    t : int
        First time index of the patch (because noise of length n_t/2 is added in the beginning of data_X)
    n_t : int
        Patch width
    data_X : np.ndarray
        Data matrix with feature values

    Returns
    -------
    patch : np.ndarray [shape=(K, n_t)]
        A path from the input feature matrix
    """

    if n_t == 1:
        out = data_X[:, t]
    else:
        out = data_X[:, t-int(n_t/2):t+int(n_t/2)]

    return np.atleast_2d(out).astype(np.float32)


def get_patch_y(t, data_y):
    """Given a TF-patch, get the label of the corresponding target frame (center frame)

    Parameters
    ----------
    t : int
        Time index of the patch
    data_y : np.ndarray
        Data matrix with target values

    Returns
    -------
    patch : np.ndarray [shape=(K, 1)]
        A path from the input feature matrix
    """
    return data_y[t].astype(np.float32)


def subsample(data_X, data_y, n):
    """Subsamples the input and target data by taking the maximum over n frames without overlap.
    Setting 'n = 1' equals no subsampling.

    Parameters
    ----------
    data_X : np.ndarray
        Input data
    data_y : np.ndarray
        Target data
    n : int
        Subsampling parameter

    Returns
    -------
    data_X : np.ndarray
        Subsampled input data
     data_y : np.ndarray
        Subsampled target data
    """

    if n > 1:
        # cut away last length modulo subsampling indices
        data_X = data_X[:, :data_X.shape[1] - (data_X.shape[1] % n)]
        data_y = data_y[:data_y.shape[0] - (data_y.shape[0] % n)]

        # reshape data to one dimension higher to take max of n frames
        data_X = np.reshape(data_X, (data_X.shape[0], int(data_X.shape[1] / n), n))
        data_y = np.reshape(data_y, (int(data_y.shape[0] / n), n))

        # take max
        data_X = np.amax(data_X, axis=2)
        data_y = np.amax(data_y, axis=1)

    return data_X, data_y


def smear_target(data_y, target_smear, win='rectangle'):
    """Smears the positive examples of the target data.

    Parameters
    ----------
    data_y : np.ndarray
        Target data
    target_smear : int
        Duration of the target smearing in frames
    win : str
        Type of window to be created. 'one' for Target window, 'gaussian'
        for gaussian window for loss function weighting

    Yields
    ------
     data_y : np.ndarray
        Target data with smeared boundaries.
    """

    # we force odd window sizes
    if np.mod(target_smear, 2) == 0:
        target_smear += 1

    # helper variable
    n_win_half = int(target_smear / 2)

    # init window
    if win == 'rectangle':
        kernel = np.ones(target_smear)

    if win == 'gaussian':  # fix if and elif for boundaries
        kernel = utils.gaussian(target_smear)

    # get all positive target indices
    boundaries = np.where(data_y == 1)[0]

    for cur_bdry in boundaries.tolist():
        # boundary is at the beginning and index is smaller than target smearing half
        if cur_bdry < n_win_half:
            data_y[:cur_bdry + n_win_half + 1] = kernel[n_win_half - cur_bdry:]

        # cur_boundary is at the end and index is bigger than (size + target smearing half)
        elif cur_bdry >= (data_y.shape[0] - n_win_half):
            data_y[cur_bdry - n_win_half:] = kernel[:n_win_half + data_y.shape[0] - cur_bdry]

        # boundary is in the middle of file and target smearing works
        else:
            data_y[cur_bdry - n_win_half:cur_bdry + n_win_half + 1] = kernel

    # sometimes, boundaries are too close, so windows overlap
    # we want to make sure that boundaries are always equal to one
    # Sorry for the hack...
    data_y[boundaries] = 1.0

    # check that boundaries are 1 at boundary positions
    try:
        assert np.array_equal(data_y[boundaries], np.ones(len(boundaries)))
    except AssertionError:
        print('Error: Not all boundaries are equal to 1.')

    return data_y


def stream_generator(path_X, key_X, path_y, key_y, patch_width,
                     flatten_X=False, shuffle=True, add_dimension=True, target_smear=False,
                     class_balance=0.5, subsampling=False):
    """Generator that yields samples from a feature and a corresponding label file.

    Parameters
    ----------
    path_X : str
        Path to the input data
    key_X : str
        Key for the input feature matrix
    path_y : str
        Path to the target data
    key_y : str
        Key for the target feature matrix
    patch_width : int n_t
        Size of the spectrogram patches in time (n_t) direction
    flatten_X : bool
        Some networks need flattened inputs (e.g., fully-connected)
    shuffle : bool
        Shuffle the time indices. Usually for training it is `True`, for testing `False`.
    add_dimension : bool
        Add dimension to input data, needed for settting channel dimension when creating 2D-CNN.
    target_smear : float
        Temporal smearing of the boundaries in frames.
    class_balance : float [0, 1]
        Balances the two classes for binary classifiction with oversampling.
        It controls the balance between boundaries and non-boundaries,
        e.g., a value of 0.5 imposes parity, a value of 0.70 will result in mini-batches
        with 70% boundaries and 30% non-boundaries.
        Deactivated with setting to `False`.
    subsampling : int
        Subsampling parameter

    Yields
    ------
    sample : dict
        A dictionary with `X` and `y` values for a single sample.
    """

    # load data
    data_X = np.load(path_X)[key_X]
    data_y = np.load(path_y)[key_y]

    # Data Preprocessing

    # logarithmic amplitudes
    data_X = librosa.power_to_db(data_X)

    # subsample data
    if subsampling:
        [data_X, data_y] = subsample(data_X, data_y, subsampling)

    # target smearing with rectangular window +-target_smear (seconds)
    weight_y = data_y.copy()

    if target_smear:
        data_y = smear_target(data_y, target_smear, win='rectangle')
        weight_y = smear_target(weight_y, target_smear, win='gaussian')

    weight_y[weight_y == 0] = 1  # every other are set to 1

    # get all t indices
    t_vals = np.arange(0, data_y.shape[0])

    # get boundary indices
    boundaries = np.where(data_y == 1)[0]
    non_boundaries = np.where(data_y == 0)[0]

    # add pink noise to data_X at the beginning and end of the song
    n_noise = 0

    if patch_width > 1:
        n_noise = int(patch_width / 2)
        noise = utils.add_noise_patch(n_noise)
        data_X = np.concatenate((noise, data_X, noise), axis=1)
        data_y = np.concatenate((np.zeros(n_noise), data_y, np.zeros(n_noise)))
        weight_y = np.concatenate((np.ones(n_noise), weight_y, np.ones(n_noise)))

    # offset t_vals and boundaries
    t_vals += n_noise
    boundaries += n_noise
    non_boundaries += n_noise

    # TODO: Combine these two loops...share A LOT of code...
    if shuffle:
        while True:
            if class_balance:
                # balance the classes through oversampling
                if np.random.rand() < class_balance:
                    cur_t = np.random.choice(boundaries)
                else:
                    cur_t = np.random.choice(non_boundaries)
            else:
                cur_t = np.random.choice(t_vals)

            # get spectrogram patch
            cur_X = get_patch_X(cur_t, patch_width, data_X)

            # check shape
            try:
                assert cur_X.shape == (data_X.shape[0], patch_width)
            except AssertionError:
                print('Shape error in sample selection: {}, t={}'.format(cur_X.shape, cur_t))

            # get corresponding center frame
            cur_y = get_patch_y(cur_t, data_y)

            # get corresponding weight for output
            cur_w = weight_y[cur_t]

            if flatten_X:
                cur_X = cur_X.flatten()

            if add_dimension:
                # transpose and add channel dimension
                # Shape: (mini_batch_size, patch_width, freqs, 1)
                yield dict(X=cur_X.T[:, :, None],
                           y=np.squeeze(cur_y),
                           w=np.squeeze(cur_w))
            else:
                # Shape: (mini_batch_size, patch_width, freqs)
                yield dict(X=cur_X.T, y=cur_y, w=cur_w)
    else:
        for cur_t in t_vals:
            # get spectrogram patch
            cur_X = get_patch_X(cur_t, patch_width, data_X)

            # get corresponding center frame
            cur_y = get_patch_y(cur_t, data_y)

            # get corresponding weight for output
            cur_w = weight_y[cur_t]

            if flatten_X:
                cur_X = cur_X.flatten()

            if add_dimension:
                # transpose and add channel dimension
                # Shape: (mini_batch_size, patch_width, freqs, 1)
                yield dict(X=cur_X.T[:, :, None],
                           y=np.squeeze(cur_y),
                           w=np.squeeze(cur_w))
            else:
                # Shape: (mini_batch_size, patch_width, freqs)
                yield dict(X=cur_X.T, y=cur_y, w=cur_w)


def data_streamer(pathes_X, key_X, pathes_y, key_y,
                  patch_width, mini_batch_size, flatten_X=False,
                  add_dimension=True, target_smear=False, class_balance=False, shuffle=True,
                  random_state=20171010, subsampling=False, rate=None, **kwargs):
    """Generator to be passed to a keras model

    Parameters
    ----------
    pathes_X : str or list
        Path to the input data
    key_X : str
        Key for the input feature matrix
    pathes_y : str or list
        Path to the target data
    key_y : str
        Key for the target feature matrix
    patch_width : int n_t
        Size of the spectrogram patches in time (n_t) direction
    mini_batch_size : int
        Mini-batch size
    flatten_X : bool
        Some networks need flattened inputs (e.g., fully-connected)
    add_dimension : bool
        Forwarded to stream_generator. Add dimension to input data, needed for setting channel dimension
        when creating 2D-CNN.
    target_smear : float
        Forwarded to stream_generator. Temporal smearing of the boundaries in frames.
    class_balance : float [0, 1], `False` to deactivate
        Forwarded to stream_generator. Balances the two classes for binary classifiction with oversampling.
    random_state: int
        Random state to create pseudo random numbers. Parameter needed for bagging
    subsampling : int
        Taking the max of N neighbouring frames and subsampling.
    **kwargs
        Forwarded to `pescador.Mux`

    Yields
    ------
    mini_batch : np.ndarray, shape=[mini_batch_size, n_f, n_t]
        A tensor for containing all the patches for a mini-batch
    """
    streams = []

    # get all file streams
    for cur_path_X, cur_path_y in zip(pathes_X, pathes_y):

        streams.append(pescador.Streamer(stream_generator,
                                         cur_path_X, key_X,
                                         cur_path_y, key_y,
                                         patch_width=patch_width, flatten_X=flatten_X,
                                         add_dimension=add_dimension, class_balance=class_balance,
                                         target_smear=target_smear, subsampling=subsampling,
                                         shuffle=shuffle))

    # mux all streams together
    stream_mux = pescador.StochasticMux(streams, n_active=ACTIVE_STREAMS, rate=rate,
                                        random_state=random_state, **kwargs)

    mini_batch_generator = pescador.buffer_stream(stream_mux, mini_batch_size)

    for mini_batch in mini_batch_generator:
        yield mini_batch


if __name__ == '__main__':
    import pandas as pd
    PATH_DATA = 'data'
    PATH_X = os.path.join(PATH_DATA, 'salami_features')
    PATH_y = os.path.join(PATH_DATA, 'salami_targets')
    PATCH_WIDTH = 116
    TESTFILE = '1358'

    path_train = os.path.join(PATH_DATA, 'salami_train.txt')
    dataset_train = pd.read_csv(path_train)['id'].tolist()

    dataset_train = [TESTFILE, ]

    streamer_train = data_streamer(PATH_X, dataset_train, 'f_mel',
                                   PATH_y, dataset_train, 'target',
                                   PATCH_WIDTH, mini_batch_size=64, target_smear=10,
                                   class_balance=0.5, add_dimension=True,
                                   rate=64, shuffle=True)
    streamer_train = pescador.tuples(streamer_train, 'X', 'y')

    for cur_mini_batch in streamer_train:
        print('Shape X: %s' % str(cur_mini_batch[0].shape))
        print('Shape y: %s' % str(cur_mini_batch[1].shape))
