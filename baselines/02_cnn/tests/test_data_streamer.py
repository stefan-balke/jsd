import os
import pescador
import data_streamer


def test_shape():
    """Test mini-batch shapes."""

    PATH_DATA = 'data'
    PATH_X = os.path.join(PATH_DATA, 'salami_features')
    PATH_y = os.path.join(PATH_DATA, 'salami_targets')
    PATCH_WIDTH = 116
    MINI_BATCH_SIZE = 64
    TESTFILE = '1358'

    dataset_train = [TESTFILE, ]

    streamer_train = data_streamer.data_streamer(PATH_X, dataset_train, 'f_mel',
                                                 PATH_y, dataset_train, 'target',
                                                 PATCH_WIDTH, mini_batch_size=MINI_BATCH_SIZE,
                                                 target_smear=10, add_dimension=True)
    streamer_train = pescador.tuples(streamer_train, 'X', 'y')

    # get 10 mini-batches
    for _ in range(10):
        cur_mini_batch = next(streamer_train)
        assert cur_mini_batch[0].shape == (MINI_BATCH_SIZE, PATCH_WIDTH, 80, 1)
        assert cur_mini_batch[1].shape == (MINI_BATCH_SIZE,)


def test_smearing():
    """Test if smearing works."""
    pass


def test_subsampling():
    """Test is shape is correct after subsampling"""
    pass
