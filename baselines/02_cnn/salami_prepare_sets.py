"""
    Given a fixed test set, create train and validation fold.
"""
import os
import pandas as pd
import glob
import numpy as np


if __name__ == '__main__':
    PATH_DATA = 'data'
    PATH_ANNOTATIONS = os.path.join(PATH_DATA, 'salami_annotations')

    # load available annotations filenames
    salami = [int(os.path.splitext(os.path.basename(cur_song))[0])
              for cur_song in glob.glob(os.path.join(PATH_ANNOTATIONS, '*.txt'))]
    len_salami_org = len(salami)
    print('SALAMI: {}'.format(len(salami)))

    # load fixed test set
    test = pd.read_csv('data/salami_test.txt')
    print('Test: {}'.format(len(test)))

    # remove test set from salami
    for cur_test in test['id'].tolist():
        if cur_test in salami:
            salami.remove(cur_test)

    print('SALAMI - Test: {}'.format(len(salami)))
    assert len(salami) + len(test) == len_salami_org

    # this is the train set
    train = salami.copy()

    # draw validation set from train set
    val = np.random.choice(salami, 100, replace=False)

    # remove validation set from train set
    for cur_val in val:
        if cur_val in train:
            train.remove(cur_val)
    print('Train: {}'.format(len(train)))

    print('Validation: {}'.format(len(val)))
    assert len(train) + len(val) == len(salami)

    # dump to text files
    with open('salami_train.txt', 'w') as f:
        f.write('id\n')
        for cur_line in train:
            f.write('%s\n' % cur_line)

    with open('salami_val.txt', 'w') as f:
        f.write('id\n')
        for cur_line in val:
            f.write('%s\n' % cur_line)
