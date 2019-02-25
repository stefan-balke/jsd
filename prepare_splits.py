"""
    Takes all the annotation files in the specified folders and
    uses scikit-learn to create k folds for cross-validation,
    each with train, validation, test splits, respectively.
    Biasing "album effects" are avoided by never putting songs
    from the same record into train and test set.
"""
import os
import glob
import tqdm
import yaml
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold, train_test_split


def main():
    RANDOM_STATE = 4711
    N_FOLDS = 10
    PATH_DATA = 'data'
    PATH_X = os.path.join(PATH_DATA, 'features_wjd_audio_solos')
    PATH_y = os.path.join(PATH_DATA, 'annotations_csv')
    PATH_SPLITS = 'splits'

    # create output folder
    os.makedirs(PATH_SPLITS, exist_ok=True)

    # initialize output data
    # a dataset is a list which consists of N samples
    # a sample is a dict with three keys: {'fn_X', 'fn_y', 'recordid'}
    dataset = []

    # read relationships table for labels
    relationships = pd.read_csv(os.path.join(PATH_DATA, 'track_relationships.csv'))

    # loop through annotations y
    for cur_path_y in tqdm.tqdm(glob.glob(os.path.join(PATH_y, '*.csv'))):
        cur_track_name = os.path.splitext(os.path.split(cur_path_y)[1])[0]

        # select relationship by filename_solo and set recordid as label
        cur_label = relationships[relationships['filename_track'].str.contains(cur_track_name, regex=False)]
        cur_label = cur_label['recordid'].tolist()[0]

        cur_sample = {'track_name': cur_track_name, 'recordid': cur_label}
        dataset.append(cur_sample)

    # convert to DataFrame and shuffle
    dataset = pd.DataFrame(dataset)
    dataset = sklearn.utils.shuffle(dataset, random_state=RANDOM_STATE).reset_index(drop=True)

    # Do labeled k-fold splits on the recordid to avoid album effects
    lkf = GroupKFold(n_splits=N_FOLDS)

    for cur_fold, (cur_train_idx, cur_test_idx) in enumerate(lkf.split(dataset['track_name'], groups=dataset['recordid'])):
        cur_fold_test = cur_test_idx

        # further split training data in train and validation
        cur_fold_train, cur_fold_val = train_test_split(cur_train_idx, test_size=0.20, random_state=RANDOM_STATE)

        print('Fold %02d:\n n_train: \t%03d\n n_test: \t%03d\n n_val: \t%03d'
              % (cur_fold, len(cur_fold_train), len(cur_fold_test), len(cur_fold_val)))

        # get corresponding samples from dataset
        subset_train = dataset.iloc[cur_fold_train]
        subset_val = dataset.iloc[cur_fold_val]
        subset_test = dataset.iloc[cur_fold_test]

        # check that recordids from train set do not appear in test set
        for cur_idx, cur_row in subset_train.iterrows():
            try:
                assert (subset_test['recordid'] == cur_row['recordid']).sum() == 0
            except AssertionError:
                print('Fold {}: Overlapping RecordID ({}) detected!'.format(cur_fold, cur_row['recordid']))

        # save as YAML
        fold_output = {'train': subset_train.track_name.values.tolist(),
                       'val': subset_val.track_name.values.tolist(),
                       'test': subset_test.track_name.values.tolist()}

        with open(os.path.join(PATH_SPLITS, 'jsd_fold-{}.yml'.format(cur_fold)), 'w') as yaml_file:
            yaml.dump(fold_output, yaml_file, default_flow_style=False)


if __name__ == '__main__':
    main()
