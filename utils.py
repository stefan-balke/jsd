import os
import tqdm
import pandas as pd


def create_track_db(annotation_files):

    track_db = pd.DataFrame(None)

    for cur_path_anno in tqdm.tqdm(annotation_files):

        track_name = os.path.splitext(os.path.basename(cur_path_anno))[0]

        # read annotation file
        annotations = pd.read_csv(cur_path_anno, usecols=[0, 1, 2, 3], sep=';')

        # create empty index list
        idx_list = []

        # find index of not needed rows (non solo parts)
        for cur_index, cur_annotation in annotations.iterrows():
            if 'silence' in cur_annotation['label'] or 'intro' in cur_annotation['label'] \
                                                    or 'outro' in cur_annotation['label'] \
                                                    or 'theme' in cur_annotation['label'] \
                                                    or 'all' in cur_annotation['instrument']:
                idx_list = idx_list + [cur_index]

        # drop the rows which are not needed
        annotations = annotations.drop(annotations.index[idx_list])

        annotations['trackname'] = track_name
        # add dataframe of one track to dataframe of all songs
        # if track_name != 'JohnColtrane_GiantSteps_Orig':
        track_db = track_db.append(annotations)

    track_db = track_db.reset_index()

    # extract the first string between _ and (, or end) of str
    track_db['instrument'] = track_db['instrument'].str.extract('(?<=\_)(.*?)(?=\,|$)')

    # convert *1 and *2 to *
    for cur_index, cur_annotation in track_db.iterrows():
        if cur_annotation['instrument'] == 'ts1' or cur_annotation['instrument'] == 'ts2':
            track_db.loc['instrument', cur_index] = 'ts'
        if cur_annotation['instrument'] == 'tp1' or cur_annotation['instrument'] == 'tp2':
            track_db.loc['instrument', cur_index] = 'tp'
        if cur_annotation['instrument'] == 'as1' or cur_annotation['instrument'] == 'as2':
            track_db.loc['instrument', cur_index] = 'as'

    return track_db
