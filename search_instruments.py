"""
    Small script to let you search for instrument combinations etc.
"""
import os
import glob
import pandas as pd
from jsd_utils import load_jsd


if __name__ == '__main__':

    # setting global variables
    path_output = 'general_statistics'
    path_data = 'data'
    path_annotations = os.path.join(path_data, 'annotations_csv')
    annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))

    os.makedirs(os.path.join(path_data, path_output), exist_ok=True)

    track_db = load_jsd(annotation_files)
    nr_tracks = len(track_db['trackname'].unique())

    for cur_idx, cur_group in track_db.groupby('trackname'):
        if (len(cur_group[cur_group['instrument'] == 'tp']) > 0 and
            len(cur_group[cur_group['instrument'] == 'tb'])):
            pd.set_option('display.width', 1000)
            print(cur_group)
