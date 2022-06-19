import pandas as pd
import os
import sys
import glob
import tqdm

# hacky relative import
sys.path.append(os.path.join('..'))
import jsd_utils


def load_salami(track_durs, path_annotations):
    salami_data = []

    for cur_track_path in tqdm.tqdm(glob.glob(os.path.join(path_annotations, '*.txt')), desc='Loading Salami'):
        cur_track_name = os.path.splitext(os.path.basename(cur_track_path))[0]

        try:
            cur_track_data = pd.read_csv(cur_track_path, sep='\t', header=None, names=['segment_start', 'label'])
        except FileNotFoundError:
            print(cur_track_path)
            continue

        cur_track_data['label'] = cur_track_data['label'].str.lower()
        cur_track_data['track_name'] = cur_track_name
        cur_track_data['segment_end'] = cur_track_data['segment_start'].shift(-1)
        cur_track_data['segment_dur'] = cur_track_data['segment_end'] - cur_track_data['segment_start']
        cur_track_data = cur_track_data[:-1]
        cur_track_data = jsd_utils.flag_non_musical_boundaries(cur_track_data)

        salami_data.append(cur_track_data)

    salami_data = pd.concat(salami_data)

    return salami_data


if __name__ == '__main__':
    PATH_DATA = os.path.join('data')
    path_annotations = os.path.join(PATH_DATA, 'salami_annotations')
    track_durs = pd.read_csv(os.path.join(PATH_DATA, 'salami_track_durations.csv'))
    track_durs = track_durs.astype(str)

    salami_track_db = load_salami(track_durs, path_annotations)
    track_names = salami_track_db['track_name'].unique()

    for cur_track_name in pd.Series(track_names).sample(10):
        cur_track = salami_track_db[salami_track_db['track_name'] == cur_track_name]
        cur_boundaries_ref = jsd_utils.get_boundaries(cur_track, musical_only=True)
        print(cur_track_name)
        print(cur_track)
        print(cur_boundaries_ref)
