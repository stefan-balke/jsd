import os
import tqdm
import numpy as np
import pandas as pd
import re
import json


def load_jsd(path_annotation_files):
    """Read JSD annotations from CSV files.

    Each row in the output is an annotation instance.

    Parameters
    ----------
    path_annotation_files : list
        List with paths of the annotation CSVs

    Returns
    -------
    annotations : pd.DataFrame
        All annotations as a pandas DataFrame.
        Columns: ['index', 'region_start', 'region_end', 'label', 'instrument',
                  'track_name', 'segment_class', 'segment_class_id', 'segment_chorus_id',
                  'instrument_solo', 'instrument_acc', 'mixed_solo']
    """

    annotations = pd.DataFrame(None)

    # load CSV files
    for cur_path_anno in tqdm.tqdm(path_annotation_files, desc='Loading Annotations'):

        track_name = os.path.splitext(os.path.basename(cur_path_anno))[0]

        # read annotation file
        cur_csv = pd.read_csv(cur_path_anno, usecols=[0, 1, 2, 3], sep=';')
        cur_csv['track_name'] = track_name

        annotations = annotations.append(cur_csv)

    annotations = annotations.reset_index()
    annotations['segment_class'] = ''
    annotations['segment_class_id'] = np.nan
    annotations['segment_chorus_id'] = np.nan
    annotations['instrument_solo'] = ''
    annotations['instrument_acc'] = ''
    annotations['mixed_solo'] = np.nan

    # Parse annotations and expand metadata
    for cur_index, cur_row in annotations.iterrows():
        # set segment_class and ids
        if (
            'silence' in cur_row['label'] or
            'intro' in cur_row['label'] or
            'outro' in cur_row['label']
        ):
            annotations.at[cur_index, 'segment_class'] = cur_row['label']

        if 'solo' in cur_row['label'] or 'theme' in cur_row['label']:
            try:
                matches = re.findall(r'(\w*)_(\d{2})_(\d{2})', cur_row['label'])[0]
                annotations.at[cur_index, 'segment_class'] = matches[0]
                annotations.at[cur_index, 'segment_class_id'] = int(matches[1])
                annotations.at[cur_index, 'segment_chorus_id'] = int(matches[2])
            except IndexError:
                print('Problem parsing: {}'.format(cur_row['track_name']))
                print(cur_row)

        if 'solo' in cur_row['label']:
            # set solo instrument and accompaniment
            instruments = cur_row['instrument'].split(',')
            instr_solo = []
            instr_acc = []

            # check if instrument is a solo instrument or accompaniment
            for cur_instr in instruments:
                try:
                    indicator, cur_instr_str = cur_instr.split('_')
                except ValueError:
                    print('Detected mal-formatted label field.')
                    print(cur_row)

                # sometimes we have more than a single ts
                # but this is not helpful for the instrument classes,
                # thus we normalize
                if cur_instr_str == 'ts1' or cur_instr_str == 'ts2':
                    cur_instr_str = 'ts'

                if cur_instr_str == 'tp1' or cur_instr_str == 'tp2':
                    cur_instr_str = 'tp'

                if cur_instr_str == 'as1' or cur_instr_str == 'as2':
                    cur_instr_str = 'as'

                if indicator == 's':
                    instr_solo.append(cur_instr_str)

                if indicator == 'b':
                    instr_acc.append(cur_instr_str)

            annotations.at[cur_index, 'instrument_solo'] = ','.join(instr_solo)
            annotations.at[cur_index, 'instrument_acc'] = ','.join(instr_acc)

            # is it a mixed solo, e.g., trading 4s...
            if len(instr_solo) == 1:
                annotations.at[cur_index, 'mixed_solo'] = 0

            if len(instr_solo) > 1:
                annotations.at[cur_index, 'mixed_solo'] = 1

    # add part duration as additional field
    annotations['segment_dur'] = annotations['segment_end'] - annotations['segment_start']

    return annotations


def filter_db_by_solo(track_db):
    # work on copy
    track_db_filtered = track_db.copy(deep=True)
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] == 'solo']  # only consider solos

    # init solos df
    track_db_solos = pd.DataFrame(columns=track_db_filtered.columns.to_list())

    # group by track_name
    for _, cur_track_df in track_db_filtered.groupby('track_name'):
        groups_solo = cur_track_df.groupby('segment_class_id')
        solos_first = groups_solo.first()
        solos_last = groups_solo.last()

        # keep solos_first as basis and update segment_end
        solos_merge = solos_first
        solos_merge['segment_end'] = solos_last['segment_end']

        track_db_solos = track_db_solos.append(solos_merge, ignore_index=True)

    # update durations
    track_db_solos['segment_dur'] = track_db_solos['segment_end'] - track_db_solos['segment_start']

    return track_db_solos


def get_instruments():
    instruments = pd.read_csv('data/instruments.csv', sep=';')

    return instruments


def get_boundaries(track_data):
    """Helper function to go from segments to boundaries.
    Start and end positions are concatenated and only the unique values survive.
    """
    boundaries = np.unique(list(track_data['segment_start'].values) +
                           list(track_data['segment_end'].values))

    return np.sort(boundaries)
