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
    for cur_path_anno in tqdm.tqdm(path_annotation_files, desc='Loading JSD'):

        track_name = os.path.splitext(os.path.basename(cur_path_anno))[0]

        # read annotation file
        cur_csv = pd.read_csv(cur_path_anno, usecols=[0, 1, 2, 3], sep=';')
        cur_csv['track_name'] = track_name

        cur_csv = flag_non_musical_segments(cur_csv)

        annotations = annotations.append(cur_csv)

    annotations = annotations.reset_index(drop=True)
    annotations['segment_class'] = ''
    annotations['segment_class_id'] = np.nan
    annotations['segment_chorus_id'] = np.nan
    annotations['instrument_solo'] = ''
    annotations['n_instruments_solo'] = np.nan
    annotations['instrument_acc'] = ''
    annotations['n_instruments_acc'] = np.nan
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
            annotations.at[cur_index, 'n_instruments_solo'] = len(instr_solo)
            annotations.at[cur_index, 'instrument_acc'] = ','.join(instr_acc)
            annotations.at[cur_index, 'n_instruments_acc'] = len(instr_acc)

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


def get_boundaries(track_data, musical_only=False):
    """Helper function to go from segments to boundaries.
    Start positions of each segment are taken as boundaries and only the unique values survive.

    Parameters
    ----------
    track_data : pd.DataFrame
        DataFrame containing JSD's annotations.
    musical_only : boolean
        Filter boundaries to musical boundaries, i.e. containing no silence boundaries
        and only boundaries which are surrounded by musical parts.

    Returns
    -------
    boundaries : np.ndarray, shape=(N, 1)
        Boundary positions.
    """
    cur_track_data = track_data.copy()

    if musical_only:
        cur_track_data = cur_track_data[cur_track_data['is_musical'] == True]

    boundaries = np.unique(list(cur_track_data['segment_start'].values))

    return np.sort(boundaries)


def flag_non_musical_segments(track_data):
    """Filter segments to musical boundaries, i.e. containing no silence boundaries
        and only boundaries which are surrounded by musical parts.

    Parameters
    ----------
    track_data : pd.DataFrame
        DataFrame containing JSD's annotations.

    Returns
    -------
    segments : pd.DataFrame, shape=(N, 2)
        Start and end positions from the boundary.
    """
    cur_track_data = track_data.copy()

    cur_track_data['is_musical'] = True
    # drop_idcs = cur_track_data[cur_track_data['label'] == 'silence'].index.tolist()
    # drop_idcs.extend(cur_track_data[cur_track_data['label'] == 'end'].index.tolist())
    # first and last bounardy are always non-musical
    non_musical_idcs = [cur_track_data.index[0], cur_track_data.index[-1]]

    # filter all boundaries to musical boundaries
    for cur_idx in range(1, len(cur_track_data) - 1):
        prev_segment = cur_track_data.iloc[cur_idx - 1]['label']
        curr_segment = cur_track_data.reset_index().iloc[cur_idx]
        next_segment = cur_track_data.iloc[cur_idx + 1]['label']

        # filter trivial boundaries like silence->intro or outro->silence
        # check if surrounding segments contain music
        if (prev_segment == 'silence') or (next_segment == 'silence') or (next_segment == 'end'):
            non_musical_idcs.append(curr_segment['index'])

        # non-musical segments in salami dataset
        if (prev_segment == 'z') or (next_segment == 'z'):
            non_musical_idcs.append(curr_segment['index'])

    cur_track_data.loc[cur_track_data.index.isin(non_musical_idcs), 'is_musical'] = 'False'

    return cur_track_data
