import os
import tqdm
import numpy as np
import pandas as pd
import re


def load_jsd(path_annotation_files):
    """Read JSD annotations from CSV files.

    Each row in the output is an annotation instance.

    Parameters
    ----------
    path_annotation_files : list
        List with pathes of the annotation CSVs

    Returns
    -------
    annotations : pd.DataFrame
        All annotations as a pandas DataFrame.
        Columns: ['index', 'region_start', 'region_end', 'label', 'instrument',
                  'track_name', 'segment_class', 'segment_class_id', 'region_chorus_id',
                  'instrument_solo', 'instrument_acc']
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
    annotations['region_chorus_id'] = np.nan
    annotations['instrument_solo'] = ''
    annotations['instrument_acc'] = ''

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
                annotations.at[cur_index, 'region_chorus_id'] = int(matches[2])
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

    return annotations
