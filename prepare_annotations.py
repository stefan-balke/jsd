"""
    This script adds silence segments at the start and end of the annotation files.
"""
import os
import glob
import pandas as pd
import tqdm


def add_silence(annotation, track_dur, track_name):
    # determine the end of the last region and add a silence segment
    # from the end of the last segement to the end of the file
    end_last_region = annotation['segment_start'].values[len(annotation) - 1] \
                      + annotation['segment_dur'].values[len(annotation) - 1]
    end_silence = pd.DataFrame([[end_last_region, track_dur - end_last_region, 'silence']],
                               columns=annotation.columns.values)
    end_silence = end_silence.set_index(end_silence.index + len(annotation) + 1)

    # add a silence segment from the start of the file to the start of the first segment
    annotation = annotation.set_index(annotation.index + 1)

    try:
        assert float(annotation['segment_start'].values[0]) >= 0
    except AssertionError:
        print('{}: Region starting point <0 detected.'.format(track_name))

    start = pd.DataFrame([[0, annotation['segment_start'].values[0], 'silence']],
                         columns=annotation.columns.values)

    # add both segments to the annotation file, indexes are set for correct segment position
    annotation = pd.concat([annotation, start])
    annotation = pd.concat([annotation, end_silence])
    annotation = annotation.sort_index()

    return annotation


def convert_to_regions(annotation, track_name, gap_threshold=1.0):
    # create empty pandas dataframe
    out_df = pd.DataFrame(None, columns=['segment_start', 'segment_end', 'label', 'instrument'])

    # loop over all rows from the annotations
    for cur_idx, cur_row in annotation.iterrows():
        # split labels into instruments and labels
        labels_raw = cur_row['label'].replace(' ', '').split(',')
        labels_raw = list(filter(None, labels_raw))

        # segment label
        label = labels_raw[0]

        # instruments
        instruments = ','.join(labels_raw[1:])

        # convert the duration column into segment_end
        cur_start = float(cur_row['segment_start'])

        # for the last segment we take the annotated end,
        # otherwise we take the start position of the next segment
        # to close any annotation gaps
        if cur_idx + 1 <= annotation.index[-1]:
            cur_end_adj = float(annotation.loc[cur_idx + 1, 'segment_start'])
        else:
            cur_end_adj = cur_start + float(cur_row['segment_dur'])

        # create dataframe for new row data and append it to out_df
        row = pd.DataFrame([[cur_start, cur_end_adj, label, instruments]],
                           columns=out_df.columns.values)

        out_df = pd.concat([out_df, row])

    return out_df


def main():
    # paths to annotations and output folder
    path_data = 'data'
    path_annotations = os.path.join(path_data, 'annotations_raw')
    path_out = os.path.join(path_data, 'annotations_csv')

    # make sure the folders exist before trying to save things
    os.makedirs(path_out, exist_ok=True)

    columnnames = ['Trackname', 'Duration']
    track_durs = pd.read_csv(os.path.join(path_data, 'track_durations.csv'), names=columnnames, sep=',')

    # loop over all annotation files
    for cur_annotation_file in tqdm.tqdm(glob.glob(os.path.join(path_annotations, '*.txt'))):
        # extract trackname
        cur_track_name = os.path.splitext(os.path.basename(cur_annotation_file))[0]

        # check if audio with same name exists, should be always the case
        try:
            assert cur_track_name in track_durs['Trackname'].values
        except AssertionError:
            print('Error: {} was not found in track durations.'.format(cur_track_name))
            break

        # create pandas dataframe and read raw annotations
        col_names = ['segment_start', 'segment_dur', 'label']
        cur_annotation = pd.read_csv(cur_annotation_file, names=col_names, usecols=[0, 2, 3], sep='\t')

        # read the duration from the meta data
        cur_track_dur = float(track_durs.loc[track_durs['Trackname'] == cur_track_name]['Duration'])

        # modification 1: add silence
        cur_annotation = add_silence(cur_annotation, cur_track_dur, cur_track_name)

        # modification 2: convert to start-end regions
        cur_annotation = convert_to_regions(cur_annotation, cur_track_name)

        # write the final annotations into a csv file
        cur_annotation.to_csv(os.path.join(path_out, cur_track_name + '.csv'), sep=';', index=False)


if __name__ == '__main__':
    main()
