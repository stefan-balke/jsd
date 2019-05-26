"""
    This script plots various statistics of the annotations from the dataset.
"""

import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import load_jsd
sns.set(style='white')
sns.set_context('paper', font_scale=1.6, rc={'lines.linewidth': 0.75})
sns.set_color_codes('muted')


def plot_hist_segments_per_track(track_db, path_output):
    """Histogram over number of segments per track

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
    """

    # consider all but silence
    track_db_filtered = track_db[track_db['segment_class'] != 'silence']
    data = pd.DataFrame({'count': track_db_filtered.groupby(['track_name']).size()}).reset_index()

    # plotting
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.countplot(x='count', data=data, color='b')
    ax.set_xlabel('#Segments')
    ax.set_ylabel('#Tracks')
    sns.despine()
    plt.tight_layout()
    plt.savefig(path_output, bbox_inches='tight')


def plot_hist_choruses_per_track(track_db, path_output):
    """Histogram over number of choruses per track

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
    """

    # filter out silence segments
    track_db_filtered = track_db[track_db['segment_class'] != 'silence']
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] != 'intro']
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] != 'outro']
    data = pd.DataFrame({'count': track_db_filtered.groupby(['track_name']).size()}).reset_index()

    # plotting
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.countplot(x='count', data=data, color='b')
    ax.set_xlabel('#Choruses')
    ax.set_ylabel('#Tracks')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'hist_choruses_per_track.pdf'), bbox_inches='tight')


def plot_hist_solos_per_track(track_db, path_output):
    """Histogram over number of solos per track

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
    """

    # only consider solo segments
    track_db_filtered = track_db[track_db['segment_class'] == 'solo']
    data = pd.DataFrame({'count': track_db_filtered.groupby(['track_name']).last().segment_class_id})

    # plotting
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.countplot(x='count', data=data, color='b')
    ax.set_xlabel('#Solos')
    ax.set_ylabel('#Tracks')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'hist_solos_per_track.pdf'), bbox_inches='tight')


def hist_solo_choruses_per_instrument(track_db, path_output):
    """Histogram over number of choruses per track

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
    """

    track_db_filtered = track_db[track_db['segment_class'] == 'solo']

    # plotting
    fig, ax = plt.subplots(figsize=(15, 4))
    sns.countplot(x='instrument_solo', data=track_db_filtered, color='b',
                  order=track_db_filtered['instrument_solo'].value_counts().index)
    plt.xticks(rotation=90)
    ax.set_xlabel('Solo Instrument')
    ax.set_ylabel('#Choruses')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'hist_solo_choruses_per_instrument.pdf'), bbox_inches='tight')


def plot_hist_solos_per_instr(track_db, path_output):

    track_db_mod = track_db
    # create empty index list
    idx_list = []

    # find index of not needed rows
    for cur_index, cur_annotation in track_db_mod.iterrows():
        if '01' not in cur_annotation['label'][-2:]:
            idx_list = idx_list + [cur_index]
        if 'ts1' in str(cur_annotation['instrument']) or 'ts2' in str(cur_annotation['instrument']):
            idx_list = idx_list + [cur_index]

    # drop the rows which are not needed
    track_db_mod = track_db_mod.drop(track_db_mod.index[idx_list])
    # print(track_db)
    data = pd.DataFrame({'count': track_db_mod['instrument']})

    data = data['count'].value_counts()
    # data = data.groupby(['count']).size().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    data.plot(kind='bar', color='b', rot=0)
    #              order=['cl', 'bcl', 'ss', 'as', 'ts', 'ts-c', 'bs', 'tp', 'cor', 'tb', 'g', 'p', 'vib', 'dr'])
    ax.set_xlabel('Instruments')
    ax.set_ylabel('#Solos')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'hist_solos_per_instr.pdf'), bbox_inches='tight')


def plot_length_solos_per_instrument(track_db, path_output):

    track_db_mod = track_db
    grps = track_db_mod.groupby(['instrument'])
    # data = pd.DataFrame(None, columns=column_names)
    names = []
    durations = []
    for name, group in grps:
        duration = 0
        if name == 'ss':
            pd.set_option('display.max_rows', 1000)
        for cur_index, cur_grp in group.iterrows():
            duration = duration + cur_grp['region_end'] - cur_grp['region_start']
        # add dataframe of one track to dataframe of all songs
        names = names + [name]
        durations = durations + [duration/60]

    data = pd.DataFrame({'names': names, 'durations': durations})
    data = data.sort_values('durations', ascending=False)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax = data.plot(kind='bar', x='names', y='durations', color='b', rot=0, legend=False)
    # order=['cl', 'bcl', 'ss', 'as', 'ts', 'ts-c', 'bs', 'tp', 'cor', 'tb', 'g', 'p', 'vib', 'dr'])
    ax.set_xlabel('Instruments')
    ax.set_ylabel('Length of Solos (min)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'length_solos_per_instrument'), bbox_inches='tight')


def plot_average_length_solos_per_instrument(track_db, path_output):

    track_db_mod = track_db
    # extract the first string between _ and _
    track_db_mod['label'] = track_db_mod['label'].str.extract('(?<=\_)(.*?)(?=\_)')

    grps = track_db_mod.groupby(['instrument'])

    names = []
    durations_per_instrument = []

    for name, group in grps:

        durations = []
        solo_per_song = group.groupby(['trackname'])

        for name_grp, track_group in solo_per_song:

            solos_per_song = track_group.groupby(['label'])

            for name_grp_solo, solo_group in solos_per_song:
                duration = 0
                for cur_index, cur_grp in solo_group.iterrows():

                    duration = duration + cur_grp['region_end'] - cur_grp['region_start']

                durations = durations + [duration]
        # add dataframe of one track to dataframe of all songs
        names = names + [name]
        durations_per_instrument = durations_per_instrument + [durations]

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=durations_per_instrument, color='b')
    plt.xticks(plt.xticks()[0], names)
    ax.set_ylim(0, 350)
    ax.set_xlabel('Instruments')
    ax.set_ylabel('Average Length of Solos (s)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'average_length_solos_per_instrument'), bbox_inches='tight')


def plot_average_nr_solos_per_instrument(track_db, path_output, nr_tracks):

    track_db_mod = track_db
    # extract the first string between _ and _
    track_db_mod['label'] = track_db_mod['label'].str.extract('(?<=\_)(.*?)(?=\_)')

    grps = track_db_mod.groupby(['instrument'])
    # data = pd.DataFrame(None, columns=column_names)
    names = []
    numbers_per_instrument = []
    for name, group in grps:

        number = 0
        solo_per_song = group.groupby(['trackname'])

        for name_grp, track_group in solo_per_song:

            number = number + 1

        # add dataframe of one track to dataframe of all songs
        names = names + [name]
        numbers_per_instrument = numbers_per_instrument + [number/nr_tracks]
    numbers_per_instrument = [float(i*100) for i in numbers_per_instrument]
    data = pd.DataFrame({'names': names, 'durations': numbers_per_instrument})
    data = data.sort_values('durations', ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax = data.plot(kind='bar', x='names', y='durations', color='b', rot=0, legend=False)
    # plt.xticks(plt.xticks()[0], names)
    ax.set_xlabel('Instruments')
    ax.set_ylabel('Percentage of Tracks with Solo (%)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'Percentage_solos_per_instrument'),
                bbox_inches='tight')


if __name__ == '__main__':

    # setting global variables
    PATH_OUTPUT = 'figures_statistics'
    PATH_DATA = 'data'
    path_annotations = os.path.join(PATH_DATA, 'annotations_csv')
    path_annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))

    os.makedirs(PATH_OUTPUT, exist_ok=True)

    jsd_track_db = load_jsd(path_annotation_files)
    nr_tracks = len(jsd_track_db['track_name'].unique())

    print('Number of segments per class:')
    print(jsd_track_db.groupby('segment_class').describe())

    plot_hist_segments_per_track(jsd_track_db, PATH_OUTPUT)
    plot_hist_choruses_per_track(jsd_track_db, PATH_OUTPUT)
    plot_hist_solos_per_track(jsd_track_db, PATH_OUTPUT)
    hist_solo_choruses_per_instrument(jsd_track_db, PATH_OUTPUT)

    # plot_hist_solos_per_instr(jsd_track_db, PATH_OUTPUT)
    # plot_length_solos_per_instrument(track_db, PATH_OUTPUT)
    # plot_average_length_solos_per_instrument(track_db, PATH_OUTPUT)
    # plot_average_nr_solos_per_instrument(track_db, PATH_OUTPUT, nr_tracks)
