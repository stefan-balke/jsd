"""
    This script plots various statistics of the annotations from the dataset.
"""

import os
import glob
import seaborn as sns
import pandas as pd
from utils import create_track_db
sns.set(style='white')
sns.set_context('paper', font_scale=1.8, rc={'lines.linewidth': 0.75})


def plot_nr_soloparts(track_db, path_data, path_output):

    data = pd.DataFrame({'count': track_db.groupby(['trackname']).size()}).reset_index()
    fig, ax = sns.plt.subplots(figsize=(7, 4))
    sns.set_color_codes("muted")
    sns.countplot(x='count', data=data, color='b')
    ax.set_xlabel('Number of Solo Choruses in Track')
    ax.set_ylabel('Number of Occurances')
    ax.tick_params(labelsize=11)
    sns.despine()
    sns.plt.tight_layout()
    sns.plt.savefig(os.path.join(path_data, path_output, 'nr_soloparts'), bbox_inches='tight', format='pdf')


def plot_nr_solos(track_db, path_data, path_output):

    grps = track_db.groupby(['trackname']).last()
    # extract the first string between _ and _
    solo_nr = grps['label'].str.extract('\_(.*)\_')
    solo_nr = pd.to_numeric(solo_nr)
    data = pd.DataFrame({'count': solo_nr})
    fig, ax = sns.plt.subplots(figsize=(7, 4))
    sns.set_color_codes("muted")
    sns.countplot(x='count', data=data, color='b')
    ax.set_xlabel('Number of Solos in Track')
    ax.set_ylabel('Number of Occurances')
    ax.tick_params(labelsize=11)
    sns.despine()
    sns.plt.tight_layout()
    sns.plt.savefig(os.path.join(path_data, path_output, 'nr_solos'), bbox_inches='tight', format='pdf')


def plot_nr_solos_per_instrument(track_db, path_data, path_output):

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
    fig, ax = sns.plt.subplots(figsize=(7, 4))
    sns.set_color_codes("muted")
    data.plot(kind='bar', color='b', rot=0)
    #              order=['cl', 'bcl', 'ss', 'as', 'ts', 'ts-c', 'bs', 'tp', 'cor', 'tb', 'g', 'p', 'vib', 'dr'])
    ax.set_xlabel('Instruments')
    ax.set_ylabel('Number of Solos')
    ax.tick_params(labelsize=11)
    sns.despine()
    sns.plt.tight_layout()
    sns.plt.savefig(os.path.join(path_data, path_output, 'nr_solos_per_instrument'), bbox_inches='tight', format='pdf')


def plot_length_solos_per_instrument(track_db, path_data, path_output):

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
    fig, ax = sns.plt.subplots(figsize=(20, 20))
    sns.set_color_codes("muted")
    ax = data.plot(kind='bar', x='names', y='durations', color='b', rot=0, legend=False)
    # order=['cl', 'bcl', 'ss', 'as', 'ts', 'ts-c', 'bs', 'tp', 'cor', 'tb', 'g', 'p', 'vib', 'dr'])
    ax.set_xlabel('Instruments')
    ax.set_ylabel('Length of Solos (min)')
    ax.tick_params(labelsize=11)
    sns.despine()
    sns.plt.tight_layout()
    sns.plt.savefig(os.path.join(path_data, path_output, 'length_solos_per_instrument'),
                    bbox_inches='tight', format='pdf')


def plot_average_length_solos_per_instrument(track_db, path_data, path_output):

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

    fig, ax = sns.plt.subplots(figsize=(7, 4))
    sns.set_color_codes("muted")
    sns.boxplot(data=durations_per_instrument, color='b')
    sns.plt.xticks(sns.plt.xticks()[0], names)
    ax.set_ylim(0, 350)
    ax.set_xlabel('Instruments')
    ax.set_ylabel('Average Length of Solos (s)')
    ax.tick_params(labelsize=11)
    sns.despine()
    sns.plt.tight_layout()
    sns.plt.savefig(os.path.join(path_data, path_output, 'average_length_solos_per_instrument'),
                    bbox_inches='tight', format='pdf')


def plot_average_nr_solos_per_instrument(track_db, path_data, path_output, nr_tracks):

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
    fig, ax = sns.plt.subplots(figsize=(7, 4))
    sns.set_color_codes("muted")
    ax = data.plot(kind='bar', x='names', y='durations', color='b', rot=0, legend=False)
    # sns.plt.xticks(sns.plt.xticks()[0], names)
    ax.set_xlabel('Instruments')
    ax.set_ylabel('Percentage of Tracks with Solo (%)')
    ax.tick_params(labelsize=11)
    # sns.plt.show()
    sns.despine()
    sns.plt.tight_layout()
    sns.plt.savefig(os.path.join(path_data, path_output, 'Percentage_solos_per_instrument'),
                    bbox_inches='tight', format='pdf')


if __name__ == '__main__':

    # setting global variables
    path_output = 'general_statistics'
    path_data = 'data'
    path_annotations = os.path.join(path_data, 'annotation_csv')
    annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))

    if not os.path.isdir(os.path.join(path_data, path_output)):
        os.mkdir(os.path.join(path_data, path_output))

    track_db = create_track_db(annotation_files)
    nr_tracks = len(track_db['trackname'].unique())

    plot_nr_soloparts(track_db, path_data, path_output)
    plot_nr_solos(track_db, path_data, path_output)
    plot_nr_solos_per_instrument(track_db, path_data, path_output)
    plot_length_solos_per_instrument(track_db, path_data, path_output)
    plot_average_length_solos_per_instrument(track_db, path_data, path_output)
    plot_average_nr_solos_per_instrument(track_db, path_data, path_output, nr_tracks)
