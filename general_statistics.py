"""
    This script plots various statistics of the annotations from the dataset.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import jsd_utils
sns.set(style='white')
sns.set_context('paper', font_scale=1.6, rc={'lines.linewidth': 0.75})
sns.set_color_codes('muted')


def plot_hist_segments_per_track(track_db, path_output, count_threshold=20):
    """Histogram over number of all segments per track

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
        count_threshold : int
            Larger counts will be treated as `other`
    """

    # consider all but silence
    track_db_filtered = track_db.copy(deep=True)
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] != 'silence']

    # pool mixed solos
    track_db_filtered.loc[track_db_filtered['mixed_solo'] > 0, 'instrument_solo'] = 'mix'

    data = pd.DataFrame({'count': track_db_filtered.groupby(['track_name']).size()}).reset_index()
    data_count = data['count'].values
    data_count[data_count > count_threshold] = count_threshold + 5

    # plotting
    _, ax = plt.subplots(figsize=(9, 4))
    sns.distplot(data_count, bins='auto', kde=False, norm_hist=False,
                 hist_kws={'align': 'right', 'alpha': 1.0})
    labels = ax.get_xticks().tolist()
    labels = [int(item) for item in labels]
    labels[-2] = '>{}'.format(count_threshold)
    ax.set_xticklabels(labels)
    ax.set_xlabel('#Segments')
    ax.set_ylabel('#Tracks')

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'hist_segments_per_track.pdf'), bbox_inches='tight')


def plot_hist_choruses_per_track(track_db, path_output, count_threshold=20):
    """Histogram over number of choruses per track

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
        count_threshold : int
            Larger counts will be treated as `other`
    """

    # filter out silence segments
    track_db_filtered = track_db.copy(deep=True)
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] != 'silence']
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] != 'intro']
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] != 'outro']

    # pool mixed solos
    track_db_filtered.loc[track_db_filtered['mixed_solo'] > 0, 'instrument_solo'] = 'mix'

    data = pd.DataFrame({'count': track_db_filtered.groupby(['track_name']).size()}).reset_index()
    data_count = data['count'].values
    data_count[data_count > count_threshold] = count_threshold + 5

    # plotting
    _, ax = plt.subplots(figsize=(9, 4))
    sns.distplot(data_count, bins='auto', kde=False, norm_hist=False,
                 hist_kws={'align': 'right', 'alpha': 1.0})
    labels = ax.get_xticks().tolist()
    labels = [int(item) for item in labels]
    labels[-2] = '>{}'.format(count_threshold)
    ax.set_xticklabels(labels)
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

    track_db_filtered = track_db.copy(deep=True)
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] == 'solo']  # only consider solos

    # pool mixed solos
    track_db_filtered.loc[track_db_filtered['mixed_solo'] > 0, 'instrument_solo'] = 'mix'

    data = pd.DataFrame({'count': track_db_filtered.groupby(['track_name']).last().segment_class_id})
    data = pd.value_counts(data['count']).sort_index()
    n_bars = len(data)

    # plotting
    _, ax = plt.subplots(figsize=(9, 4))
    ind = np.arange(n_bars)
    plt.bar(ind, data.values)
    plt.xticks(ind, data.index.astype(int).values.tolist())
    ax.set_xlabel('#Solos')
    ax.set_ylabel('#Tracks')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'hist_solos_per_track.pdf'), bbox_inches='tight')


def plot_hist_solo_choruses_per_instrument(track_db, path_output, count_threshold=5):
    """Histogram over number of choruses per track

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
        count_threshold : int
            Smaller counts will be treated as `other`
    """

    track_db_filtered = track_db.copy(deep=True)
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] == 'solo']  # only consider solos

    # pool mixed solos
    track_db_filtered.loc[track_db_filtered['mixed_solo'] > 0, 'instrument_solo'] = 'mix'

    data = pd.DataFrame({'count': track_db_filtered.groupby(['instrument_solo']).size()}).reset_index()
    data_count = data['count'].values

    # count instruments with less than count_threshold solos to other
    data_count[data_count < count_threshold] = -1
    n_others = np.sum(data_count == -1)
    n_bars = np.sum(data['count'] > 0)
    data = data.sort_values('count', ascending=False)[:n_bars]
    data = data.append({'instrument_solo': 'other', 'count': n_others}, ignore_index=True)
    n_bars = len(data)

    # plotting
    _, ax = plt.subplots(figsize=(9, 4))
    ind = np.arange(n_bars)
    plt.bar(ind, data['count'])
    labels = data['instrument_solo'][:n_bars].to_list()
    plt.xticks(ind, labels, rotation=90)
    ax.set_xlabel('Solo Instrument')
    ax.set_ylabel('#Choruses')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'hist_solo_choruses_per_instrument.pdf'), bbox_inches='tight')


def plot_hist_solos_per_instrument(track_db, path_output, count_threshold=5):
    """Histogram over number of solos per instrument

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
        count_threshold : int
            Smaller counts will be treated as `other`
    """

    track_db_filtered = track_db.copy(deep=True)
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] == 'solo']  # only consider solos
    track_db_filtered = track_db_filtered[track_db['segment_class_id'] == 1]  # count every solo only once

    # pool mixed solos
    track_db_filtered.loc[track_db_filtered['mixed_solo'] > 0, 'instrument_solo'] = 'mix'

    data = pd.DataFrame({'count': track_db_filtered.groupby(['instrument_solo']).size()}).reset_index()
    data_count = data['count'].values

    # count instruments with less than count_threshold solos to other
    data_count[data_count < count_threshold] = -1
    n_others = np.sum(data_count == -1)
    n_bars = np.sum(data['count'] > 0)
    data = data.sort_values('count', ascending=False)[:n_bars]
    data = data.append({'instrument_solo': 'other', 'count': n_others}, ignore_index=True)
    n_bars = len(data)

    # plotting
    _, ax = plt.subplots(figsize=(9, 4))
    ind = np.arange(n_bars)
    plt.bar(ind, data['count'])
    labels = data['instrument_solo'][:n_bars].to_list()
    plt.xticks(ind, labels, rotation=90)
    ax.set_xlabel('Solo Instrument')
    ax.set_ylabel('#Solos')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'hist_solos_per_instrument.pdf'), bbox_inches='tight')


def plot_hist_solodurtotal_per_instrument(track_db, path_output, dur_threshold=600):
    """Histogram of total solo durations per instrument.

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
        dur_threshold : int
            Shorter durs (in seconds) will be treated as `other`
    """

    track_db_filtered = track_db.copy(deep=True)
    track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] == 'solo']  # only consider solos

    # pool mixed solos
    track_db_filtered.loc[track_db_filtered['mixed_solo'] > 0, 'instrument_solo'] = 'mix'

    data = track_db_filtered.groupby(['instrument_solo']).sum().reset_index()
    dur_others = np.sum(data[data['segment_dur'] < dur_threshold]['segment_dur'])
    n_bars = np.sum(data['segment_dur'] >= dur_threshold)
    data = data.sort_values('segment_dur', ascending=False)[:n_bars]
    data = data.append({'instrument_solo': 'other', 'segment_dur': dur_others}, ignore_index=True)
    n_bars = len(data)

    # plotting
    _, ax = plt.subplots(figsize=(9, 4))
    ind = np.arange(n_bars)
    plt.bar(ind, data['segment_dur'] / 60)
    labels = data['instrument_solo'][:n_bars].to_list()
    plt.xticks(ind, labels, rotation=90)
    ax.set_xlabel('Solo Instrument')
    ax.set_ylabel('Solo Duration (min.)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'hist_solodurtotal_per_instrument.pdf'), bbox_inches='tight')


def plot_boxplot_solodur_per_instrument(track_db, path_output):
    """Overview of solo durations per instrument.

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
    """

    track_db_filtered = track_db.copy(deep=True)
    track_db_filtered = jsd_utils.filter_db_by_solo(track_db_filtered)
    data = track_db_filtered

    # exclude 'MilesDavis_BitchesBrew_Orig' track
    data = data.drop(track_db_filtered[track_db_filtered['track_name'] == 'MilesDavis_BitchesBrew_Orig'].index)

    # exclude 'JohnColtrane_Impressions_1961_Orig' track
    data = data.drop(track_db_filtered[track_db_filtered['track_name'] == 'JohnColtrane_Impressions_1961_Orig'].index)

    # pool mixed solos
    data.loc[data['mixed_solo'] > 0, 'instrument_solo'] = 'mix'

    # measure in minutes
    data['segment_dur'] = data['segment_dur'] / 60

    # get medians for box ordering
    medians = data.groupby('instrument_solo')['segment_dur'].median().sort_values(ascending=False)
    box_order = medians.index.to_list()

    # plotting
    _, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(x='instrument_solo', y='segment_dur', data=data, color='b', order=box_order)
    plt.xticks(rotation=60)
    ax.set_xlabel('Solo Instrument')
    ax.set_ylabel('Solo Duration (min.)')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(path_output, 'boxplot_solodur_per_instrument.pdf'), bbox_inches='tight')


def stats_per_segment_class(track_db, path_output):
    """Statistics per Segment type.

    Parameters
    ----------
        track_db : pd.DataFrame
            Pandas DataFrame storing all the annotations.
        path_output : str
            Figure saving path.
    """

    track_db_segments = track_db.groupby('segment_class')
    output = track_db_segments.agg({'segment_dur': [np.sum, np.mean, np.min, np.max]})
    output.to_csv(os.path.join(path_output, 'stats_per_segment_class.csv'))


def stats_per_instrument(track_db, path_output):
    def process(track_db, solo_level=False):
        track_db_filtered = track_db.copy(deep=True)
        # only consider solos
        track_db_filtered = track_db_filtered[track_db_filtered['segment_class'] == 'solo']

        if solo_level:
            # count every solo only once (neglect multiple chorusses)
            track_db_filtered = jsd_utils.filter_db_by_solo(track_db_filtered)

        track_db_filtered_unrolled = []

        # unroll mixed solos
        for _, cur_row in track_db_filtered.iterrows():
            instruments = cur_row['instrument_solo'].split(',')

            for cur_instrument in instruments:
                cur_row_instr = cur_row.copy()
                cur_row_instr['instrument_solo'] = cur_instrument

                track_db_filtered_unrolled.append(cur_row_instr.to_dict())

        track_db_filtered_unrolled = pd.DataFrame(track_db_filtered_unrolled)

        return track_db_filtered_unrolled['instrument_solo'].value_counts()

    counts_chorus = process(track_db, solo_level=False)
    counts_solo = process(track_db, solo_level=True)

    output = jsd_utils.get_instruments()

    output['jsd_n_chorusses'] = 0
    for cur_abbr, cur_row in counts_chorus.to_frame().iterrows():
        if (output['abbr'] == cur_abbr).sum() == 0:
            print('Warning: Instrument "{}" is not specified!'.format(cur_abbr))

        output.loc[output[output['abbr'] == cur_abbr].index,
                'jsd_n_chorusses'] = cur_row['instrument_solo']

    output['jsd_n_solos'] = 0
    for cur_abbr, cur_row in counts_solo.to_frame().iterrows():
        output.loc[output[output['abbr'] == cur_abbr].index,
                'jsd_n_solos'] = cur_row['instrument_solo']

    output['trans_perc'] = output['wjd_n_solos'] / output['jsd_n_solos'] * 100
    # add sum line
    output = output.append(output.sum(numeric_only=True), ignore_index=True)
    # except for trans_perc take the mean
    output.loc[output.tail(1).index.item(),
            'trans_perc'] = output.iloc[:-1]['trans_perc'].mean()

    with open('figures_statistics/overview_table.tex', 'w') as fh:
        fh.write(output.to_latex(index=True,
                                columns=['abbr', 'name', 'jsd_n_solos',
                                        'jsd_n_chorusses', 'wjd_n_solos', 'trans_perc'],
                                float_format="{:0.2f}".format))


if __name__ == '__main__':

    # setting global variables
    PATH_OUTPUT = 'figures_statistics'
    PATH_DATA = 'data'
    path_annotations = os.path.join(PATH_DATA, 'annotations_csv')
    path_annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))

    os.makedirs(PATH_OUTPUT, exist_ok=True)

    jsd_track_db = jsd_utils.load_jsd(path_annotation_files)
    nr_tracks = len(jsd_track_db['track_name'].unique())

    stats_per_segment_class(jsd_track_db, PATH_OUTPUT)
    stats_per_instrument(jsd_track_db, PATH_OUTPUT)
    plot_hist_segments_per_track(jsd_track_db, PATH_OUTPUT)
    plot_hist_choruses_per_track(jsd_track_db, PATH_OUTPUT)
    plot_hist_solos_per_track(jsd_track_db, PATH_OUTPUT)
    plot_hist_solo_choruses_per_instrument(jsd_track_db, PATH_OUTPUT)
    plot_hist_solos_per_instrument(jsd_track_db, PATH_OUTPUT)
    plot_hist_solodurtotal_per_instrument(jsd_track_db, PATH_OUTPUT)
    plot_boxplot_solodur_per_instrument(jsd_track_db, PATH_OUTPUT)
