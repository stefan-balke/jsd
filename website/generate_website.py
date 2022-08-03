"""
    This script generates websites for each track in the database, containing an audioplayer
    for each track and the annotation data visualized together with SSMs and NCs for different
    parameter settings.
"""

import os
import sys
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import display
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape

# hacky relative import
sys.path.append(os.path.join('..'))
import jsd_utils


def create_index_page(env, tracks, path_output):
    """Creates a simple index page, which lists the links to all track pages.

    Parameters
    ----------
    env : Environment
        The jinja-environment to be used. env must already be set up with
        the folder containing 'index_template.html'
    tracks : pd.DataFrame
        list of all track names combined with the track links
    path_output : str
        Path to the directory where the resulting html ('index.html') is stored

    """
    # load template
    template = env.get_template('index.html')

    # render template
    template_html = template.render(title='Overview', tracks=tracks)

    # save website
    with open(os.path.join(path_output, 'index.html'), encoding='utf-8', mode='w') as f:
        f.write(template_html)


def create_track_page(track_name, annotations, results_foote_short, path_results_cnn, path_output, template_env):
    """Creates a webpage from the given track, using a template called 'track.html',
       whose position is specified by the given Environment.

    Parameters
    ----------
    env : Environment
        The jinja-environment to be used. env must already be set up with the folder containing 'opera_template.html'
    path_anno: str
        Path to the csv annotation file
    path_features: str
        Path to the npz audiofeature file
    params_cens: list
        List containing the parameter tuples (smoothing_factor, downsampling_factor)
    path_output: str
        Path to the output directory
    folder_name_plots: str
        Name of the directory where the plots are stored
    path_instrument_images: str
        Path to the instrument images directory
    feature_rate: int
        Rate of features per second
    kernel_size: int
        Kernel size of the gaussian kernel for footy

    """

    # load template
    template = template_env.get_template('track.html')

    # create empty lists which will be appended in the loop
    path_ssms = []
    title_ssms = []

    ##### Foote, short
    cur_results_track_names = [x['track_name'] for x in results_foote_short]

    cur_result_idx = cur_results_track_names.index(cur_track_name)
    nc_foote_short = results_foote_short[cur_result_idx]['nc']
    ssm_foote_short = results_foote_short[cur_result_idx]['ssm']
    params_foote_short = results_foote_short[cur_result_idx]['wl_ds']
    boundaries_foote_short = results_foote_short[cur_result_idx]['boundaries']
    # boundaries_foote_short = boundaries_foote_short * (results_foote_short[cur_result_idx]['feature_rate'] / params_foote_short[1])

    cur_title = 'Foote, short (MFCC, l=%s, d=%s)' % (params_foote_short[0], params_foote_short[1])
    fig_ssm_f_mfcc, (ax_ssm, ax_anno, ax_nc) = display.ssmshow_annotations(ssm_foote_short,
                                                                           annotations,
                                                                           nc_foote_short,
                                                                           boundaries_foote_short,
                                                                           title=cur_title)

    fn_ssm_f_mfcc = '{}_foote_short_ssm.png'.format(track_name)
    fig_ssm_f_mfcc.savefig(os.path.join(path_output, 'img', fn_ssm_f_mfcc), bbox_inches='tight')
    plt.close()

    # append the path_ssms and title_ssms lists
    path_ssms.append(os.path.join('img', fn_ssm_f_mfcc))
    title_ssms.append(cur_title)

    # render template
    path_track = os.path.join('../', 'data', 'audio_wjd_mp3', track_name + '.mp3')
    template_html = template.render(title=track_name, path_ssms=path_ssms, title_ssms=title_ssms,
                                    path_track=path_track)

    # save website
    with open(os.path.join(path_output, track_name + '.html'), encoding='utf-8', mode='w') as f:
        f.write(template_html)


if __name__ == '__main__':

    # setting global variables
    PATH_OUTPUT = 'output_website'
    PATH_DATA = '../data'
    PATH_OUTPUT_IMAGES = os.path.join(PATH_OUTPUT, 'img')
    path_annotations = os.path.join(PATH_DATA, 'annotations_csv')
    path_annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))
    path_eval = os.path.join('..', 'baselines', 'data')
    path_results_foote = os.path.join(path_eval, 'foote_evaluation')
    path_results_cnn = os.path.join(path_eval, 'cnn_evaluation')

    os.makedirs(PATH_OUTPUT, exist_ok=True)
    os.makedirs(PATH_OUTPUT_IMAGES, exist_ok=True)

    ASSETS_PATH = PATH_OUTPUT + '/assets'
    if not os.path.exists(ASSETS_PATH):
        shutil.copytree('assets', ASSETS_PATH)

    # load the JSD
    jsd_track_db = jsd_utils.load_jsd(path_annotation_files)
    tracks = jsd_track_db.groupby('track_name').size().to_frame('n_segments')
    track_links = '/tracks/' + tracks.index + '.html'
    tracks['tracks_names'] = tracks.index
    tracks['duration'] = jsd_track_db.groupby('track_name').agg({'segment_end': np.max})

    track_instruments = jsd_track_db.groupby('track_name')['instrument'].apply(list).dropna()

    instruments_lists = []

    for cur_track in track_instruments:
        cur_track_wo_nans = [x for x in cur_track if isinstance(x, str)]
        instruments_list = ','.join(cur_track_wo_nans)
        instruments_list = instruments_list.replace('s_', '')
        instruments_list = instruments_list.replace('b_', '')
        instruments_list = instruments_list.split(',')
        instruments_list = pd.Series(instruments_list).drop_duplicates().tolist()

        instruments_lists.append(instruments_list)

    tracks['instruments_list'] = instruments_lists

    # setting up jinja environment
    PATH = os.path.dirname(os.path.abspath(__file__))
    template_env = Environment(loader=FileSystemLoader(os.path.join(PATH, 'templates')),
                               autoescape=select_autoescape(['html', 'xml']))

    # overview page
    create_index_page(template_env, tracks, PATH_OUTPUT)

    # track pages
    path_results_foote_short = os.path.join(path_results_foote, 'ncs_wl-(9, 4)_kernelsize-40.npz')
    results_foote_short = np.load(path_results_foote_short, allow_pickle=True)['nc_outputs']

    for cur_track_name, _ in tracks.iterrows():
        cur_annotations = jsd_track_db[jsd_track_db['track_name'] == cur_track_name]

        try:
            create_track_page(cur_track_name, cur_annotations, results_foote_short,
                              path_results_cnn, PATH_OUTPUT, template_env)
        except Exception as e:
            print('{} could not be generated!'.format(cur_track_name))
            print(e)

    """

    # create a webpage for each track by iterating over all tracks
    for cur_path_anno in tqdm.tqdm(annotation_files):
        cur_path_features = os.path.splitext(os.path.basename(cur_path_anno))[0]
        cur_path_features = os.path.join(path_features, cur_path_features + '.npz')
        # create_track_page(template_env, cur_path_anno, cur_path_features, params_cens,
        #                   path_output, folder_name_plots, path_instrument_images, feature_rate,
        #                   kernel_size)
        try:
            create_track_page(template_env, cur_path_anno, cur_path_features, params_cens,
                              path_output, folder_name_plots, path_instrument_images,
                              feature_rate, kernel_size)
        except Exception as e:
            print(os.path.splitext(os.path.basename(cur_path_anno))[0] + ' could not be generated!')
            print(e)

    # create a index page, which only contains all the sccessful created trackpages
    html_files = glob.glob(os.path.join(path_output, '*.html'))
    track_names = [os.path.splitext(os.path.basename(cur_file))[0] for cur_file in html_files]
    track_links = [cur_track + '.html' for cur_track in track_names]
    """