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


def create_track_page(env, path_anno, path_features, params_cens,
                      path_output, folder_name_plots, path_instrument_images, feature_rate, kernel_size):
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

    # get trackname from path_anno
    cur_track_name = os.path.splitext(os.path.basename(path_anno))[0]
    # print trackname for debugging purposes
    # print(cur_track_name)

    # load template
    template = env.get_template('track.html')

    # read annotation file
    annotations = pd.read_csv(path_anno, usecols=[1, 2, 3, 4], sep=';')

    # read features
    features = np.load(path_features)

    # create empty lists which will be appended in the loop
    path_ssms = []
    title_ssms = []

    # iterate over the cens_parameter list to create a plot for each setting
    for cur_params_cens in params_cens:

        # Analysis of the feature file with the current parameter settings
        # calculates SSMS, NCs and peaks of NCs for CENS and MFCC
        (ssm_f_cens, ssm_f_mfcc, nc_cens, nc_mfcc, peaks_cens, peaks_mfcc) = structure_analysis.analysis(features,
                                                                                                         cur_params_cens,
                                                                                                         kernel_size)

        # Plotting for CENS and MFCC

        # CENS: Display SSM, NC with peaks and visualized annotation
        cur_title = 'SSM (CENS, l=%s, d=%s)' % (cur_params_cens[0], cur_params_cens[1])
        fig_ssm_f_cens, (ax_ssm, ax_anno, ax_nc) = display.ssmshow_annotations(ssm_f_cens, annotations,
                                                                               nc_cens, peaks_cens,
                                                                               path_instrument_images, title=cur_title)

        # CENS: Save the plot
        fn_ssm_f_cens = '%s_cens_l_%d_d_%d.png' % (cur_track_name, cur_params_cens[0], cur_params_cens[1])
        fig_ssm_f_cens.savefig(os.path.join(path_output, folder_name_plots, fn_ssm_f_cens),
                               bbox_inches='tight')
        plt.close()

        # append the path_ssms and title_ssms lists
        path_ssms.append(os.path.join(folder_name_plots, fn_ssm_f_cens))
        title_ssms.append(cur_title)

        # MFCC: Display SSM, NC with peaks and visualized annotation
        cur_title = 'SSM (MFCC, l=%s, d=%s)' % (cur_params_cens[0], cur_params_cens[1])
        fig_ssm_f_mfcc, (ax_ssm, ax_anno, ax_nc) = display.ssmshow_annotations(ssm_f_mfcc, annotations,
                                                                               nc_mfcc, peaks_mfcc,
                                                                               path_instrument_images, title=cur_title)
        plt.close()
        # MFCC: Save the plot
        fn_ssm_f_mfcc = '%s_mfcc_l_%d_d_%d.png' % (cur_track_name, cur_params_cens[0], cur_params_cens[1])
        fig_ssm_f_mfcc.savefig(os.path.join(path_output, folder_name_plots, fn_ssm_f_mfcc),
                               bbox_inches='tight')
        plt.close()

        # append the path_ssms and title_ssms lists
        path_ssms.append(os.path.join(folder_name_plots, fn_ssm_f_mfcc))
        title_ssms.append(cur_title)

    # render template
    template_html = template.render(title=cur_track_name, path_ssms=path_ssms, title_ssms=title_ssms,
                                    path_track=os.path.join('../', path_data, 'audio_wjd_mp3',
                                                            cur_track_name + '.mp3'))

    # save website
    with open(os.path.join(path_output, cur_track_name + '.html'), encoding='utf-8', mode='w') as f:
        f.write(template_html)


if __name__ == '__main__':

    # setting global variables
    PATH_OUTPUT = 'output_website'
    PATH_DATA = '../data'
    path_annotations = os.path.join(PATH_DATA, 'annotations_csv')
    path_annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))

    os.makedirs(PATH_OUTPUT, exist_ok=True)

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
        instruments_list = ','.join(instruments_list)

        instruments_lists.append(instruments_list)

    tracks['instruments_list'] = instruments_lists

    # setting up jinja environment
    PATH = os.path.dirname(os.path.abspath(__file__))
    template_env = Environment(loader=FileSystemLoader(os.path.join(PATH, 'templates')),
                               autoescape=select_autoescape(['html', 'xml']))

    create_index_page(template_env, tracks, PATH_OUTPUT)
    # breakpoint()
    """
    # setting global variables
    path_output = 'output'
    path_data = 'data'
    path_tracks = os.path.join(path_data, 'audio_wjd_mp3')
    path_features = os.path.join(path_data, 'features_wjd')
    path_annotations = os.path.join(path_data, 'anno_wjd')
    path_instrument_images = os.path.join(path_data, 'instrument_images')
    annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))
    folder_name_plots = 'plots_structure'
    kernel_size = 50
    feature_rate = 10
    params_cens = [(9, 2), (11, 5), (21, 5), (41, 10), (81, 10)]


    # make sure the folders exist before trying to save things
    if not os.path.isdir(path_output):
        os.mkdir(path_output)
    if not os.path.isdir(os.path.join(path_output, folder_name_plots)):
        os.mkdir(os.path.join(path_output, folder_name_plots))

    # render templates
    # create_index_page(template_env, zip(track_names, track_links), path_output)

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