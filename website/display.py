"""
    This script contains all the plotting functions for visualizing the dataset.
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import scipy
import scipy.misc
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np


def colorbar_ct():
    # exponential colormap5 copied from matlab toolbox
    x = 50
    y = (np.power(x, np.arange(0.0, 1.0, 0.01)) - 1) / (x - 1)
    colors_ct = 1 - np.array([y, y, y]).transpose()
    cmap_ct = matplotlib.colors.ListedColormap(colors_ct, 'cmap_ct')

    return cmap_ct


def ssmshow(ssm, axes='', cmap='ct'):
    if cmap == 'ct':
        cmap = colorbar_ct()

    axes.imshow(ssm, interpolation='nearest', aspect='auto',
                origin='lower', cmap=cmap)

    return axes


def ncshow(nc, peaks, axes=''):

    axes.plot(nc)
    axes.set_xlim([0, len(nc)])
    # axes.set_title('Chroma Novelty Curve')
    [axes.axvline(peak, color='r') for peak in peaks]

    return axes


def solo_colormap(instrument, solo_nr):

    solo_instrument_list = ['s_tp', 's_as', 's_key', 's_b', 's_vib', 's_voc',
                            's_dr', 's_g', 's_fln', 's_cor', 's_bs', 's_cl',
                            's_ts', 's_p', 's_ss', 's_tb', 's_bcl', 's_fl', 's_bjo', 's_vc', 's_all']
    nr_solo_instruments = len(solo_instrument_list)
    colornr = solo_instrument_list.index(instrument)
    colors = cm.rainbow(np.linspace(0, 1, nr_solo_instruments))
    color = colors[colornr]
    return color


def plot_annotations(cur_annotation, path_instrument_images, axes=None):
    # loop over rows in pandas DataFrame and add rectangles
    # solos start at color 5

    #  SPAGETHI CODE DO NOT TOUCH !!!
    cur_annotation = cur_annotation.replace(to_replace='s_ts1', value='s_ts', regex=True)
    cur_annotation = cur_annotation.replace(to_replace='s_ts2', value='s_ts', regex=True)
    cur_annotation = cur_annotation.replace(to_replace='s_tp1', value='s_tp', regex=True)
    cur_annotation = cur_annotation.replace(to_replace='s_tp2', value='s_tp', regex=True)
    cur_annotation = cur_annotation.replace(to_replace='s_as1', value='s_as', regex=True)
    cur_annotation = cur_annotation.replace(to_replace='s_as2', value='s_as', regex=True)
    solocounter = 5

    # parsed solo instruments from all waves
    for cur_index, cur_annotation in cur_annotation.iterrows():
        cur_annotation_label_list = '' if isinstance(cur_annotation[3], float) else cur_annotation[3]
        cur_annotation_label_list = cur_annotation_label_list.split(',')
        cur_theme = cur_annotation[2]
        soloinstruments = [cur_instr for cur_instr in cur_annotation_label_list if "s_" in cur_instr]
        if cur_theme.startswith('solo'):
            cur_theme_nr = solocounter
            solocounter = solocounter + 1
            nr_instruments = len(cur_annotation_label_list) - 1
            if len(soloinstruments) > 1:
                colors = solo_colormap('s_all', cur_theme[-2:])
            else:
                colors = solo_colormap(cur_annotation_label_list[0], cur_theme[-2:])
        elif cur_theme == 'intro':
            cur_theme_nr = 1
            nr_instruments = len(cur_annotation_label_list)
            colors = [190, 190, 190]
            colors = np.divide(colors, 255)
        elif cur_theme == 'outro':
            cur_theme_nr = 2
            nr_instruments = len(cur_annotation_label_list)
            colors = [190, 190, 190]
            colors = np.divide(colors, 255)
        elif cur_theme.startswith('theme'):
            cur_theme_nr = 3
            nr_instruments = len(cur_annotation_label_list)
            colors = [105, 105, 105]
            colors = np.divide(colors, 255)
        elif cur_theme.startswith('silence'):
            cur_theme_nr = 4
            nr_instruments = 0
            colors = [255, 255, 255]
            colors = np.divide(colors, 255)

        axes.add_patch(
            patches.Rectangle(
                (cur_annotation.region_start, 0),   # (x,y)
                cur_annotation.region_end-cur_annotation.region_start,       # width
                1, facecolor=colors,
                edgecolor='black'
            )
        )

        # list with all solo instruments
        row_nr = 2
        seg_height = 0.33
        box_mid = cur_annotation.region_start + (cur_annotation.region_end-cur_annotation.region_start)/2
        x_pos_even = cur_annotation.region_start + (cur_annotation.region_end-cur_annotation.region_start)/4
        x_pos_uneven = cur_annotation.region_start + 3*(cur_annotation.region_end-cur_annotation.region_start)/4
        soloinstruments = [0]

        y_pos_stat = (1-seg_height)/(np.ceil(nr_instruments/row_nr)+1)
        # load instrument pictograms
        # print(int(0.9*(cur_annotation.region_end-cur_annotation.region_start)))
        size_image = 30
        if int(cur_annotation.region_end-cur_annotation.region_start) < 30:
            size_image = int(0.9*(cur_annotation.region_end-cur_annotation.region_start))
        size_image_y = size_image
        if len(cur_annotation_label_list) > 0 and cur_theme_nr != 4:
            if cur_theme_nr == 3:
                img = scipy.misc.imread(os.path.join(path_instrument_images, 'letter-t.png'), mode='RGBA')
                size_image = 25
                size_image_y = int(1.1*size_image)
            elif cur_theme_nr == 1:
                img = scipy.misc.imread(os.path.join(path_instrument_images, 'letter-i.png'), mode='RGBA')
                size_image = 5
                size_image_y = 30
            elif cur_theme_nr == 2:
                size_image = 25
                size_image_y = size_image
                img = scipy.misc.imread(os.path.join(path_instrument_images, 'letter-o.png'), mode='RGBA')
            else:
                soloinstruments = [cur_instr for cur_instr in cur_annotation_label_list if "s_" in cur_instr]
                img = scipy.misc.imread(os.path.join(path_instrument_images, cur_annotation_label_list[0][2:] + '.png'))
                if len(soloinstruments) > 1:
                    for ii in range(len(soloinstruments)):
                        seg_height = 0.5
                        y_pos_stat2 = (1-seg_height)/(np.ceil(len(soloinstruments)/row_nr)+1)
                        img = scipy.misc.imread(os.path.join(path_instrument_images, cur_annotation_label_list[ii][2:] + '.png'),
                                                 mode='RGBA')
                        size_image = int(0.8*(cur_annotation.region_end-cur_annotation.region_start)/row_nr)
                        size_image_y = size_image
                        if size_image < 1:
                            size_image = 1
                            size_image_y = 1
                        img = scipy.misc.imresize(img, (size_image, size_image))
                        imagebox = OffsetImage(img)
                        x_pos = x_pos_uneven if np.mod((ii+1), 2) == 0 else x_pos_even
                        y_pos = ((1)-y_pos_stat2*np.ceil((ii+1)/row_nr))
                        # print(y_pos)
                        xy = [x_pos, y_pos]               # coordinates to position this image
                        ab = AnnotationBbox(imagebox, xy,
                                            xybox=(0., 0.),
                                            xycoords='data',
                                            boxcoords="offset points",
                                            bboxprops=dict(facecolor='none', boxstyle='round', color='none'))
                        axes.add_artist(ab)

            # old value:70
            # if int(cur_annotation.region_end-cur_annotation.region_start) < 100:
            #     # size_image = int(1.2*(cur_annotation.region_end-cur_annotation.region_start))
            #     size_image = int(1*(cur_annotation.region_end-cur_annotation.region_start))
            # print(len(soloinstruments))
            if size_image < 1:
                size_image = 1
                size_image_y = 1
            # print(size_image)
            if len(soloinstruments) == 1:
                #
                # if cur_theme_nr == 1:
                #     size_image = int(0.9*(cur_annotation.region_end-cur_annotation.region_start))

                img = scipy.misc.imresize(img, (size_image_y, size_image))
                imagebox = OffsetImage(img)
                xy = [box_mid, 1-(seg_height/2)]               # coordinates to position this image

                ab = AnnotationBbox(imagebox, xy,
                                    xybox=(0., 0.),
                                    xycoords='data',
                                    boxcoords="offset points",
                                    bboxprops=dict(facecolor='none', boxstyle='round', color='none'))
                axes.add_artist(ab)

        for i in range(nr_instruments-len(soloinstruments)+1):

            if cur_annotation_label_list[i] != []:

                if cur_theme_nr < 4:
                    instrument = cur_annotation_label_list[i]
                else:
                    instrument = cur_annotation_label_list[i+1+len(soloinstruments)-1][2:]
                instrument = instrument.replace(' ', '')
                instrument = instrument.replace('_', '')
                instrument = instrument.replace('1', '')
                instrument = instrument.replace('2', '')
                img = scipy.misc.imread(path_instrument_images + '/' + instrument + '.png', mode='RGBA')
                # size_image= 30
                # if int((cur_annotation.region_end-cur_annotation.region_start)/row_nr) < 100/row_nr:
                size_image = int(0.8*(cur_annotation.region_end-cur_annotation.region_start)/row_nr)
                if size_image < 1:
                    size_image = 1
                    size_image_y = 1
                img = scipy.misc.imresize(img, (size_image, size_image))
                imagebox = OffsetImage(img)
                x_pos = x_pos_uneven if np.mod((i+1), 2) == 0 else x_pos_even
                y_pos = (1-seg_height)-y_pos_stat*np.ceil((i+1)/row_nr)

                xy = [x_pos, y_pos]               # coordinates to position this image
                ab = AnnotationBbox(imagebox, xy,
                                    xybox=(0., 0.),
                                    xycoords='data',
                                    boxcoords="offset points",
                                    bboxprops=dict(facecolor='none', boxstyle='round', color='none'))
                axes.add_artist(ab)

    axes.axes.get_yaxis().set_visible(False)
    axes.set_xlim([0, cur_annotation.tail().region_end])

    return axes


def ssmshow_annotations(ssm, annotations, nc, peaks, path_instrument_images, title=None):
    fig = plt.figure(figsize=(15/1.3, 17/1.3))
    gs = gridspec.GridSpec(11, 4)
    gs.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

    ax_ssm = fig.add_subplot(gs[:8, :])
    ax_ssm = ssmshow(ssm, axes=ax_ssm)
    ax_ssm.yaxis.set_ticklabels([])
    ax_ssm.set_title(title)
    ax_ssm.get_xaxis().set_visible(False)

    ax_anno = fig.add_subplot(gs[9:, :])
    ax_anno = plot_annotations(annotations, path_instrument_images, ax_anno)

    ax_nc = fig.add_subplot(gs[8:9, :])
    ax_nc = ncshow(nc, peaks, ax_nc)
    ax_nc.get_xaxis().set_visible(False)
    ax_nc.get_yaxis().set_visible(False)

    return fig, (ax_ssm, ax_anno, ax_nc)


def ssmshow_annotations_report(ssm, annotations, nc, peaks, path_instrument_images, title=None):
    fig = plt.figure(figsize=(15/1.3, 17/1.3))
    gs = gridspec.GridSpec(11, 4)
    gs.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

    ax_ssm = fig.add_subplot(gs[:8, :])
    ax_ssm = ssmshow(ssm, axes=ax_ssm)
    ax_ssm.yaxis.set_ticklabels([])
    ax_ssm.get_xaxis().set_visible(False)

    ax_anno = fig.add_subplot(gs[9:, :])
    ax_anno = plot_annotations(annotations, path_instrument_images, ax_anno)
    plt.xlabel('Time (seconds)', fontsize=12)

    ax_nc = fig.add_subplot(gs[8:9, :])
    ax_nc = ncshow(nc, peaks, ax_nc)
    ax_nc.get_xaxis().set_visible(False)
    ax_nc.get_yaxis().set_visible(False)

    return fig, (ax_ssm, ax_anno, ax_nc)
