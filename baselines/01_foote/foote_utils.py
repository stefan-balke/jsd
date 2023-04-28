"""
    This script identifies the boundaries of a given track using the Foote
    method:

    Foote, J. (2000). Automatic Audio Segmentation Using a Measure Of Audio
    Novelty. In Proc. of the IEEE International Conference of Multimedia and
    Expo (pp. 452–455). New York City, NY, USA.
"""

import os
import sys
import numpy as np
import scipy.signal
from scipy.ndimage import filters
import pandas as pd
import librosa
import tqdm
from scipy import signal
import mir_eval

# hacky relative import
sys.path.append(os.path.join('..', '..'))
import jsd_utils


def compute_ssm(X):
    """Computes the self-similarity matrix of X."""
    ssm = np.dot(X.T, X)

    return ssm


def smooth_features(features, win_len_smooth):
    # Apply temporal smoothing
    win = scipy.signal.hanning(win_len_smooth + 2, sym=False)
    win /= np.sum(win)
    win = np.atleast_2d(win)

    features = scipy.signal.convolve2d(features, win, mode='same')

    return features


def compute_kernel_checkerboard_gaussian(L, var=1, normalize=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1]
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L: Parameter specifying the kernel size M=2*L+1
        var: Variance parameter determing the tapering (epsilon)

    Returns:
        kernel: Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2) / (L*var)
    axis = np.arange(-L, L + 1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D

    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))

    return kernel


def compute_novelty_SSM(S, kernel, exclude=False):
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S: SSM
        kernel: Checkerboard kernel (if kernel==None, it will be computed)
        exclude: Sets the first L and last L values of novelty function to zero

    Returns:
        nov: Novelty function
    """

    N = int(S.shape[0])
    M = int(kernel.shape[0])
    L = int((M - 1) / 2)

    nov = np.zeros(N)
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        nov[n] = np.sum(S_padded[n:n+M, n:n+M] * kernel)

    if exclude:
        right = np.min([L, N])
        left = np.max([0, N-L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov


def peak_picking(x, direction=None, abs_thresh=None, rel_thresh=None, descent_thresh=None, tmin=None, tmax=None):
    """Computes the positive peaks of the input vector x
       Python port from the Matlab Roeder_Peak_Picking script "peaks.m"
       reckjn 2017

    Parameters
    ----------
    x : np.array_like
        Signal to be searched for (positive) peaks

    dir : {+1, -1}
        +1 for forward peak searching, -1 for backward peak
        searching. default is dir == -1.

    abs_thresh
        absolute threshold signal, i.e. only peaks
        satisfying x(i)>=abs_thresh(i) will be reported.
        abs_thresh must have the same number of samples as x.
        a sensible choice for this parameter would be a global or local
        average or median of the signal x.
        if omitted, half the median of x will be used.

    rel_thresh
        relative threshold signal. only peak positions i with an
        uninterrupted positive ascent before position i of at least
        rel_thresh(i) and a possibly interrupted (see parameter descent_thresh)
        descent of at least rel_thresh(i) will be reported.
        rel_thresh must have the same number of samples as x.
        a sensible choice would be some measure related to the
        global or local variance of the signal x.
        if omitted, half the standard deviation of x will be used.

    descent_thresh
        descent threshold. during peak candidate verfication, if a slope change
        from negative to positive slope occurs at sample i BEFORE the descent has
        exceeded rel_thresh(i), and if descent_thresh(i) has not been exceeded yet,
        the current peak candidate will be dropped.
        this situation corresponds to a secondary peak
        occuring shortly after the current candidate peak (which might lead
        to a higher peak value)!

        the value descent_thresh(i) must not be larger than rel_thresh(i).

        descent_thresh must have the same number of samples as x.
        a sensible choice would be some measure related to the
        global or local variance of the signal x.
        if omitted, 0.5*rel_thresh will be used.

    tmin : int
        index of start sample. peak search will begin at x(tmin).

    tmax : int
        index of end sample. peak search will end at x(tmax).

    Returns
    -------
    P : np.array_like
        column vector of peak positions
    """

    # set default values
    if direction is None:
        direction = -1
    if abs_thresh is None:
        abs_thresh = np.tile(0.5 * np.median(x), len(x))
    if rel_thresh is None:
        rel_thresh = np.tile(0.5 * np.sqrt(np.var(x)), len(x))
    if descent_thresh is None:
        descent_thresh = 0.5*rel_thresh
    if tmin is None:
        tmin = 1
    if tmax is None:
        tmax = len(x)

    dyold = 0
    dy = 0
    rise = 0  # current amount of ascent during a rising portion of the signal x
    riseold = 0  # accumulated amount of ascent from the last rising portion of x
    descent = 0  # current amount of descent (<0) during a falling portion of the signal x
    searching_peak = True
    candidate = 1
    P = []

    if direction == 1:
        my_range = np.arange(tmin, tmax)
    elif direction == -1:
        my_range = np.arange(tmin, tmax)
        my_range = my_range[::-1]

    # run through x
    for cur_idx in my_range:
        # get local gradient
        dy = x[cur_idx + direction] - x[cur_idx]

        if (dy >= 0):
            rise = rise + dy
        else:
            descent = descent + dy

        if (dyold >= 0):
            if (dy < 0):  # slope change positive->negative
                if (rise >= rel_thresh[cur_idx] and searching_peak is True):
                    candidate = cur_idx
                    searching_peak = False
                riseold = rise
                rise = 0
        else:  # dyold < 0
            if (dy < 0):  # in descent
                if (descent <= -rel_thresh[candidate] and searching_peak is False):
                    if (x[candidate] >= abs_thresh[candidate]):
                        P.append(candidate)  # verified candidate as True peak
                    searching_peak = True
            else:  # dy >= 0 slope change negative->positive
                if searching_peak is False:  # currently verifying a peak
                    if (x[candidate] - x[cur_idx] <= descent_thresh[cur_idx]):
                        rise = riseold + descent  # skip intermediary peak
                    if (descent <= -rel_thresh[candidate]):
                        if x[candidate] >= abs_thresh[candidate]:
                            P.append(candidate)  # verified candidate as True peak
                    searching_peak = True
                descent = 0
        dyold = dy
    return P


def detect_peaks(activations, threshold=0.5, fps=100, include_scores=False, combine=0,
                 pre_avg=12, post_avg=6, pre_max=6, post_max=6):
    """Detects peaks.

    Implements the peak-picking method described in:
    Sebastian Böck, Florian Krebs and Markus Schedl:
    Evaluating the Online Capabilities of Onset Detection Methods.
    Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2012.

    Modified by Jan Schlüter, 2014-04-24

    Parameters
    ----------
    activations : np.nadarray
        vector of activations to process
    threshold : float
        threshold for peak-picking
    fps : float
        frame rate of onset activation function in Hz
    include_scores : boolean
        include activation for each returned peak
    combine :
        only report 1 onset for N seconds
    pre_avg :
        use N past seconds for moving average
    post_avg :
        use N future seconds for moving average
    pre_max :
        use N past seconds for moving maximum
    post_max :
        use N future seconds for moving maximum

    Returns
    -------
    stamps : np.ndarray
    """

    import scipy.ndimage.filters as sf
    activations = activations.ravel()

    # detections are activations equal to the moving maximum
    max_length = int((pre_max + post_max) * fps) + 1
    if max_length > 1:
        max_origin = int((pre_max - post_max) * fps / 2)
        mov_max = sf.maximum_filter1d(
            activations, max_length, mode='constant', origin=max_origin)
        detections = activations * (activations == mov_max)
    else:
        detections = activations

    # detections must be greater than or equal to the moving average + threshold
    avg_length = int((pre_avg + post_avg) * fps) + 1
    if avg_length > 1:
        avg_origin = int((pre_avg - post_avg) * fps / 2)
        mov_avg = sf.uniform_filter1d(
            activations, avg_length, mode='constant', origin=avg_origin)
        detections = detections * (detections >= mov_avg + threshold)
    else:
        # if there is no moving average, treat the threshold as a global one
        detections = detections * (detections >= threshold)

    # convert detected onsets to a list of timestamps
    if combine:
        stamps = []
        last_onset = 0
        for i in np.nonzero(detections)[0]:
            # only report an onset if the last N frames none was reported
            if i > last_onset + combine:
                stamps.append(i)
                # save last reported onset
                last_onset = i
        stamps = np.array(stamps)
    else:
        stamps = np.where(detections)[0]

    # include corresponding activations per peak if needed
    if include_scores:
        scores = activations[stamps]
        if avg_length > 1:
            scores -= mov_avg[stamps]
        return stamps / float(fps), scores
    else:
        return stamps / float(fps)


def analysis(features, params, kernel_size):
    """Analysis of the audiofeature file with the current parameter settings.
    Calculates the SSM, NC and peaks of NC.

    Parameters
    ----------
    features : np.array
        Audiofeatures of the track extracted by extract_features_audio.py
    params : tuple
        Parameter tuples (smoothing_factor, downsampling_factor)
    kernel_size : int
        Kernel size of the gaussian kernel for Foote

    Returns
    -------
    ssm_f_mfcc : np.array
        SSM computed with MFCC features
    nc_mfcc : np.array
        NC computed for SSM based on MFCC features
    """

    # read features and abolish first two bands
    f_mfcc = features['f_mfcc'][2:]

    # normalize MFCC
    cur_f_mfcc = librosa.util.normalize(f_mfcc, norm=2, axis=0)

    # smooth MFCC
    cur_f_mfcc = smooth_features(cur_f_mfcc, win_len_smooth=params[0])

    # downsample by params[1]
    cur_f_mfcc = cur_f_mfcc[:, ::params[1]]

    # compute the SSM
    ssm_f_mfcc = compute_ssm(cur_f_mfcc)
    ssm_f_mfcc /= ssm_f_mfcc.max()

    # Compute gaussian kernel
    G = compute_kernel_checkerboard_gaussian(kernel_size / 2)

    # Compute the novelty curve
    nc_mfcc = compute_novelty_SSM(ssm_f_mfcc, G, exclude=True)
    nc_mfcc = np.abs(nc_mfcc) / nc_mfcc.max()
    # nc_mfcc = filters.gaussian_filter1d(nc_mfcc, sigma=2)

    return ssm_f_mfcc, nc_mfcc


def foote_experiment(track_db, params, thresholds, feature_rate, path_features, path_eval=None, save_ncs=False):
    eval_output = []

    # analyze and evaluate the dataset with different kernel sizes
    for cur_params in params:
        cur_kernel_size = cur_params['kernel_size']
        cur_wl_ds = cur_params['wl_ds']
        print('--> Params: {}, Kernel Size: {}'.format(cur_wl_ds, cur_kernel_size))

        # output container for the novelty functions
        nc_outputs = []

        # loop over all tracks
        for cur_track_name in tqdm.tqdm(track_db['track_name'].unique()):
            cur_track = track_db[track_db['track_name'] == cur_track_name]
            cur_boundaries_ref = jsd_utils.get_boundaries(cur_track, musical_only=True)

            try:
                assert len(cur_boundaries_ref) > 0
            except AssertionError:
                # print('Skipping Track {}: No musical boundaries.'.format(cur_track_name))
                continue

            # get the non-musical boundaries only
            cur_boundaries_ref_nm = jsd_utils.get_boundaries(cur_track, musical_only=False)
            cur_boundaries_ref_nm = np.setdiff1d(cur_boundaries_ref_nm, cur_boundaries_ref)

            # import info for the evaluation region
            time_first_musical_boundary = np.min(cur_boundaries_ref)
            time_last_musical_boundary = np.max(cur_boundaries_ref)

            # get path to audiofeature file
            cur_path_features = os.path.join(path_features, cur_track_name + '.npz')

            # load features
            features = np.load(cur_path_features)

            # analyze the audio features
            (ssm_mfcc, nc_mfcc) = analysis(features, cur_wl_ds, cur_kernel_size)

            # keep boundaries for visualization
            boundaries_mfcc = []
            boundaries_mfcc_eval = []
            est_boundaries_exclude = []

            for cur_threshold in thresholds:
                # Compute the peaks of the NCs
                # boundaries_mfcc = peak_picking(nc_mfcc, abs_thresh=np.tile(cur_threshold, len(nc_mfcc)))
                # boundaries_mfcc = detect_peaks(nc_mfcc, threshold=cur_threshold, fps=feature_rate / cur_wl_ds[1])
                prominence = 0.1
                cur_boundaries_mfcc = signal.find_peaks(nc_mfcc, height=cur_threshold, prominence=prominence)[0]
                cur_boundaries_mfcc = np.asarray(cur_boundaries_mfcc)
                cur_boundaries_mfcc = np.sort(cur_boundaries_mfcc)
                boundaries_mfcc.append(cur_boundaries_mfcc.copy())

                # convert frame indices to seconds
                cur_boundaries_mfcc = cur_boundaries_mfcc / (feature_rate / cur_wl_ds[1])

                # filter out boundary candidates around non-musical boundaries
                window = 3.0  # in seconds
                cur_est_boundaries_exclude_idcs = []

                for cur_boundary_nm in cur_boundaries_ref_nm:
                    # check if estimate is in near nm-boundary
                    cur_excludes = np.abs(cur_boundaries_mfcc - cur_boundary_nm)
                    cur_excludes = np.where(cur_excludes <= window)
                    cur_excludes = cur_excludes[0].tolist()

                    cur_est_boundaries_exclude_idcs.extend(cur_excludes)

                # add boundaries outside evaluation window [t_first - window, t_last + window]
                cur_est_boundaries_exclude_idcs.extend(np.where(cur_boundaries_mfcc < (time_first_musical_boundary - window))[0].tolist())
                cur_est_boundaries_exclude_idcs.extend(np.where(cur_boundaries_mfcc > (time_last_musical_boundary + window))[0].tolist())

                cur_est_boundaries_exclude_idcs = np.asarray(cur_est_boundaries_exclude_idcs)
                cur_est_boundaries_exclude_idcs = np.unique(cur_est_boundaries_exclude_idcs)

                if cur_est_boundaries_exclude_idcs.shape[0] > 1:
                    cur_est_boundaries_exclude = cur_boundaries_mfcc[cur_est_boundaries_exclude_idcs]
                else:
                    cur_est_boundaries_exclude = []

                est_boundaries_exclude.append(cur_est_boundaries_exclude)
                if len(cur_est_boundaries_exclude_idcs) > 0:
                    cur_boundaries_mfcc = np.delete(cur_boundaries_mfcc, cur_est_boundaries_exclude_idcs)
                boundaries_mfcc_eval.append(cur_boundaries_mfcc)

                cur_eval_row = {}
                cur_eval_row['track_name'] = cur_track_name
                cur_eval_row['wl_ds'] = cur_wl_ds
                cur_eval_row['kernel_size'] = cur_kernel_size
                cur_eval_row['threshold'] = cur_threshold

                # evaluate the boundaries for 0.5 seconds
                cur_eval_row['F_mfcc_05'], cur_eval_row['P_mfcc_05'], cur_eval_row['R_mfcc_05'] = \
                    mir_eval.onset.f_measure(cur_boundaries_ref, cur_boundaries_mfcc, window=0.5)

                # evaluate the boundaries for 3.0 seconds
                cur_eval_row['F_mfcc_3'], cur_eval_row['P_mfcc_3'], cur_eval_row['R_mfcc_3'] = \
                    mir_eval.onset.f_measure(cur_boundaries_ref, cur_boundaries_mfcc, window=3.0)

                # add dataframe of one track to dataframe of all songs
                eval_output.append(cur_eval_row)

            # debugging visualization
            debug = False
            if debug:
                import matplotlib.pyplot as plt
                scaler = (feature_rate / cur_wl_ds[1])
                threshold_idx = 0
                fig, ax = plt.subplots(nrows=2)
                fig.suptitle('{}, Params: {}, Threshold: {:.2f}'.format(cur_track_name, cur_params, thresholds[threshold_idx]))
                ax[0].imshow(features['f_mfcc'][2:], origin='lower', aspect='auto')
                ax[1].plot(nc_mfcc, label='NC')
                ax[1].set_xlim(0, len(nc_mfcc))
                ax[1].set_ylim(0, 1)

                ax[1].vlines(np.round(cur_boundaries_ref * scaler), 0.8, 1, color='g', label='Ref.')
                ax[1].vlines(np.round(cur_boundaries_ref_nm * scaler), 0.7, 1, color='b', label='Ref. NM')

                ax[1].vlines(boundaries_mfcc[threshold_idx], 0, 0.8, color='r', label='Est.')
                ax[1].vlines(np.round(boundaries_mfcc_eval[threshold_idx] * scaler), 0, 0.3, color='k', label='Est. Eval')
                if len(est_boundaries_exclude[threshold_idx]) > 0:
                    ax[1].vlines(np.round(est_boundaries_exclude[threshold_idx] * scaler), 0, 0.5, color='y', label='Excluded')

                ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

                plt.show()

            # save ncs
            if save_ncs:
                cur_nc_output = {}
                cur_nc_output['track_name'] = cur_track_name
                cur_nc_output['wl_ds'] = cur_wl_ds
                cur_nc_output['kernel_size'] = cur_kernel_size
                cur_nc_output['nc'] = nc_mfcc
                cur_nc_output['ssm'] = ssm_mfcc
                cur_nc_output['boundaries'] = boundaries_mfcc[0]
                cur_nc_output['feature_rate'] = feature_rate

                nc_outputs.append(cur_nc_output)

        if save_ncs:
            fn_nc_output = 'ncs_wl-{}_kernelsize-{}.npz'.format(cur_wl_ds, cur_kernel_size)
            np.savez_compressed(os.path.join(path_eval, fn_nc_output), nc_outputs=nc_outputs)

    eval_output = pd.DataFrame(eval_output)

    return eval_output
