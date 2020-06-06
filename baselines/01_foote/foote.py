"""
    This script identifies the boundaries of a given track using the Foote
    method:

    Foote, J. (2000). Automatic Audio Segmentation Using a Measure Of Audio
    Novelty. In Proc. of the IEEE International Conference of Multimedia and
    Expo (pp. 452–455). New York City, NY, USA.
"""

import numpy as np
import scipy.signal
from scipy import signal


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
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode='constant')

    for n in range(N):
        # Does not work with numba/jit
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
        abs_thresh = np.tile(0.5*np.median(x), len(x))
    if rel_thresh is None:
        rel_thresh = 0.5*np.tile(np.sqrt(np.var(x)), len(x))
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
        dy = x[cur_idx+direction] - x[cur_idx]

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
                            P.append(candidate)    # verified candidate as True peak
                    searching_peak = True
                descent = 0
        dyold = dy
    return P
