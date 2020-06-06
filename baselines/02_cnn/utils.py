import yaml
import numpy as np
import librosa


def load_config(path_config=False):
    """Load config from YAML file."""

    if not path_config:
        path_config = 'configs/config.yml'

    with open(path_config, 'rb') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    return config


def gaussian(win_len):
    """Creates a gaussian window.

    Parameters
    ----------
    win_len : int
        Window length

    Returns
    -------
    win : np.ndarray [shape=(win_len, )]
        Gaussian window
    """

    d = np.arange(win_len)
    win_len_half = int(win_len / 2)

    d = np.abs(d - win_len_half)

    win = np.exp(-2*d**2 / win_len_half**2)

    return win


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
        mov_max = sf.maximum_filter1d(activations, max_length, mode='constant', origin=max_origin)
        detections = activations * (activations == mov_max)
    else:
        detections = activations

    # detections must be greater than or equal to the moving average + threshold
    avg_length = int((pre_avg + post_avg) * fps) + 1
    if avg_length > 1:
        avg_origin = int((pre_avg - post_avg) * fps / 2)
        mov_avg = sf.uniform_filter1d(activations, avg_length, mode='constant', origin=avg_origin)
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


def pink(N, depth=80):
    """
    N-length vector with (approximate) pink noise
    pink noise has 1/f PSD
    Source: https://github.com/aewallin/allantools/blob/master/allantools/noise.py

    """
    a = []
    s = iterpink(depth)
    for n in range(N):
        a.append(next(s))
    return np.asarray(a)


def iterpink(depth=20):
    """Generate a sequence of samples of pink noise.
    pink noise generator
    from http://pydoc.net/Python/lmj.sound/0.1.1/lmj.sound.noise/
    Based on the Voss-McCartney algorithm, discussion and code examples at
    http://www.firstpr.com.au/dsp/pink-noise/
    depth: Use this many samples of white noise to calculate the output. A
      higher  number is slower to run, but renders low frequencies with more
      correct power spectra.
    Generates a never-ending sequence of floating-point values. Any continuous
    set of these samples will tend to have a 1/f power spectrum.

    Source: https://github.com/aewallin/allantools/blob/master/allantools/noise.py
    """
    values = np.random.randn(depth)
    smooth = np.random.randn(depth)
    source = np.random.randn(depth)
    sumvals = values.sum()
    i = 0
    while True:
        yield sumvals + smooth[i]

        # advance the index by 1. if the index wraps, generate noise to use in
        # the calculations, but do not update any of the pink noise values.
        i += 1
        if i == depth:
            i = 0
            smooth = np.random.randn(depth)
            source = np.random.randn(depth)
            continue

        # count trailing zeros in i
        c = 0
        while not (i >> c) & 1:
            c += 1

        # replace value c with a new source element
        sumvals += source[i] - values[c]
        values[c] = source[i]


def add_noise_patch(noise_duration, noise_level=-70, sr=44100, hop_size=1024):
    """Creates pink noise patch of size noise_duration

    Parameters
    ----------
    noise_duration : int
        Noise duration in frames
    noise_level : int
        Volume in db of noise
    sr: int
        Sampling rate
    hop_size: int
        hopsize

    Returns
    -------
    n_mel : np.ndarray [shape=(noise_duration, 80)]
        Pink noise in Mel spectra
    """
    noise = pink(noise_duration*hop_size)

    # transform noise in mel spectrogram
    n_mel = librosa.feature.melspectrogram(y=noise, sr=sr, S=None, n_fft=2*hop_size,
                                           hop_length=hop_size, power=2.0, fmin=80, fmax=16000, n_mels=80)
    # convert to db
    n_mel = librosa.power_to_db(n_mel) + noise_level

    return n_mel
