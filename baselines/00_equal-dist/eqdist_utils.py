import numpy as np
import pandas as pd


def get_baseline_segments(track_dur, n_segments, silence_start, silence_end):
    """Takes the number of annotations per track and the track duration.
    The boundaries are then spread equally along the time axis.

    Parameters
    ----------
    track_dur : float
        Duration of the track in seconds.

    n_segments : int
        Number of segments given in the annotations.

    silence_start : float
        Silence to add at beginning.

    silence_end : float
        Silence to add at the end.

    Returns
    -------
    track_data : pd.DataFrame
        Track with segements.
    """

    boundaries = []
    boundaries.append({'segment_start': 0,
                       'segment_end': silence_start,
                       'label': 'silence'})

    # add silence to start
    boundary_starts = np.linspace(silence_start,
                                  track_dur - silence_end,
                                  num=int(n_segments - 1))

    for cur_idx in np.arange(len(boundary_starts) - 1):
        cur_boundary = dict()
        cur_boundary['segment_start'] = boundary_starts[cur_idx]
        cur_boundary['segment_end'] = boundary_starts[cur_idx + 1]
        cur_boundary['label'] = 'musical_segment'
        boundaries.append(cur_boundary)

    # add silence to end
    boundaries.append({'segment_start': track_dur - silence_end,
                       'segment_end': track_dur,
                       'label': 'silence'})

    boundaries = pd.DataFrame(boundaries)

    return boundaries
