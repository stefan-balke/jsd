import numpy as np
import pandas as pd
import os
import sys
import glob
import mir_eval

# hacky relative import
sys.path.append(os.path.join('..', '..'))
import utils


def get_baseline_boundaries(track_dur, n_boundaries):
    """Takes the number of annotations per track and the track duration.
    The boundaries are then spread equally along the time axis.

    Parameters
    ----------
    track_dur : float
        Duration of the track in seconds.
    
    n_boundaries : int
        Number of boundaries given the annotations.

    Returns
    -------
    boundaries : np.ndarray
        Positions of the boundaries in seconds.
    """

    boundaries = np.linspace(0, track_dur, num=int(n_boundaries))

    return boundaries


def main():
    PATH_DATA = os.path.join('..', '..', 'data')
    path_annotations = os.path.join(PATH_DATA, 'annotations_csv')
    path_annotation_files = glob.glob(os.path.join(path_annotations, '*.csv'))
    jsd_track_db = utils.load_jsd(path_annotation_files)
    jsd_track_dur = pd.read_csv(os.path.join(PATH_DATA, 'track_durations.csv'))

    # for evaluation
    F_05 = 0
    P_05 = 0
    R_05 = 0
    F_3 = 0
    P_3 = 0
    R_3 = 0

    for cur_window in [0.5, 3.0]:
        Fs = []
        Ps = []
        Rs = []

        for _, cur_track in jsd_track_dur.iterrows():
            cur_jsd_track = jsd_track_db[jsd_track_db['track_name'] == cur_track['track_name']]
            cur_boundaries_ref = utils.get_boundaries(cur_jsd_track)
            cur_boundaries_est = get_baseline_boundaries(cur_track['duration'], cur_boundaries_ref.shape[0])
            F, P, R = mir_eval.onset.f_measure(cur_boundaries_ref,
                                               cur_boundaries_est,
                                               window=cur_window)
            Fs.append(F)
            Rs.append(R)
            Ps.append(P)

        if cur_window == 0.5:
            F_05 = np.mean(Fs)
            P_05 = np.mean(Ps)
            R_05 = np.mean(Rs)

        if cur_window == 3.0:
            F_3 = np.mean(Fs)
            P_3 = np.mean(Ps)
            R_3 = np.mean(Rs)

    print('0.5 s window: F={0:.3f}, P={1:.3f}, R={2:.3f}'.format(F_05, P_05, R_05))
    print('3.0 s window: F={0:.3f}, P={1:.3f}, R={2:.3f}'.format(F_3, P_3, R_3))


if __name__ == '__main__':
    main()
