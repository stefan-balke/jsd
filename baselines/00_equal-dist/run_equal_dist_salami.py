import numpy as np
import pandas as pd
import os
import sys
import glob
import mir_eval
import eqdist_utils
import tqdm

# hacky relative import
sys.path.append(os.path.join('..', '..'))
import jsd_utils


def load_salami(track_durs, path_annotations):
    salami_data = []
 
    for cur_track_path in tqdm.tqdm(glob.glob(os.path.join(path_annotations, '*.txt'))):
        cur_track_name = os.path.splitext(os.path.basename(cur_track_path))[0]

        try:
            cur_track_data = pd.read_csv(cur_track_path, sep='\t', header=None, names=['segment_start', 'label'])
        except FileNotFoundError:
            print(cur_track_path)
            continue
        
        cur_track_data['label'] = cur_track_data['label'].str.lower()
        cur_track_data['track_name'] = cur_track_name
        cur_track_data['segment_end'] = cur_track_data['segment_start'].shift(-1)
        cur_track_data['segment_dur'] = cur_track_data['segment_end'] - cur_track_data['segment_start']
        cur_track_data = cur_track_data[:-1]
        salami_data.append(cur_track_data)

    salami_data = pd.concat(salami_data)

    return salami_data


def main():
    PATH_DATA = os.path.join('..', '02_cnn', 'data')
    path_annotations = os.path.join(PATH_DATA, 'salami_annotations')
    track_durs = pd.read_csv(os.path.join('salami_track_durations.csv'))
    track_durs = track_durs.astype(str)

    salami_data = load_salami(track_durs, path_annotations)

    silence_start = salami_data[(salami_data['segment_start'] == 0) & (salami_data['label'] == 'silence')]
    silence_start_median = silence_start.groupby('label').median()['segment_dur'].values[0]
    silence_end = salami_data[(salami_data['segment_start'] != 0) & (salami_data['label'] == 'silence')]
    silence_end_median = silence_end.groupby('label').median()['segment_dur'].values[0]

    print('Start median: {}, End Median: {}'.format(silence_start_median, silence_end_median))

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

        for _, cur_track_data in salami_data.groupby('track_name'):
            baseline_track = eqdist_utils.get_baseline_segments(cur_track_data.tail(1)['segment_end'].values[0],
                                                                cur_track_data.shape[0],
                                                                silence_start_median,
                                                                silence_end_median)
            # print(baseline_track)
            cur_boundaries_ref = jsd_utils.get_boundaries(cur_track_data, musical_only=True)
            cur_boundaries_est = jsd_utils.get_boundaries(baseline_track, musical_only=True)

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
