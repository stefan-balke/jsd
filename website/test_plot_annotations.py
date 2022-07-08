# Test function for display

import pandas as pd
import matplotlib.pyplot as pl

from display import plot_annotations

if __name__ == '__main__':
    fn_csv = 'test/CliffordBrown_Jordu_Orig.csv'
    cur_annotation = pd.read_csv(fn_csv, sep=';', header=0)
    path_instrument_images = 'instrument_images'

    ax = pl.subplot()
    plot_annotations(cur_annotation, path_instrument_images, axes=ax)
    pl.savefig('test/CliffordBrown_Jordu_Orig.png')
    pl.close()
