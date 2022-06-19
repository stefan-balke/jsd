import mir_eval
import numpy as np


ref = np.asarray([1, 2, 3, 4, 5, 6])
est1 = np.asarray([1, 2, 3, 4, 5, 6])
est2 = np.asarray([1, 2, 3])
est3 = np.asarray([1, 2, 3, 11, 12, 13, 14, 15, 16])  # high false-positives

f, p, r = mir_eval.onset.f_measure(ref, est3, window=3.0)

print(p, r, f)