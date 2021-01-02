import yaml
import numpy as np


if __name__ == '__main__':
    with open('data/salami_split.yml') as fh:
        split = yaml.load(fh, Loader=yaml.FullLoader)

    n_train = len(split['train'])
    n_test = len(split['test'])
    n_val = len(split['val'])
    n_total = n_train + n_test + n_val

    print('Total: {}'.format(n_total))
    print('Train: {}, {:0.1f}'.format(n_train, (n_train / n_total) * 100))
    print('Test: {}, {:0.1f}'.format(n_test, (n_test / n_total) * 100))
    print('Val: {}, {:0.1f}'.format(n_val, (n_val / n_total) * 100))
