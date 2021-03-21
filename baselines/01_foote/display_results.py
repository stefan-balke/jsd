import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_csv('../data/foote_evaluation/eval_means_salami.csv', sep=';', index_col=0)
    print(data.groupby(['wl_ds', 'kernel_size', 'threshold'])[['R_mfcc_05']].mean())
    data = data[['wl_ds', 'kernel_size', 'R_mfcc_05', 'threshold']].melt(id_vars=['wl_ds', 'kernel_size', 'threshold'])
    data.boxplot(by=['variable', 'wl_ds', 'kernel_size', 'threshold'], rot=90)
    plt.tight_layout()
    plt.show()
