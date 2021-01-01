import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_csv('data/evaluation/evaluation_winlength-05.csv', sep=';', index_col=0)
    print(data.groupby(['param', 'kernel_size', 'threshold'])[['F_mfcc']].mean())
    data = data[['param', 'kernel_size', 'F_mfcc', 'F_cens', 'threshold']].melt(id_vars=['param', 'kernel_size', 'threshold'])
    data.boxplot(by=['variable', 'param', 'kernel_size', 'threshold'], rot=90)
    plt.tight_layout()
    plt.show()
