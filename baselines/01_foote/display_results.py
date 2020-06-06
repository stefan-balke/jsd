import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_csv('data/evaluation/evaluation_winlength-3.csv', sep=';', index_col=0)
    print(data.groupby(['param', 'kernel_size'])[['F_mfcc']].mean())
    data = data[['param', 'kernel_size', 'F_mfcc', 'F_cens']].melt(id_vars=['param', 'kernel_size'])
    data.boxplot(by=['variable', 'param', 'kernel_size'], rot=90)
    plt.tight_layout()
    plt.show()
