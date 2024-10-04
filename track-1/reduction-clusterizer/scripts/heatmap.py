import os
import numpy as np
from matplotlib import pyplot as plt
import dvc.api
from dvclive import Live

def make_heatmap():
    # extracting the data
    script_dir = os.path.dirname(__file__)
    data = np.load(os.path.join(script_dir, '../transformed-data/corr-data.npy'))

    # making plots
    for i in range(20):
        fig, ax = plt.subplots()
        im = ax.imshow(data[i])
        ax.set_title(f'Pearsons correlation coefficient of object {i}')
        cbar = ax.figure.colorbar(im)
        plt.savefig(os.path.join(script_dir, f'../heatmaps/corr-matrix-{i}.png'))
        plt.close()

if __name__ == '__main__':
    make_heatmap()