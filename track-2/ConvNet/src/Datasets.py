import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class BnuDataset(Dataset):
    def __init__(self, dir_path, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        # extract the data
        x1 = np.load(os.path.join(dir_path, 'bnu1.npy'))
        x2 = np.load(os.path.join(dir_path, 'bnu1.npy'))
        x = np.vstack((x1, x2))
        y = np.loadtxt(os.path.join(dir_path, 'bnu.csv'), skiprows=1)
        num_obj, num_frames, num_rois = x.shape

        # convert time series to FC
        self.x = np.empty((num_obj, num_rois, num_rois))
        for i in range(num_obj):
            self.x[i] = np.corrcoef(x[i], rowvar=False)
        
        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            self.target_transform(y)
        return x, y
    
class IhbDataset(BnuDataset):
    def __init__(self, dir_path, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.transform = transform
        self.target_transform = target_transform

        # extract the data
        x = np.load(os.path.join(dir_path, 'ihb.npy'))
        y = np.loadtxt(os.path.join(dir_path, 'ihb.csv'), skiprows=1)
        num_obj, num_frames, num_rois = x.shape

        # convert time series to FC
        self.x = np.empty((num_obj, num_rois, num_rois))
        for i in range(num_obj):
            self.x[i] = np.corrcoef(x[i], rowvar=False)
        
        self.x = torch.from_numpy(self.x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)

def make_heatmap(DatasetClass):
    file_dir = os.path.dirname(__file__)

    dataset = DatasetClass(os.path.join(file_dir, '../../contest-data'))
    
    # calculate means
    mean_neg = torch.mean(dataset.x[dataset.y == 0], 0)
    mean_pos = torch.mean(dataset.x[dataset.y == 1], 0)

    if not os.path.exists(os.path.join(file_dir, '../heatmaps')):
        os.makedirs(os.path.join(file_dir, '../heatmaps'))

    # make plots
    fig, ax = plt.subplots()
    im = ax.imshow(mean_neg)
    ax.set_title(f'{DatasetClass.__name__} mean pearsons correlation coefficient of negative objects')
    cbar = ax.figure.colorbar(im)
    plt.savefig(os.path.join(file_dir, f'../heatmaps/{DatasetClass.__name__}-maen-corr-matrix-neg.png'))
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(mean_pos)
    ax.set_title(f'{DatasetClass.__name__} mean pearsons correlation coefficient of positive objects')
    cbar = ax.figure.colorbar(im)
    plt.savefig(os.path.join(file_dir, f'../heatmaps/{DatasetClass.__name__}-maen-corr-matrix-pos.png'))
    plt.close()

if __name__ == '__main__':
    make_heatmap(BnuDataset)
    make_heatmap(IhbDataset)