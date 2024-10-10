import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

class FCDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.transform = transform

        # preprocess data
        data = np.load(file_path)
        num_obj, num_frames, num_rois = data.shape

        # standartize data
        scaler = StandardScaler()
        data = np.reshape(data, (-1, num_rois))
        data = scaler.fit_transform(data)
        data = np.reshape(data, (num_obj, num_frames, num_rois))

        # transform nan to noize
        data[np.isnan(data)] = np.random.normal(size=data[np.isnan(data)].shape)

        # convert time series to FC
        self.data = np.empty((num_obj, (1 + num_rois) * num_rois // 2))
        for i in range(num_obj):
            self.data[i] = np.corrcoef(data[i], rowvar=False)[np.triu_indices(num_rois)]

        self.data = torch.from_numpy(self.data).to(torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        obj = self.data[idx]
        if self.transform:
            obj = self.transform(obj)
        return obj
    
def make_heatmap():
    np.random.seed(0)

    # extracting the data
    file_dir = os.path.dirname(__file__)
    dataset = FCDataset(os.path.join(file_dir, '../../contest-data.npy'))
    dataloader = DataLoader(dataset, batch_size=20)

    # making plots
    for idx, vec in enumerate(next(iter(dataloader))):
        # convert vector to matrix
        matrix = torch.empty((246, 246))
        vec_idx = 0
        for i in range(246):
            for j in range(i, 246):
                matrix[i, j] = vec[vec_idx]
                matrix[j, i] = matrix[i, j]
                vec_idx += 1

        if not os.path.exists(os.path.join(file_dir, '../heatmaps')):
            os.makedirs(os.path.join(file_dir, '../heatmaps'))

        fig, ax = plt.subplots()
        im = ax.imshow(matrix)
        ax.set_title(f'Pearsons correlation coefficient of object {idx}')
        cbar = ax.figure.colorbar(im)
        plt.savefig(os.path.join(file_dir, f'../heatmaps/corr-matrix-{idx}.png'))
        plt.close()

if __name__ == '__main__':
    make_heatmap()