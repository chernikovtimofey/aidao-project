import os
import numpy as np
import scipy
import dvc.api
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # extracting params
    params = dvc.api.params_show()['preprocess']
    np.random.seed(params['seed'])

    # extracting the data
    script_dir = os.path.dirname(__file__)
    data = np.load(os.path.join(script_dir, '../../contest-data.npy'))
    num_objs, num_frames, num_rois = data.shape

    # standartizing data
    scaler = StandardScaler()
    data = np.reshape(data, (-1, num_rois))
    data = scaler.fit_transform(data)
    data = np.reshape(data, (num_objs, num_frames, num_rois))

    # replacing nan to noize 
    data[np.isnan(data)] = np.random.normal(size=data[np.isnan(data)].shape)

    # converting time series to FC
    corr_data = np.empty((num_objs, num_rois, num_rois))
    preprocessed_data = np.empty((num_objs, (1 + num_rois) * num_rois // 2))
    for i in range(num_objs):
        corr_data[i] = np.corrcoef(data[i], rowvar=False)
        preprocessed_data[i] = corr_data[i][np.triu_indices(num_rois)].flatten()

    np.save(os.path.join(script_dir, '../transformed-data/corr-data.npy'), corr_data)
    np.save(os.path.join(script_dir, '../transformed-data/preprocessed-data.npy'), preprocessed_data)

if __name__ == '__main__':
    preprocess_data()