import os
import numpy as np
from sklearn.cluster import KMeans

def clusterize():
    # extract the data
    file_dir = os.path.dirname(__file__)
    data = np.load(os.path.join(file_dir, '../encoded_data.npy'))   

    # clusterize
    clusterizer = KMeans(n_clusters=20)
    cluster_distances = clusterizer.fit_transform(data)
    
    # make predictions
    preds = np.empty(len(data))
    leftover_indexes = np.arange(len(data))
    for i in range(20):
        current_distances = cluster_distances[:, i]
        if len(current_distances) > 16:
            top16 = np.argpartition(current_distances, 16)[:16]
            preds[leftover_indexes[top16]] = i
            cluster_distances = np.delete(cluster_distances, top16, axis=0)
            leftover_indexes = np.delete(leftover_indexes, top16)
        else:
            preds[leftover_indexes] = i
            break

    np.savetxt(os.path.join(file_dir, '../submission.csv'), preds, fmt='%i', header='prediction', comments='')

if __name__ == '__main__':
    clusterize()