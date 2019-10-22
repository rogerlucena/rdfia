import numpy as np
from sklearn.cluster import KMeans
from tools import compute_split

def compute_visual_dict(sift, n_clusters=1000, n_init=1, verbose=1):
    # reorder data
    dim_sift = sift[0].shape[-1]
    sift = [s.reshape(-1, dim_sift) for s in sift]
    sift = np.concatenate(sift, axis=0)
    # remove zero vectors
    keep = ~np.all(sift==0, axis=1)
    sift = sift[keep]
    # randomly pick sift
    ids, _ = compute_split(sift.shape[0], pc=0.05)
    sift = sift[ids]
    
    # Compute kmeans on `sift`, get cluster centers, add zeros vector - # done
    kmeans = KMeans(n_clusters = n_clusters, init = 'random').fit(sift)
    cluster_centers = kmeans.cluster_centers_
    zero_cluster_center = np.zeros(128).reshape(1, -1)
    cluster_centers = np.append(cluster_centers, zero_cluster_center, axis=0)
    
    return cluster_centers