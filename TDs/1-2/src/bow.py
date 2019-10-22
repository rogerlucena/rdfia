import numpy as np

# Compute BoW for an image
def compute_feats(vdict, image_sifts):
    # reorder data
    dim_sift = image_sifts.shape[-1]
    im_sifts = image_sifts.reshape(-1, dim_sift)
    
    vdict = vdict / 255
    
    # compute BoW from `image_sifts` - done
    # coding
    H = []
    for sift in im_sifts:
        distances_vectors = vdict-sift
        distances = []
        for v in distances_vectors:
            distances.append(np.linalg.norm(v))
        
        cluster_index = np.argmin(distances)
        one_hot = np.zeros(len(vdict))
        one_hot[cluster_index] = 1
        H.append(one_hot)
    H = np.array(H)
    
    # pooling
    z = np.sum(H, axis=0)
    
    # normalization
    z = z / (np.linalg.norm(z))
    
    return z