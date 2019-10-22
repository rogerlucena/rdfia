import numpy as np

from tools import dense_sampling, auto_padding, compute_grad_ori, conv_separable

def compute_grad(I):
    ha = np.matrix([[0.5], [1], [0.5]])
    hb = np.matrix([[-0.5], [0], [0.5]])
    Ix = conv_separable(I, hb, ha)
    Iy = conv_separable(I, ha, hb)
    return Ix, Iy

def compute_grad_mod_ori(I):
    Ix, Iy = compute_grad(I)
    Gm = (Ix**2 + Iy**2)**(0.5)
    Go = compute_grad_ori(Ix, Iy, Gm)
    return Gm, Go

# Compute SIFT over a patch
# Assuming Gm and Go are already 16x16 in size
def compute_sift_region(Gm, Go, mask=None):
    
    if mask is not None:
        Gm = np.multiply(Gm, mask)
    
    def compute_subregion(Gm, Go):
        Renck = np.zeros(8)
        for i in range(Gm.shape[0]):
            for j in range(Gm.shape[1]):
                Renck[Go[i,j]] += Gm[i,j]
        return Renck
    
    Penck = np.array([])
    for j in range(4):
        for i in range(4):
            Gmp = Gm[i*4:(i+1)*4,j*4:(j+1)*4]
            Gop = Go[i*4:(i+1)*4,j*4:(j+1)*4]
            Penck = np.concatenate((Penck, compute_subregion(Gmp, Gop)))
    
    Penck_norm = np.linalg.norm(Penck)
    
    if (Penck_norm < 0.5):
        return np.zeros(128)
    
    Penck = Penck / Penck_norm
    Penck = np.vectorize(lambda t: 0.2 if t > 0.2 else t)(Penck)
    Penck_norm = np.linalg.norm(Penck)
    Penck = Penck / Penck_norm
            
#     # Note: to apply the mask only when given, do:
#     if mask is not None:
#         pass # TODO apply mask here
    return Penck

def compute_sift_image(I):
    x, y = dense_sampling(I)
    im = auto_padding(I)
    Gm, Go = compute_grad_mod_ori(im)
    
    sifts = np.zeros((len(x), len(y), 128))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            sifts[i, j, :] = compute_sift_region(Gm[xi:xi+16, yj:yj+16], Go[xi:xi+16, yj:yj+16])
    return sifts

