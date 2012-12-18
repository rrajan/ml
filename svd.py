import numpy as np

def svd(mat):
    # do np.transpose(mat) * mat if need to compute covariance
    out = np.linalg.svd(mat)
    return out
