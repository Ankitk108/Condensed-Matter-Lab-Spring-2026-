import numpy as np

def monatomic_dispersion(k, K, m):
    return 2*np.sqrt(K/m)*np.abs(np.sin(k/2))
