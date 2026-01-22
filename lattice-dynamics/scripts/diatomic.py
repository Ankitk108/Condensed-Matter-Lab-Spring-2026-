import numpy as np

def diatomic_dispersion(k, K, m1, m2):
    term1 = K*(1/m1 + 1/m2)
    term2 = K*np.sqrt(
        (1/m1 + 1/m2)**2 - 4*(np.sin(k/2)**2)/(m1*m2)
    )
    omega_acoustic = np.sqrt(term1 - term2)
    omega_optical  = np.sqrt(term1 + term2)
    return omega_acoustic, omega_optical
