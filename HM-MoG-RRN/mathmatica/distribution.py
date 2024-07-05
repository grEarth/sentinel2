import numpy as np
from numba import njit

@njit
def gaussian_distribution(x, mu, sigma):
    probability = np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
    return probability

@njit
def laplace_distribution(x, mu, beta):
    probability = np.exp(-np.abs(x-mu)/(beta))/(2*beta)
    return probability