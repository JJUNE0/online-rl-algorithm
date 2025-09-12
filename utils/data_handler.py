import numpy as np
from scipy.stats import truncnorm
from numba import vectorize


@vectorize
def fast_clip(x, l, u):
    return max(min(x, u), l)

def constant_noisy_data(data, noise):
    return data + noise

def uniform_noisy_data(data, lower, upper):
    noise = np.random.uniform(lower, upper, size=data.shape)
    return data + noise

def uniform_noisy_scalar_data(data, lower_margin, upper_margin):
    noise = np.random.uniform(-data*lower_margin, data*upper_margin)
    return data + noise

def gaussian_noisy_data(data, mean, std):
    noise = np.random.normal(mean, std, size=data.shape)
    return data + noise

def gaussian_noisy_scalar_data(data, mean, std):
    noise = np.random.normal(mean, std)
    return data + noise

def truncated_gaussian_noisy_data(data, mean, std, lower, upper):
    a = (lower - mean) / std
    b = (upper - mean) / std
    noise = truncnorm.rvs(a, b, loc=mean, scale=std, size=data.shape)
    return data + noise

def truncated_gaussian_noisy_scalar_data(data, mean, std, lower, upper):
    a = (lower - mean) / std
    b = (upper - mean) / std
    noise = truncnorm.rvs(a, b, loc=mean, scale=std)
    return data + noise
