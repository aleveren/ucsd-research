'''
Get statistics for several different types of probability distributions
based on vectors of parameters
'''

import numpy as np

def mean_dirichlet(X, axis = -1):
    return X / X.sum(axis = axis, keepdims = True)

def norm_variance_dirichlet(X, axis = -1):
    a0 = X.sum(axis = axis, keepdims = True)
    variance = X * (a0 - X) / (a0 ** 2 * (a0 + 1.0))
    norm_variance = np.linalg.norm(variance, axis = axis)
    return norm_variance

def mean_discrete(p, axis = -1, keepdims = False):
    axis = axis if axis >= 0 else len(p.shape) + axis
    values_target_shape = [1 if i != axis else p.shape[axis] for i in range(len(p.shape))]
    values = np.arange(p.shape[axis]).reshape(values_target_shape)
    values, p = np.broadcast_arrays(values, p)
    return np.sum(values * p, axis = axis, keepdims = keepdims)

def variance_discrete(p, axis = -1):
    axis = axis if axis >= 0 else len(p.shape) + axis
    values_target_shape = [1 if i != axis else p.shape[axis] for i in range(len(p.shape))]
    values = np.arange(p.shape[axis]).reshape(values_target_shape)
    values, p = np.broadcast_arrays(values, p)
    zzz = np.sum((values ** 2) * p, axis = axis)
    mean_sq = mean_discrete(p, axis = axis) ** 2
    return zzz - mean_sq
