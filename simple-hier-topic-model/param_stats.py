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

def topic_difference(true_topics, est_topics):
    num_topics = true_topics.shape[0]
    assert est_topics.shape[0] == true_topics.shape[0]

    # Greedy algorithm to reorder nodes such that order of true_topics corresponds to order of est_topics
    orig_leaf_indices = list(range(1, num_topics))
    est_leaf_topics = est_topics[1:, :]
    reorder_nodes = [0]
    for i in range(1, num_topics):
        leaf_topic = true_topics[np.newaxis, i, :]
        matching_index = np.argmin(np.sum(np.abs(est_leaf_topics - leaf_topic), axis=-1))
        reorder_nodes.append(orig_leaf_indices[matching_index])
        orig_leaf_indices = orig_leaf_indices[:matching_index] + orig_leaf_indices[matching_index+1:]
        est_leaf_topics = np.concatenate((est_leaf_topics[:matching_index, :], est_leaf_topics[matching_index+1:, :]))

    # Reorder the nodes in est_topics and compare to true topics
    est_topics = est_topics[reorder_nodes, :]
    scores = 0.5 * np.abs(true_topics - est_topics).sum(axis=-1)
    return np.mean(scores)
