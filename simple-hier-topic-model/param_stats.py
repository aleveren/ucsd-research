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

def topic_difference(true_topics, est_topics, paths):
    '''
    Measures the difference between trees of topics
    while attempting to ignore structure-preserving permutations.

    Answer should be correct if all topics in true_topics are distinct
    and est_topics is a structure-preserving permutation of true_topics.
    It should be approximately correct if there is a small amount of noise added.
    Larger amounts of noise may lead to an overestimation of the "true" difference.
    '''

    permute_nodes = find_structural_permutation(
        true_topics = true_topics,
        est_topics = est_topics,
        paths = paths)

    # Apply permutation to est_topics and compare to true topics
    est_topics = est_topics[permute_nodes, :]
    scores = 0.5 * np.abs(true_topics - est_topics).sum(axis=-1)
    return np.mean(scores)

def find_structural_permutation(true_topics, est_topics, paths):
    '''
    Greedy heuristic for attempting to find a structure-preserving permutation of the nodes
    in a tree of topics that makes est_topics look as similar as possible to true_topics.

    Note: this algorithm may not find the optimal answer.
    '''

    num_topics = true_topics.shape[0]
    assert est_topics.shape == true_topics.shape
    assert len(paths) == num_topics

    # Base case: single topic
    if num_topics == 1:
        return np.arange(num_topics)

    permute_nodes = np.arange(num_topics)

    child_indices_by_path = dict()
    descendants_by_child_index = dict()
    nodes_counted = 1  # start with 1 for root
    for i, p in enumerate(paths):
        if len(p) == 1:
            child_indices_by_path[p] = i
            descendants_by_child_index[i] = []
            nodes_counted += 1
    for i, p in enumerate(paths):
        if len(p) > 1:
            parent = p[:-1]
            parent_index = child_indices_by_path[parent]
            descendants_by_child_index[parent_index].append(i)
            nodes_counted += 1
    assert nodes_counted == num_topics

    child_indices = np.asarray(sorted(child_indices_by_path.values()))
    permute_children = find_flat_permutation(true_topics[child_indices, :], est_topics[child_indices, :])

    for orig_child_index in descendants_by_child_index.keys():
        orig_descendants = np.concatenate([[orig_child_index], descendants_by_child_index[orig_child_index]]).astype('int')
        indirect_child_index = list(child_indices).index(orig_child_index)
        new_child_index = child_indices[permute_children[indirect_child_index]]
        new_descendants = np.concatenate([[new_child_index], descendants_by_child_index[new_child_index]]).astype('int')
        new_paths = [paths[n][1:] for n in new_descendants]
        assert len(new_descendants) == len(orig_descendants), \
            "Length mismatch in descendants: {} (new) vs {} (old)".format(new_descendants, orig_descendants)
        permute_descendants = find_structural_permutation(
            true_topics = true_topics[orig_descendants, :],
            est_topics = est_topics[new_descendants, :],
            paths = new_paths)
        permute_nodes[orig_descendants] = new_descendants[permute_descendants]

    return permute_nodes

def find_flat_permutation(true_topics, est_topics):
    num_topics = true_topics.shape[0]
    orig_indices = list(range(num_topics))
    est_topics = est_topics.copy()
    permute_nodes = []
    for i in range(num_topics):
        topic = true_topics[np.newaxis, i, :]
        matching_index = np.argmin(np.sum(np.abs(est_topics - topic), axis=-1))
        permute_nodes.append(orig_indices[matching_index])
        orig_indices = orig_indices[:matching_index] + orig_indices[matching_index+1:]
        est_topics = np.concatenate((est_topics[:matching_index, :], est_topics[matching_index+1:, :]))
    return np.asarray(permute_nodes, dtype='int')
