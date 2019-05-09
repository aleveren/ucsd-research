from __future__ import division

import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import itertools

def get_alpha(num_children, pexit = None, scale = 1.0, add_exit_edge = True):
    if add_exit_edge:
        if pexit is None:
            pexit = 1.0 / (num_children + 1)
        pother = (1 - pexit) / float(num_children)
        result = pother * np.ones(num_children + 1)
        result[0] = pexit
        result *= scale * (num_children + 1)
    else:
        result = scale * np.ones(num_children)
    return result

def gen_all_paths(g, prefix=None):
    if prefix is None:
        prefix = (g.graph["root"],)
    yield prefix
    for c in g.neighbors(prefix[-1]):
        new_prefix = prefix + (c,)
        for p in gen_all_paths(g, new_prefix):
            yield p

def compute_combo_probability(g, path_combo, alpha_func = None, alpha_mode = None):
    if alpha_func is None:
        alpha_func = get_alpha

    # hPAM2 calculations
    def num_children(node):
        return len(list(g.neighbors(node)))
    def gen_transitions(r):
        assert len(r) > 0
        for i in range(len(r) - 1):
            yield (r[i], r[i + 1])
        if num_children(r[-1]) > 0:
            # Include any non-trivial ending transitions (ie, exiting the DAG before reaching a sink node)
            yield (r[-1], None)
    transitions = defaultdict(Counter)

    for path in path_combo:
        for src, dest in gen_transitions(path):
            transitions[src][dest] += 1
    
    result = 1
    for src, dest_counter in transitions.items():
        if alpha_mode is None:
            nc = num_children(src)
            alpha = alpha_func(nc)
        elif alpha_mode == "node_id":
            alpha = alpha_func(src)
        else:
            raise ValueError("Unrecognized alpha_mode: {}".format(alpha_mode))
        alpha_exit, alpha_child = alpha[0], alpha[1]
        denom = 1
        for i in range(sum(dest_counter.values())):
            denom *= alpha.sum() + i
        numer = 1
        for dest, count in dest_counter.items():
            a = alpha_exit if dest is None else alpha_child
            for i in range(count):
                numer *= a + i
        result *= numer / denom

    return result

def compute_combo_tensor(g, combo_size = 2, alpha_func = None, alpha_mode = None):
    num_nodes = len(g.nodes())
    result = np.zeros(tuple(num_nodes for i in range(combo_size)))
    for combo in itertools.product(gen_all_paths(g), repeat = combo_size):
        coords = tuple(c[-1] for c in combo)
        result[coords] += compute_combo_probability(g, combo, alpha_func = alpha_func, alpha_mode = alpha_mode)
    return result

def compute_combo_tensor_pam(g, combo_size = 2, alpha = 1.0, return_leaf_paths = False, ndarray_kwargs = None):
    if ndarray_kwargs is None:
        ndarray_kwargs = dict()
    leaf_paths = []
    for node, d in dict(g.out_degree).items():
        if d == 0:
            path = nx.shortest_path(g, g.graph["root"], node)
            leaf_paths.append(path)
    num_leaves = len(leaf_paths)
    result = np.zeros(tuple(num_leaves for i in range(combo_size)), **ndarray_kwargs)
    for combo in itertools.product(range(len(leaf_paths)), repeat = combo_size):
        path_combo = [leaf_paths[i] for i in combo]
        result[combo] += compute_combo_probability_pam(
            g, path_combo, alpha = alpha)
    if return_leaf_paths:
        return result, leaf_paths
    return result

def compute_combo_probability_pam(g, path_combo, alpha = 1.0):
    def gen_transitions(r):
        assert len(r) > 0
        for i in range(len(r) - 1):
            yield (r[i], r[i + 1])
    transitions = defaultdict(Counter)
    alpha = AlphaCalc.create(alpha)

    for path in path_combo:
        for src, dest in gen_transitions(path):
            transitions[src][dest] += 1

    result = 1
    for src, dest_counter in transitions.items():
        denom = 1
        sum_alphas = np.sum([alpha.calc(node_id = nbr) for nbr in g.neighbors(src)])
        for i in range(sum(dest_counter.values())):
            denom *= sum_alphas + i
        numer = 1
        for dest, count in dest_counter.items():
            for i in range(count):
                numer *= alpha.calc(node_id = dest) + i
        result *= numer / denom

    return result

class AlphaCalc(object):
    def calc(self, node_id):
        raise NotImplementedError("'calc' not implemented")

    @staticmethod
    def create(x):
        if isinstance(x, AlphaCalc):
            return x
        elif np.isscalar(x):
            return ConstAlphaCalc(x)
        elif isinstance(x, dict):
            return IndividualNodeAlphaCalc(x)
        else:
            raise ValueError("Unrecognized type for AlphaCalc.create")

class ConstAlphaCalc(AlphaCalc):
    def __init__(self, val):
        self.val = val

    def calc(self, node_id):
        return self.val

class IndividualNodeAlphaCalc(AlphaCalc):
    def __init__(self, values):
        self.values = values

    def calc(self, node_id):
        return self.values[node_id]

def tensor_to_matrix(T):
    if np.ndim(T) < 2:
        T = np.atleast_2d(T)
    while np.ndim(T) > 2:
        if np.ndim(T) == 3:
            axis = 0 if T.shape[0] * T.shape[1] < T.shape[1] * T.shape[2] else 1
        else:
            axis = np.argmin(T.shape)
        T = np.concatenate(T, axis=axis)
    return T
