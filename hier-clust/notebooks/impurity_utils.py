from __future__ import print_function, division

import context
from hier_clust.tree_util import Tree

from collections import OrderedDict, Counter, defaultdict
import scipy.cluster.hierarchy as sch
import numpy as np

def avg_gini_impurity_tree(tree):
    assert "counts" in tree.data
    assert "total_count" in tree.data
    assert "gini_impurity_within_node" in tree.data
    numer = tree.reduce_leaf_data(
        combine = lambda x, y: x + y,
        leaf_func = lambda x: x['gini_impurity_within_node'] * x['total_count'])
    total_count_from_root = tree.data['total_count']
    return numer / float(total_count_from_root)

def avg_gini_impurity_from_assignments(assignments, labels):
    assert len(assignments) == len(labels)

    indices_grouped_by_leaf = defaultdict(list)
    for i, a in assignments:
        indices_grouped_by_leaf[a].append(i)

    numer = 0.0
    for leaf, indices in indices_grouped_by_leaf.items():
        labels_at_leaf = labels[indices]
        count_by_label = Counter(labels_at_leaf)
        leaf_size = len(labels_at_leaf)
        gini = 0.0
        for label, count in count_by_label.items():
            gini += (count * (leaf_size - count)) / float(leaf_size * leaf_size)
        numer += gini * leaf_size
    return numer / float(len(labels))

def gini_impurity_single_node(counts):
    '''Turns a Counter object for a single node into Gini impurity'''
    gini = 0.0
    leaf_size = sum(counts.values())
    for label, count in counts.items():
        gini += (count * (leaf_size - count)) / float(leaf_size * leaf_size)
    return gini

def add_counters_to_tree(tree, labels):
    def helper(node):
        if len(node.children) == 0:
            indices = node.data["orig_indices"]
            node.data["counts"] = Counter(labels[i] for i in indices)
        else:
            counts = Counter()
            for c in node.children:
                helper(c)
                counts += c.data["counts"]
            node.data["counts"] = counts
            
        node.data["total_count"] = len(node.data["orig_indices"])
        node.data["gini_impurity_within_node"] = gini_impurity_single_node(node.data["counts"])

    helper(tree)
    return tree

def convert_linkage_to_tree(link, labels = None):
    def helper(tree, path):
        data = {
            "orig_indices": [],
            "tree_path": path,
            "dist": tree.dist,
        }
        if tree.is_leaf():
            data["orig_indices"] = [tree.get_id()]
            return Tree.leaf(data)
        else:
            left = helper(tree.get_left(), path=path+'L')
            right = helper(tree.get_right(), path=path+'R')
            indices = left.data["orig_indices"] + right.data["orig_indices"]
            data["orig_indices"] = sorted(indices)
            return Tree(
                data = data,
                children = [left, right])
        
    root = sch.to_tree(link)
    root = helper(root, '')
    if labels is not None:
        root = add_counters_to_tree(root, labels)
    return root

def construct_random_tree(indices, path=''):
    if len(indices) == 0:
        raise Exception("Reached leaf with zero elements")
    elif len(indices) == 1:
        return Tree.leaf(data=dict(orig_indices=indices, tree_path=path))
    else:
        num_left = int(len(indices) / 2)
        num_right = len(indices) - num_left
        left_mask = np.array([True for i in xrange(num_left)] + [False for i in xrange(num_right)], dtype='bool')
        np.random.shuffle(left_mask)
        ltree = construct_random_tree(indices = indices[left_mask], path = path+'L')
        rtree = construct_random_tree(indices = indices[~left_mask], path = path+'R')
        data = dict(orig_indices = indices, tree_path = path)
        return Tree(data = data, children = [ltree, rtree])

def random_leaf_impurity(labels):
    ''' Calculate the impurity-vs-depth curve for a tree that
        randomly peels off a single observation at each layer'''
    N = len(labels)
    depths = np.arange(N)
    impurity = []
    unique_labels = np.unique(labels)
    count_by_label = Counter(labels)
    cumulative_count_by_label = Counter()
    for d in depths:
        numer = 0.0
        for k in unique_labels:
            Nk = count_by_label[k]
            ck = cumulative_count_by_label[k]
            numer += (Nk - ck) * (N - d - Nk + ck)
        imp = numer / float((N - d) * N)
        impurity.append(imp)
        cumulative_count_by_label[labels[d]] += 1
    return depths, np.asarray(impurity)
