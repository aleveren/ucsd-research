import numpy as np
from collections import namedtuple

'''
Utilities for converting a tree into a "linkage" matrix of the form required
for `scipy.cluster.hierarchy.dendrogram`

See the following documentation pages:
* http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
* http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage

The linkage matrix has one row for each internal node and 4 columns.
The columns at indices 0 and 1 represent the cluster id's being joined
(cluster id's < N represent individual data points).  Cluster id n+i is defined
in the ith row of the matrix.
The column at index 2 is the "distance" between the two clusters.
The column at index 3 is the total number of observations in the joined cluster.
'''

class Leaf(namedtuple("Leaf", ["node_id"])):
    def is_leaf(self): return True
class Split(namedtuple("Split",
        ["node_id", "index0", "index1", "node_height", "size"])):
    def is_leaf(self): return False

def get_linkage(tree):
    num_obs = len(tree.data["orig_indices"])
    linkage = get_linkage_helper(tree, num_obs, tree.depth())
    result = []
    for item in linkage:
        assert not item.is_leaf()
        result.append([item.index0, item.index1, item.node_height, item.size])
    return np.array(result, dtype='float')
    
def get_linkage_helper(tree, next_node_id, tot_depth):
    indices = tree.data["orig_indices"]
    path = tree.data["tree_path"]
    
    if tree.depth() == 0:
        assert len(indices) == 1
        return [Leaf(indices[0])]
    if tree.depth() == 1:
        assert len(indices) == 2
        return [Split(next_node_id, indices[0], indices[1],
            tot_depth - len(path), len(indices))]

    link_left = get_linkage_helper(tree.children[0], next_node_id, tot_depth)
    node_id_left = link_left[-1].node_id
    if link_left[-1].is_leaf():
        link_left = link_left[:-1]
    next_node_id += len(link_left)

    link_right = get_linkage_helper(tree.children[1], next_node_id, tot_depth)
    node_id_right = link_right[-1].node_id
    if link_right[-1].is_leaf():
        link_right = link_right[:-1]
    next_node_id += len(link_right)

    entry = Split(next_node_id, node_id_left, node_id_right,
        tot_depth - len(path), len(indices))    
    return link_left + link_right + [entry]