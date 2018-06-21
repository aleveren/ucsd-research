import numpy as np
import networkx as nx

def extract_tree_helper(X, root_index, branching_factors, indices):
    g = nx.DiGraph()
    root = indices[root_index]
    g.add_node(root)
    if len(branching_factors) == 0:
        return g

    num_children = branching_factors[0]
    remaining_branching_factors = branching_factors[1:]
    descendants_per_child = int(np.prod(remaining_branching_factors))

    assert len(indices) >= num_children
    assert X.shape[0] == X.shape[1]
    assert X.shape[0] == len(indices), "{} vs {}".format(X.shape[0], len(indices))
    if len(indices) == num_children:
        for i in indices:
            g.add_edge(root, i)
        return g

    mask_child_candidates = np.ones(indices.shape, dtype='bool')
    mask_child_candidates[root_index] = False
    children_indirect = np.argpartition(X[root_index, mask_child_candidates], -num_children)[-num_children:]
    children = np.where(mask_child_candidates)[0][children_indirect]

    mask_candidates = np.ones(indices.shape, dtype='bool')
    mask_candidates[root_index] = False
    mask_candidates[children] = False

    for c in children:
        to_recurse = [c]
        indirect = np.argpartition(X[c, mask_candidates], -descendants_per_child)[-descendants_per_child:]
        to_recurse.extend(np.where(mask_candidates)[0][indirect])
        to_recurse_col = np.reshape(to_recurse, (-1, 1))
        # Compute subtree and add to graph
        Xsub = X[to_recurse_col, to_recurse_col.T]
        remaining_indices = indices[to_recurse]
        subtree = extract_tree_helper(Xsub, 0, remaining_branching_factors, remaining_indices)
        g = nx.compose(g, subtree)
        g.add_edge(root, indices[c])
        # Update candidates
        mask_candidates[to_recurse] = False

    return g

def extract(X, branching_factors):
    indices = np.arange(X.shape[0])
    p_node = X.sum(axis = 0)
    root = np.argmax(p_node)
    g = extract_tree_helper(X, root, branching_factors, indices)
    g.graph["root"] = root
    return g
