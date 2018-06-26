import numpy as np
import networkx as nx

def extract(X, threshold, apply_cond = True):
    node_ids = np.arange(X.shape[0])
    p_node = X.sum(axis = 0)
    root_index = np.argmax(p_node)

    #X = without_diag(X.copy())
    X = X.copy()
    if apply_cond:
        X /= p_node

    g = extract_tree_helper(X, root_index, threshold, node_ids)
    g.graph["root"] = root_index
    return g

def extract_tree_helper(X, root_index, threshold, node_ids):
    g = nx.DiGraph()
    root = node_ids[root_index]
    g.add_node(root)

    if len(node_ids) == 1:
        return g

    # Identify nodes for which P(node | root) >= (1-threshold) * max[n in node][P(n | root)]
    mask_child_candidates = np.ones(node_ids.shape, dtype='bool')
    mask_child_candidates[root_index] = False

    candidate_probs = X[mask_child_candidates, root_index]

    children_indirect = np.where(candidate_probs >= (1 - threshold) * np.max(candidate_probs))[0]
    children_indices = np.where(mask_child_candidates)[0][children_indirect]
    children_ids = node_ids[children_indices]

    # For each child node, find descendants of that node
    mask_descendant_candidates = mask_child_candidates.copy()
    mask_descendant_candidates[children_indices] = False

    children_indices_to_descendant_indices = dict()
    for ci in children_indices:
        children_indices_to_descendant_indices[ci] = [ci]
    for i in range(len(node_ids)):
        if not mask_descendant_candidates[i]:
            continue
        parent_indirect = np.argmax(X[children_indices, i])
        parent_index = children_indices[parent_indirect]
        children_indices_to_descendant_indices[parent_index].append(i)

    # Recursively call tree extraction on submatrices
    for child_index, desc_indices in children_indices_to_descendant_indices.items():
        assert desc_indices[0] == child_index
        desc_indices = np.atleast_2d(desc_indices)
        subtree = extract_tree_helper(X[desc_indices.T, desc_indices], root_index = 0, threshold = threshold, node_ids = node_ids[desc_indices.flatten()])
        g = nx.compose(g, subtree)
        g.add_edge(root, node_ids[child_index])

    return g
