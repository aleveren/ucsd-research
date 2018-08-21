import numpy as np
import networkx as nx
from collections import namedtuple


def extract(m, threshold = 0, apply_ratio = True):
    if apply_ratio:
        m = get_ratio_matrix(m)

    constraints = get_constraints(m, threshold = threshold)

    tree = build_tree(
        nodes = list(range(m.shape[0])),
        constraints = constraints)

    if tree is None:
        # Binary search to find largest set of constraints that
        # yields a non-null tree
        lo = 0
        hi = len(constraints)
        mid = int((lo + hi) / 2)
        prev_mid = None

        while mid != prev_mid:
            mid_tree = build_tree(
                nodes = list(range(m.shape[0])),
                constraints = constraints[:mid])

            if mid_tree is None:
                hi = mid
            else:
                lo = mid
                tree = mid_tree
            prev_mid = mid
            mid = int((lo + hi) / 2)

    return tree

def get_ratio_matrix(R):
    p_node = R.sum(axis = 0)
    return R / np.outer(p_node, p_node)

def build_tree(nodes, constraints):
    # Use list of triplet constraints of the form ({a,b},c); ie, LCA(a,b) < LCA(a,c)
    # to build a tree via Aho et al's algorithm
    if isinstance(nodes, set):
        nodes = list(nodes)

    if len(nodes) == 0:
        raise ValueError("Empty set of nodes")
    elif len(nodes) == 1:
        result = nx.DiGraph()
        result.add_node(nodes[0])
        result.graph["root"] = nodes[0]
        return result
    
    graph = nx.Graph()
    for n in nodes:
        graph.add_node(n)
    for c in constraints:
        graph.add_edge(c[0], c[1])
    components = list(nx.algorithms.connected_components(graph))
    
    if len(components) == 1:
        return None

    subtrees = []
    cumulative_internal = 0
    for S in components:
        C = [c for c in constraints if c[0] in S and c[1] in S and c[2] in S]
        T = build_tree(S, C)
        if T is None:
            return None
        subtrees.append((T, cumulative_internal))

        num_internal = len(list(T.nodes())) - len(S)
        cumulative_internal += num_internal

    new_root = Internal(cumulative_internal)
    result = nx.DiGraph()
    result.add_node(new_root)
    result.graph["root"] = new_root
    for T, offset in subtrees:
        mapper = dict()
        for n in T.nodes():
            if isinstance(n, Internal):
                nprime = Internal(n.name + offset)
            else:
                nprime = n
            mapper[n] = nprime
            result.add_node(nprime)
        for a, b in T.edges():
            result.add_edge(mapper[a], mapper[b])
        result.add_edge(new_root, mapper[T.graph["root"]])
    return result

def tree_satisfies_constraint(tree, c):
    root = tree.graph["root"]
    lca_ij = nx.lowest_common_ancestor(tree, c[0], c[1])
    lca_ijk = nx.lowest_common_ancestor(tree, lca_ij, c[2])
    path_lca_ij = nx.algorithms.shortest_path(tree, root, lca_ij)
    return lca_ijk in path_lca_ij[:-1]

class Internal(namedtuple("Internal", ["name"])):
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "IN({})".format(self.name)

class TripletConstraint(namedtuple("TripletConstraint", ["i", "j", "k", "strength"])):
    pass

def get_constraints(C, threshold=0):
    result = []
    for i in range(C.shape[0]):
        for j in range(i):
            for k in range(C.shape[0]):
                if k == i or k == j:
                    continue
                if C[i,j] > C[i,k] + threshold and C[i,j] > C[j,k] + threshold:
                    s = min(C[i,j] - C[i,k], C[i,j] - C[j,k])
                    constraint = TripletConstraint(i, j, k, strength = s)
                    result.append(constraint)

    # Sort by descending strength
    result = sorted(result, key = lambda x: -x.strength)

    return result

def test():
    import matplotlib.pyplot as plt
    import sys, os
    sys.path.append(os.path.abspath('..'))
    from utils import bfs_layout, niceprint_graph

    g = build_tree(list(range(11)), [(1,2,3),(2,3,4),(4,5,1),(8,9,6),(9,10,6),(0,4,6)])
    niceprint_graph(g)

    #fig, ax = plt.subplots(1, 2)
    fig, ax = plt.subplots()

    #nx.draw(g, pos=bfs_layout(g), with_labels=True, ax=ax[0])

    R = example_R()
    g = extract(R)
    print(type(g))
    if g is not None:
        niceprint_graph(g)
    else:
        print("Returned null tree")

    #nx.draw(g, pos=bfs_layout(g), with_labels=True, ax=ax[1])
    nx.draw(g, pos=bfs_layout(g), with_labels=True, ax=ax)

    plt.show()

def example_R():
    from compute_pam import compute_combo_tensor, get_alpha
    from functools import partial
    import sys, os
    sys.path.append(os.path.abspath('..'))
    from example_graphs import make_tree
    tree = make_tree([3,3,3])
    R = compute_combo_tensor(tree, combo_size = 2, alpha_func=partial(get_alpha, pexit=0.3, scale=1.0))
    return R


if __name__ == "__main__":
    # Run this test via `python3 -m tree_extraction.Aho`
    test()
