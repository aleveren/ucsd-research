import numpy as np
import networkx as nx
from collections import namedtuple


def extract(m, threshold = 1e-8, apply_ratio = True):
    if apply_ratio:
        p_node = m.sum(axis = 0)
        m = m / np.outer(p_node, p_node)

    constraints = get_constraints(m, threshold=threshold)

    tree = aho_tree_build(
        nodes = np.arange(m.shape[0]),
        constraints = constraints)

    return tree

def get_ratio_matrix(R):
    p_node = R.sum(axis = 0)
    return R / np.outer(p_node, p_node)

def aho_tree_build(nodes, constraints):
    # Use list of triplet constraints of the form ({a,b},c); ie, LCA(a,b) < LCA(a,c)
    # to build a tree via Aho et al's algorithm
    if isinstance(nodes, set):
        nodes = list(nodes)
    
    if len(nodes) == 1:
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
        T = aho_tree_build(S, C)
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

class Internal(namedtuple("Internal", ["name"])):
    def __str__(self):
        return self.__repr__()
    def __repr__(self):
        return "IN({})".format(self.name)

class TripletConstraint(namedtuple("TripletConstraint", ["i", "j", "k", "strength"])):
    pass

def get_constraints(C, threshold=0, strength_func=None):
    if strength_func is None:
        strength_func = lambda a, b: min(a, b)

    result = []
    for i in range(C.shape[0]):
        for j in range(i):
            for k in range(C.shape[0]):
                if k == i or k == j:
                    continue
                if C[i,j] > C[i,k] + threshold and C[i,j] > C[j,k] + threshold:
                    s = strength_func(C[i,j] - C[i,k], C[i,j] - C[j,k])
                    constraint = TripletConstraint(i, j, k, strength = s)
                    result.append(constraint)

    if len(result) > 0 and isinstance(result[0], TripletConstraint):
        # Sort by descending strength, if available
        result = sorted(result, key = lambda x: -x.strength)

    return result

def test():
    import matplotlib.pyplot as plt
    import sys, os
    sys.path.append(os.path.abspath('..'))
    from utils import bfs_layout, niceprint_graph

    g = aho_tree_build(np.arange(11), [(1,2,3),(2,3,4),(4,5,1),(8,9,6),(9,10,6),(0,4,6)])
    niceprint_graph(g)

    fig, ax = plt.subplots(1, 2)

    nx.draw(g, pos=bfs_layout(g), with_labels=True, ax=ax[0])

    R = example_R()
    g = extract(R)
    print(type(g))
    if g is not None:
        niceprint_graph(g)
    else:
        print("Returned null tree")

    nx.draw(g, pos=bfs_layout(g), with_labels=True, ax=ax[1])

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
