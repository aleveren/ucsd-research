import numpy as np
import io
from scipy.special import digamma

def load_vocab(filename):
    vocab = []
    with io.open(filename, mode='r', encoding='utf8') as f:
        for line in f:
            vocab.append(line.rstrip())
    return vocab

def softmax(X, axis):
    X = np.asarray(X)
    eX = np.exp(X)
    return eX / eX.sum(axis = axis, keepdims = True)

def expectation_log_dirichlet(nu, axis):
    return digamma(nu) - digamma(nu.sum(axis = axis, keepdims = True))

def explore_branching_factors(factors):
    return list(_generator_explore_branching_factors(factors, prefix = ()))

def without_diag(X):
    result = X.copy()
    for i in range(min(X.shape)):
        coord = tuple(i for j in range(np.ndim(X)))
        result[coord] = 0
    return result

def _generator_explore_branching_factors(factors, prefix):
    yield prefix
    if len(factors) > 0:
        first = factors[0]
        rest = factors[1:]
        for i in range(first):
            new_prefix = prefix + (i,)
            for path in _generator_explore_branching_factors(rest, new_prefix):
                yield path

def niceprint_str(X, precision = 4, **kwargs):
    fmt = "{{:.{}f}}".format(precision)
    formatter = dict(float = lambda x: fmt.format(x))
    a2s_kwargs = dict(max_line_width=10000, threshold=10000, formatter=formatter)
    a2s_kwargs.update(kwargs)
    result = np.array2string(X, **a2s_kwargs)
    return result

def niceprint(*args, **kwargs):
    print(niceprint_str(*args, **kwargs))

def nicesubplots(rows, cols, scale=4, **kwargs):
    import matplotlib.pyplot as plt
    dims = np.array([rows, cols])
    return plt.subplots(*dims, figsize = dims[::-1] * np.atleast_1d(scale), **kwargs)

def _bfs_layout_helper(tree, source, spacing, center):
    pos = dict()
    results_by_child = []
    try:
        nbr_list = sorted(tree.neighbors(source))
    except TypeError:
        nbr_list = tree.neighbors(source)
    for n in nbr_list:
        results_by_child.append(_bfs_layout_helper(tree, n, spacing, center))
    pos[source] = center
    width = 0.0
    for i, (subtree_pos, subtree_width) in enumerate(results_by_child):
        if i > 0:
            width += spacing[0]
        width += subtree_width
    shiftx = -width / 2.0
    for subtree_pos, subtree_width in results_by_child:
        shiftx += subtree_width / 2.0
        for k, (x, y) in subtree_pos.items():
            pos[k] = (x + shiftx, y - spacing[1])
        shiftx += subtree_width / 2.0 + spacing[0]
    return pos, width

def bfs_layout(G, sources=None, spacing=(1.0, 1.0), center=(0.0, 0.0)):
    import networkx as nx
    pos = dict()
    if G.number_of_nodes() == 0:
        return pos
    if sources is None:
        if "root" in G.graph:
            sources = [G.graph["root"]]
        else:
            d = G.in_degree()
            sources = [n for n in G.nodes() if d[n] == 0]
    if G.is_directed():
        components = nx.algorithms.weakly_connected_component_subgraphs(G)
    else:
        components = nx.algorithms.connected_component_subgraphs(G)
    cumulative_width = 0
    for comp in components:
        current_source = None
        for s in sources:
            if s in comp.nodes():
                current_source = s
                break
        if current_source is None:
            current_source = list(comp.nodes())[0]
        tree = nx.bfs_tree(G, current_source)
        current_pos, width = _bfs_layout_helper(tree, current_source, spacing, center)
        for k in current_pos.keys():
            current_pos[k] = (current_pos[k][0] + cumulative_width, current_pos[k][1])
        pos.update(current_pos)
        cumulative_width += width + spacing[0]
    return pos

def niceprint_str_graph(g):
    result = "nodes: {}".format(list(g.nodes()))
    if "root" in g.graph:
        result += ", root: {}".format(g.graph["root"])
    result += ", edges: [{}]".format(", ".join(["{} <-> {}".format(a, b) for a, b in g.edges()]))
    return result

def niceprint_graph(g):
    print(niceprint_str_graph(g))

def nice_tree_plot(t, ax = None, **kwargs):
    import matplotlib.pyplot as plt
    import networkx as nx
    if ax is None:
        _, ax = plt.subplots()
    if t is None:
        return
    draw_kwargs = dict(with_labels=True, pos=bfs_layout(t))
    draw_kwargs.update(kwargs)
    nx.draw(t, ax = ax, **draw_kwargs)

def invert_permutation(p):
    p = np.asarray(p)
    result = np.empty(p.shape, dtype='int')
    result[p] = np.arange(p.size)
    return result

def permute_square(X, perm):
    perm = np.reshape(perm, (-1, 1))
    return X[perm, perm.T]
