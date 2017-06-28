import numpy as np
from collections import Counter, defaultdict
try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
except:
    plt = None
    animation = None
try:
    import networkx as nx
except:
    nx = None

class CRP(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def simulate(self, n_rounds):
        '''Simulate a finite number of rounds of CRP using the
        given concentration parameter.'''

        seating = []
        for i in range(n_rounds):
            next_seat = self.simulate_round(seating)
            seating.append(next_seat)

        return np.array(seating)

    def simulate_round(self, seating):
        if not seating:
            return 0  # First customer sits at table 0
        n_occupied = max(seating) + 1
        distrib = np.zeros(n_occupied + 1)
        for i in range(len(seating)):
            distrib[seating[i]] += 1.0
        distrib[n_occupied] = self.alpha
        distrib /= float(self.alpha + len(seating))
        next_seat = np.random.choice(np.arange(n_occupied + 1, dtype='int'), p = distrib)
        return next_seat

class NCRP(object):
    def __init__(self, alpha):
        self.alpha = alpha
        self.crp = CRP(alpha)

    def simulate_round(self, seating, truncate_level):
        path = []
        for level_index in range(truncate_level):
            seating_at_level = [t[level_index] for t in seating if t[:len(path)] == tuple(path)]
            next_seat = self.crp.simulate_round(seating_at_level)
            path.append(next_seat)
        return tuple(path)

    def simulate(self, n_rounds, truncate_level):
        seating = []
        for i in range(n_rounds):
            next_seat = self.simulate_round(seating, truncate_level)
            seating.append(next_seat)
        return seating

def plot_ncrp_subtree(seating, highlight_last = False, ax = None):
    assert plt is not None, "Could not import matplotlib.pyplot"
    assert nx is not None, "Could not import networkx"

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    if not seating:
        return

    last = seating[-1]

    g = nx.Graph()
    pos = dict()
    labels = dict()
    node_list = []
    node_color = []
    x_by_parent = defaultdict(list)

    truncate_level = len(seating[0])
    while True:
        count = Counter(seating)
        for node_index, node in enumerate(sorted(count.keys())):
            g.add_node(node)
            if node not in x_by_parent:
                x = 2 * node_index
            else:
                x = 0.5 * (np.min(x_by_parent[node]) + np.max(x_by_parent[node]))

            node_list.append(node)
            if highlight_last and last[:len(node)] == node:
                node_color.append('#00ff00')
            else:
                node_color.append('white')

            pos[node] = (x, -truncate_level)
            labels[node] = count[node]
            if len(node) > 0:
                g.add_edge(tuple(node[:-1]), node)
                x_by_parent[tuple(node[:-1])].append(x)

        if truncate_level == 0:
            break
        seating = [s[:-1] for s in seating]
        truncate_level = len(seating[0])

    nx.draw(g, nodelist=node_list, pos=pos, labels=labels, ax=ax, node_color=node_color, node_size=600)

def plot_ncrp_animation(seating):
    t_max = len(seating)

    fig = plt.figure(figsize=(8, 6))

    class AnimManager(object):
        def init_anim(self):
            self.ax = fig.gca()

        def animate(self, i):
            self.ax.clear()
            plot_ncrp_subtree(seating = seating[:i], highlight_last = True, ax = self.ax)
            self.ax.autoscale()

    mgr = AnimManager()

    anim = animation.FuncAnimation(fig,
        func=mgr.animate,
        init_func=mgr.init_anim,
        frames=range(1, t_max+1),
        interval=500,
        repeat_delay=1000)
    return anim

def nice_hist(x, bin_width = 1.0, ax = None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    if "bins" not in kwargs:
        low = bin_width * np.floor(np.min(x) / float(bin_width))
        high = np.max(x) + bin_width*2
        kwargs["bins"] = np.arange(low, high, bin_width)
    return ax.hist(x, **kwargs)
