import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from matplotlib import animation
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
        if len(seating) == 0:
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

def plot_ncrp_subtree(seating, highlight_last = False, depth_last = None, figsize = (8, 6), ax = None):
    assert plt is not None, "Could not import matplotlib.pyplot"
    assert nx is not None, "Could not import networkx"

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if len(seating) == 0:
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
                if depth_last == len(node):
                    node_color.append('#ffff00')
                else:
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

def plot_ncrp_animation(path_history, depth_history = None, figsize = (8, 6),
        interval = 500, repeat_delay = 1000):
    fig = plt.figure(figsize=figsize)

    steps = []
    for p_index, p in enumerate(path_history):
        if depth_history is None:
            steps.append((p_index, None))
        else:
            for d in depth_history[p_index]:
                steps.append((p_index, d))

    class AnimManager(object):
        def __init__(self, steps):
            self.steps = steps

        def init_anim(self):
            self.ax = fig.gca()

        def animate(self, i):
            self.ax.clear()
            if depth_history is None:
                p_index, depth_last = self.steps[i]
            else:
                p_index, depth = self.steps[i // 2]
                depth_last = depth if (i % 2 == 0) else None
            plot_ncrp_subtree(
                seating = path_history[:p_index+1],
                highlight_last = True,
                depth_last = depth_last,
                ax = self.ax)
            self.ax.autoscale()

    mgr = AnimManager(steps)

    frames = range(2*len(steps)) if depth_history is not None else range(len(steps))

    anim = animation.FuncAnimation(fig,
        func=mgr.animate,
        init_func=mgr.init_anim,
        frames=frames,
        interval=interval,
        repeat_delay=repeat_delay)
    return anim

def nice_hist(x, bin_width = 1.0, ax = None, **kwargs):
    if ax is None:
        _, ax = plt.subplots()
    if "bins" not in kwargs:
        low = bin_width * np.floor(np.min(x) / float(bin_width))
        high = np.max(x) + bin_width*2
        kwargs["bins"] = np.arange(low, high, bin_width)
    return ax.hist(x, **kwargs)

class GEM(object):
    def __init__(self, pi, m):
        self.pi = pi
        self.m = m
        self.stick_breaking = BetaStickBreaking(m * pi, (1 - m) * pi)

    def draw(self):
        return self.stick_breaking.draw()

class BetaStickBreaking(object):
    '''Represents a distribution over distributions over integers,
    based on a stick-breaking construction that uses beta distributions.'''
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def draw(self):
        return BetaStickBreakingDraw(a = self.a, b = self.b)

class BetaStickBreakingDraw(object):
    '''Represents a lazily-evaluated instance of a single draw from a
    stick-breaking contruction based on beta distributions.
    Note: lazy evaluation is required, because a single evaluation represents
    an infinite distribution over all non-negative ingeters.'''
    def __init__(self, a, b, prepopulate = 10):
        self.a = a
        self.b = b
        self.cached_thetas = []
        self.sum_thetas = 0.0
        for i in range(prepopulate):
            self.extend_thetas()

    def extend_thetas(self):
        remaining = 1.0 - self.sum_thetas
        next_theta = remaining * np.random.beta(self.a, self.b)
        self.cached_thetas.append(next_theta)
        self.sum_thetas += next_theta

    def draw(self):
        '''Draw a random non-negative integer based on the distribution
        defined by this instance.'''
        cumulative = 0.0
        p = np.random.uniform(0.0, 1.0)
        i = 0
        while True:
            if i == len(self.cached_thetas):
                self.extend_thetas()
            cumulative += self.cached_thetas[i]
            if cumulative > p:
                return i
            i += 1

class TopicGenerator(object):
    def __init__(self, eta, gamma, m, pi, vocab):
        self.eta = eta
        self.vocab = vocab
        self.gem = GEM(m = m, pi = pi)
        self.ncrp = NCRP(alpha = gamma)
        self.m = m
        self.pi = pi
        self.gamma = gamma
        self.reset()

    def reset(self):
        self.path_history = []
        self.depth_history_by_document = []
        self.max_depth = 0
        self.depth_distrib_by_document = []
        self.cached_topics = dict()

    def lookup_topic(self, path):
        if path not in self.cached_topics:
            if np.iterable(self.eta):
                i = min(len(path), len(self.eta) - 1)
                current_eta = self.eta[i]
            else:
                current_eta = self.eta
            topic = np.random.dirichlet(current_eta * np.ones(len(self.vocab)))
            self.cached_topics[path] = topic
        return self.cached_topics[path]

    def draw_word(self, path):
        topic = self.lookup_topic(path)
        return np.random.choice(self.vocab, p = topic)

    def extend_paths(self, length):
        '''Make sure each path in path history has been extended to the proper length'''
        if len(self.path_history) == 0:
            return
        while len(self.path_history[0]) < length:
            target_len = len(self.path_history[0]) + 1
            for i, h in enumerate(self.path_history):
                seating_at_level = [t[target_len - 1]
                    for t in self.path_history
                    if len(t) >= target_len and t[:target_len-1] == h]
                next_seat = self.ncrp.crp.simulate_round(seating = seating_at_level)
                self.path_history[i] = h + (next_seat,)

    def draw_document(self, doc_length):
        doc = []
        thetas = self.gem.draw()
        self.depth_distrib_by_document.append(thetas)
        depth_history = []
        for word_index in range(doc_length):
            depth_history.append(thetas.draw())

        self.max_depth = max(self.max_depth, np.max(depth_history))
        self.depth_history_by_document.append(depth_history)

        self.extend_paths(length = self.max_depth)
        path = self.ncrp.simulate_round(
            seating = self.path_history,
            truncate_level = self.max_depth)
        self.path_history.append(path)

        for word_index in range(doc_length):
            level = depth_history[word_index]
            word = self.draw_word(path[:level])
            doc.append(word)
        return doc

    def draw_corpus(self, n_documents, doc_length):
        if not np.iterable(doc_length):
            doc_length = [doc_length for i in range(n_documents)]
        corpus = []
        for doc_index, current_length in enumerate(doc_length):
            doc = self.draw_document(current_length)
            corpus.append(doc)
        return corpus
