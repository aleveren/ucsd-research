import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda *args, **kwargs: args[0]
from functools import partial

from simple_hierarchical_topic_model import explore_branching_factors
from compute_pam import get_alpha

class SHTMSampler(object):
    '''Generate a simulated dataset'''
    def __init__(
            self,
            branching_factors,
            num_docs,
            doc_length,
            topic_sharpness,
            alpha_leaves,
            alpha_depths,
            vocab_size = None,
            overlap = None,
            heavy_words_per_topic = 2,
            heavy_indices = None):
        self.branching_factors = branching_factors
        self.nodes = explore_branching_factors(self.branching_factors)
        self.num_nodes = len(self.nodes)
        self.max_depth = max([len(x) for x in self.nodes])
        self.num_leaves = len([x for x in self.nodes if len(x) == self.max_depth])
        self.num_depths = len(np.unique([len(x) for x in self.nodes]))
        self.vocab_size = vocab_size
        self.overlap = "none" if overlap is None else overlap.lower()
        self.heavy_words_per_topic = heavy_words_per_topic
        self.heavy_indices = heavy_indices
        self.num_docs = num_docs
        self.doc_length = doc_length
        self.topic_sharpness = topic_sharpness
        self.alpha_leaves = np.broadcast_to(alpha_leaves, (self.num_leaves,)).astype('float')
        self.alpha_depths = np.broadcast_to(alpha_depths, (self.num_depths,)).astype('float')
        self.init_topics_and_vocab()

    def init_topics_and_vocab(self):
        if self.heavy_indices is None:
            self.heavy_indices = list(self.get_heavy_indices(overlap = self.overlap))

        min_vocab_size = 2
        for h in self.heavy_indices:
            min_vocab_size = max(min_vocab_size, 1 + np.max(h))

        if self.vocab_size is None:
            self.vocab_size = min_vocab_size + 4
        assert self.vocab_size >= min_vocab_size, \
            "Vocab size must be at least {}".format(min_vocab_size)
        self.vocab = ["w{}".format(i) for i in range(self.vocab_size)]

        self.heavy_indicator = np.zeros((self.num_nodes, self.vocab_size))
        for node_index in range(self.num_nodes):
            self.heavy_indicator[node_index, self.heavy_indices[node_index]] = 1

        self.topics_by_index = self.heavy_indicator * (self.topic_sharpness - 1.0) + 1.0
        self.topics_by_index /= self.topics_by_index.sum(axis = -1, keepdims = True)

        self.topics_by_path = dict()
        self.leaves = []
        for node_index, path in enumerate(self.nodes):
            self.topics_by_path[path] = self.topics_by_index[node_index, :]
            if len(path) == self.max_depth:
                self.leaves.append(path)

    def get_heavy_indices(self, overlap):
        if overlap == "none":
            for node_index in range(self.num_nodes):
                lo = node_index * self.heavy_words_per_topic
                hi = lo + self.heavy_words_per_topic
                yield np.arange(lo, hi)
        elif overlap == "full":
            prev = np.arange(self.heavy_words_per_topic)
            yield prev
            for node_index in range(1, self.num_nodes):
                current = prev.copy()
                # Find the first "moveable" position
                i = 0
                while i < len(current) - 1 and current[i] + 1 >= current[i + 1]:
                    i += 1
                if i > 0:
                    # Reset leftmost indices to far left
                    current[:i] = np.arange(i)
                # Advance the index at the first "moveable" position
                current[i] += 1
                yield current
                prev = current
        else:
            # TODO: support "siblings" mode (only sibling topics can have major overlap)
            raise ValueError("Overlap '{}' not supported".format(overlap))

    def generate(self):
        self.docs = []
        self.docs_aux = []

        for i in range(self.num_docs):
            leaf_distrib = np.random.dirichlet(self.alpha_leaves)
            depth_distrib = np.random.dirichlet(self.alpha_depths)
            node_distrib = self.get_node_distrib(leaf_distrib = leaf_distrib, depth_distrib = depth_distrib)
            current_node_indices = np.random.choice(len(self.nodes), size = self.doc_length, p = node_distrib)
            current_doc = []
            for j in range(self.doc_length):
                topic = self.topics_by_index[current_node_indices[j]]
                word_index = np.random.choice(self.vocab_size, p = topic)
                current_doc.append(self.vocab[word_index])
            current_doc = " ".join(current_doc)
            self.docs.append(current_doc)
            self.docs_aux.append({
                "doc": current_doc,
                "leaf_distrib": leaf_distrib,
                "depth_distrib": depth_distrib,
                "node_distrib": node_distrib,
                "node_indices_by_word_slot": current_node_indices,
            })

        return self.docs

    def get_node_distrib(self, leaf_distrib, depth_distrib):
        result = np.zeros(self.num_nodes)
        for node_index, path in enumerate(self.nodes):
            prob_depth = depth_distrib[len(path)]
            prob_descendants = 0.0
            for leaf_index, leaf in enumerate(self.leaves):
                if leaf[:len(path)] == path:
                    prob_descendants += leaf_distrib[leaf_index]
            result[node_index] = prob_depth * prob_descendants
        return result

class PAMSampler(object):
    def __init__(self, g, num_docs, words_per_doc, vocab_size,
            topic_dirichlet = 1.0,
            topic_func = None,
            alpha_func = None):
        self.g = g
        self.num_docs = num_docs
        self.words_per_doc = words_per_doc
        self.vocab_size = vocab_size
        self.topic_dirichlet = np.broadcast_to(topic_dirichlet, (self.vocab_size,)).astype('float')
        self.topic_func = topic_func
        if alpha_func is None:
            alpha_func = partial(get_alpha, add_exit_edge = False)
        self.alpha_func = alpha_func

    def sample(self):
        self.thetas_by_doc = []
        self.docs = []
        self.doc_nodes = []
        self.alphas = dict()
        outdegree = self.g.out_degree()
        # Sample topics
        self.topics = dict()
        for node in self.g.nodes():
            if outdegree[node] == 0:
                if self.topic_func is not None:
                    self.topics[node] = self.topic_func(node)
                else:
                    self.topics[node] = np.random.dirichlet(self.topic_dirichlet)
            else:
                self.alphas[node] = self.alpha_func(outdegree[node])
        # Sample documents
        for i in tqdm(range(self.num_docs)):
            thetas = dict()
            for node in self.g.nodes():
                if outdegree[node] > 0:
                    thetas[node] = np.random.dirichlet(self.alphas[node])
            self.thetas_by_doc.append(thetas)
            current_doc = []
            current_doc_nodes = []
            for j in range(self.words_per_doc):
                node = self.sample_leaf(thetas)
                current_doc_nodes.append(node)
                word_index = np.random.choice(self.vocab_size, p=self.topics[node])
                current_doc.append(word_index)
            self.docs.append(current_doc)
            self.doc_nodes.append(current_doc_nodes)
        return self

    def sample_leaf(self, thetas):
        current = self.g.graph["root"]
        options = list(self.g.successors(current))
        while len(options) > 0:
            current = options[np.random.choice(len(options), p=thetas[current])]
            options = list(self.g.successors(current))
        return current

class HPAM1Sampler(object):
    def __init__(self, g, num_docs, words_per_doc, vocab_size,
            topic_dirichlet = 1.0,
            topic_func = None,
            alpha_func = None):
        self.g = g
        self.num_docs = num_docs
        self.words_per_doc = words_per_doc
        self.vocab_size = vocab_size
        self.topic_dirichlet = np.broadcast_to(topic_dirichlet, (self.vocab_size,)).astype('float')
        self.topic_func = topic_func
        if alpha_func is None:
            alpha_func = partial(get_alpha, add_exit_edge = False)
        self.alpha_func = alpha_func

    def sample(self):
        self.thetas_by_doc = []
        self.docs = []
        self.doc_paths = []
        self.doc_nodes = []
        self.alphas = dict()
        outdegree = self.g.out_degree()
        # Sample topics
        self.topics = dict()
        for node in self.g.nodes():
            if self.topic_func is not None:
                self.topics[node] = self.topic_func(node)
            else:
                self.topics[node] = np.random.dirichlet(self.topic_dirichlet)
            if outdegree[node] > 0:
                self.alphas[node] = self.alpha_func(outdegree[node])
        # Sample documents
        for i in tqdm(range(self.num_docs)):
            thetas = dict()
            for node in self.g.nodes():
                if outdegree[node] > 0:
                    thetas[node] = np.random.dirichlet(self.alphas[node])
            self.thetas_by_doc.append(thetas)
            current_doc = []
            current_doc_paths = []
            current_doc_nodes = []
            for j in range(self.words_per_doc):
                path = self.sample_leaf_path(thetas)
                current_doc_paths.append(path)
                # NOTE: assuming zeta_{path} is the same for all paths
                depth_distrib = np.ones(len(path)) / float(len(path))
                node = np.random.choice(path, p=depth_distrib)
                current_doc_nodes.append(node)
                word_index = np.random.choice(self.vocab_size, p=self.topics[node])
                current_doc.append(word_index)
            self.docs.append(current_doc)
            self.doc_paths.append(current_doc_paths)
            self.doc_nodes.append(current_doc_nodes)
        return self

    def sample_leaf_path(self, thetas):
        path = []
        current = self.g.graph["root"]
        options = list(self.g.successors(current))
        path.append(current)
        while len(options) > 0:
            current = options[np.random.choice(len(options), p=thetas[current])]
            options = list(self.g.successors(current))
            path.append(current)
        return path

class HPAM2Sampler(object):
    def __init__(self, g, num_docs, words_per_doc, vocab_size,
            topic_dirichlet = 1.0,
            topic_func = None,
            alpha_func = None):
        self.g = g
        self.num_docs = num_docs
        self.words_per_doc = words_per_doc
        self.vocab_size = vocab_size
        self.topic_dirichlet = np.broadcast_to(topic_dirichlet, (self.vocab_size,)).astype('float')
        self.topic_func = topic_func
        if alpha_func is None:
            alpha_func = partial(get_alpha, add_exit_edge = True)
        self.alpha_func = alpha_func

    def sample(self):
        self.thetas_by_doc = []
        self.docs = []
        self.doc_nodes = []
        self.alphas = dict()
        outdegree = self.g.out_degree()
        # Sample topics
        self.topics = dict()
        for node in self.g.nodes():
            if self.topic_func is not None:
                self.topics[node] = self.topic_func(node)
            else:
                self.topics[node] = np.random.dirichlet(self.topic_dirichlet)
            if outdegree[node] > 0:
                self.alphas[node] = self.alpha_func(outdegree[node])
        # Sample documents
        for i in tqdm(range(self.num_docs)):
            thetas = dict()
            for node in self.g.nodes():
                if outdegree[node] > 0:
                    thetas[node] = np.random.dirichlet(self.alphas[node])
            self.thetas_by_doc.append(thetas)
            current_doc = []
            current_doc_nodes = []
            for j in range(self.words_per_doc):
                node = self.sample_node(thetas)
                current_doc_nodes.append(node)
                word_index = np.random.choice(self.vocab_size, p=self.topics[node])
                current_doc.append(word_index)
            self.docs.append(current_doc)
            self.doc_nodes.append(current_doc_nodes)
        return self

    def sample_node(self, thetas):
        current = self.g.graph["root"]
        options = list(self.g.successors(current))
        while len(options) > 0:
            assert len(thetas[current] == len(options) + 1)
            choice = np.random.choice(len(thetas[current]), p=thetas[current])
            if choice == 0:
                return current
            current = options[choice - 1]
            options = list(self.g.successors(current))
        return current

def topics_griffiths_steyvers(num_topics, dimension = None, topic_sharpness = 20.0):
    if dimension is None:
        dimension = int(np.ceil(num_topics / 2.0))
    assert num_topics <= dimension * 2, \
        "Too many topics ({}) for dimension {}".format(num_topics, dimension)
    topics = []
    for i in range(dimension):
        topic = np.ones((dimension, dimension))
        topic[i, :] *= topic_sharpness
        topic /= topic.sum()
        topics.append(topic.flatten().copy())
        topics.append(topic.transpose().flatten().copy())
    topics = np.array(topics[:num_topics])
    assert topics.shape == (num_topics, dimension * dimension)
    return topics

class GriffithsSteyversSampler(object):
    '''
    Samples a corpus of documents according to the square-image scheme of
    Griffiths & Steyvers (2004), "Finding Scientific Topics"
    '''
    def __init__(self, num_docs, words_per_doc, dimension = 5, topic_sharpness = 20, alpha = 1.0):
        self.num_docs = num_docs
        self.words_per_doc = words_per_doc
        self.dimension = dimension
        self.vocab_size = dimension * dimension
        self.num_topics = dimension * 2
        self.alpha = np.broadcast_to(alpha, (self.num_topics,))
        self.topics = topics_griffiths_steyvers(num_topics = self.num_topics, dimension = dimension)

    def sample(self):
        from scipy.sparse import dok_matrix, csr_matrix, csc_matrix
        from collections import Counter

        self.doc_topic_mixtures = []
        self.doc_topic_indicators = []
        self.docs = []

        self.data = dok_matrix((self.num_docs, self.vocab_size), dtype='int')  # TODO: use this, or transpose??

        counters = []

        for doc_index in tqdm(range(self.num_docs)):
            topic_mixture = np.random.dirichlet(self.alpha)
            self.doc_topic_mixtures.append(topic_mixture)

            topic_indicators = np.random.choice(
                np.arange(self.num_topics),
                size = self.words_per_doc, p = topic_mixture)
            self.doc_topic_indicators.append(topic_indicators)

            current_doc = []
            current_counter = Counter()
            for word_slot_index in range(self.words_per_doc):
                w = np.random.choice(
                    np.arange(self.vocab_size),
                    p = self.topics[topic_indicators[word_slot_index]])
                current_doc.append(w)
                current_counter[w] += 1
                self.data[doc_index, w] += 1
            self.docs.append(current_doc)
            counters.append(current_counter)

        self.data = csr_matrix(self.data)

        self.gensim_corpus = [list(ctr.items()) for ctr in counters]

        return self
