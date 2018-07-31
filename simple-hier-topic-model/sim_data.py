import numpy as np
from tqdm import tqdm

from simple_hierarchical_topic_model import explore_branching_factors

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
    def __init__(self, g, num_docs, words_per_doc, vocab_size):
        self.g = g
        self.num_docs = num_docs
        self.words_per_doc = words_per_doc
        self.vocab_size = vocab_size

    def sample(self):
        self.thetas_by_doc = []
        self.docs = []
        self.doc_nodes = []
        # Sample topics
        self.topics = dict()
        for node in self.g.nodes():
            nc = len(list(self.g.successors(node)))
            if nc == 0:
                self.topics[node] = np.random.dirichlet(np.ones(self.vocab_size))
        # Sample documents
        for i in tqdm(range(self.num_docs)):
            thetas = dict()
            for node in self.g.nodes():
                nc = len(list(self.g.successors(node)))
                if nc > 0:
                    alpha = np.ones(nc)
                    thetas[node] = np.random.dirichlet(alpha)
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
    def __init__(self, g, num_docs, words_per_doc, vocab_size):
        self.g = g
        self.num_docs = num_docs
        self.words_per_doc = words_per_doc
        self.vocab_size = vocab_size

    def sample(self):
        self.thetas_by_doc = []
        self.docs = []
        self.doc_paths = []
        self.doc_nodes = []
        # Sample topics
        self.topics = dict()
        for node in self.g.nodes():
            self.topics[node] = np.random.dirichlet(np.ones(self.vocab_size))
        # Sample documents
        for i in tqdm(range(self.num_docs)):
            thetas = dict()
            for node in self.g.nodes():
                nc = len(list(self.g.successors(node)))
                if nc > 0:
                    alpha = np.ones(nc)
                    thetas[node] = np.random.dirichlet(alpha)
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
    def __init__(self, g, num_docs, words_per_doc, vocab_size):
        self.g = g
        self.num_docs = num_docs
        self.words_per_doc = words_per_doc
        self.vocab_size = vocab_size

    def sample(self):
        self.thetas_by_doc = []
        self.docs = []
        self.doc_nodes = []
        # Sample topics
        self.topics = dict()
        for node in self.g.nodes():
            self.topics[node] = np.random.dirichlet(np.ones(self.vocab_size))
        # Sample documents
        for i in tqdm(range(self.num_docs)):
            thetas = dict()
            for node in self.g.nodes():
                nc = len(list(self.g.successors(node)))
                if nc > 0:
                    alpha = np.ones(nc + 1)
                    thetas[node] = np.random.dirichlet(alpha)
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
