import numpy as np

from simple_hierarchical_topic_model import explore_branching_factors

class SimData(object):
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
            heavy_words_per_topic = 2):
        self.branching_factors = branching_factors
        self.nodes = explore_branching_factors(self.branching_factors)
        self.num_nodes = len(self.nodes)
        self.max_depth = max([len(x) for x in self.nodes])
        self.num_leaves = len([x for x in self.nodes if len(x) == self.max_depth])
        self.num_depths = len(np.unique([len(x) for x in self.nodes]))
        self.vocab_size = vocab_size
        self.overlap = "none" if overlap is None else overlap.lower()
        self.heavy_words_per_topic = heavy_words_per_topic
        self.num_docs = num_docs
        self.doc_length = doc_length
        self.topic_sharpness = topic_sharpness
        self.alpha_leaves = np.broadcast_to(alpha_leaves, (self.num_leaves,)).astype('float')
        self.alpha_depths = np.broadcast_to(alpha_depths, (self.num_depths,)).astype('float')
        self.init_topics_and_vocab()

    def init_topics_and_vocab(self):
        heavy_indices = list(self.get_heavy_indices(overlap = self.overlap))

        min_vocab_size = 2
        for h in heavy_indices:
            min_vocab_size = max(min_vocab_size, 1 + np.max(h))

        if self.vocab_size is None:
            self.vocab_size = min_vocab_size + 4
        assert self.vocab_size >= min_vocab_size, \
            "Vocab size must be at least {}".format(min_vocab_size)
        self.vocab = ["w{}".format(i) for i in range(self.vocab_size)]

        self.heavy_indicator = np.zeros((self.num_nodes, self.vocab_size))
        for node_index in range(self.num_nodes):
            self.heavy_indicator[node_index, heavy_indices[node_index]] = 1

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
