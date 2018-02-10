import numpy as np

from simple_hierarchical_topic_model import explore_branching_factors

class SimData(object):
    '''Generate a simulated dataset'''
    def __init__(self, branching_factors, num_docs, doc_length, topic_sharpness, alpha_leaves, alpha_depths):
        self.branching_factors = branching_factors
        self.nodes = explore_branching_factors(self.branching_factors)
        self.num_nodes = len(self.nodes)
        self.max_depth = max([len(x) for x in self.nodes])
        self.num_leaves = len([x for x in self.nodes if len(x) == self.max_depth])
        self.num_depths = len(np.unique([len(x) for x in self.nodes]))
        self.vocab_size = self.num_nodes * 2 + 4
        self.num_docs = num_docs
        self.doc_length = doc_length
        self.topic_sharpness = topic_sharpness
        self.vocab = ["w{}".format(i) for i in range(self.vocab_size)]
        self.alpha_leaves = np.broadcast_to(alpha_leaves, (self.num_leaves,)).astype('float')
        self.alpha_depths = np.broadcast_to(alpha_depths, (self.num_depths,)).astype('float')
        self.init_topics()

    def init_topics(self):
        self.topics_by_path = dict()
        self.topics_by_index = []
        self.leaves = []
        for node_index, path in enumerate(self.nodes):
            current_topic = np.ones(self.vocab_size)
            heavy_vocab_indices = slice(node_index * 2, (node_index + 1) * 2)
            current_topic[heavy_vocab_indices] *= self.topic_sharpness
            current_topic /= current_topic.sum()
            self.topics_by_path[path] = current_topic
            self.topics_by_index.append(current_topic)
            if len(path) == self.max_depth:
                self.leaves.append(path)
        self.topics_by_index = np.stack(self.topics_by_index)

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
