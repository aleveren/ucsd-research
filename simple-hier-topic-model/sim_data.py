import numpy as np

class SimData(object):
    '''Generate a simulated dataset'''
    def __init__(self, num_leaves, num_docs, doc_length, topic_sharpness):
        self.num_leaves = num_leaves
        self.num_depths = 2
        self.vocab_size = self.num_leaves * 2 + 6
        self.num_docs = num_docs
        self.doc_length = doc_length
        self.topic_sharpness = topic_sharpness
        self.vocab = ["w{}".format(i) for i in range(self.vocab_size)]
        self.alpha_leaves = 100.0 * np.ones((self.num_leaves,))
        self.alpha_depths = 100.0 * np.ones((self.num_depths,))
        self.init_topics()

    def init_topics(self):
        self.nodes = [()] + [(i,) for i in range(self.num_leaves)]
        self.topics_by_path = dict()
        self.topics_by_index = []
        for node_index, path in enumerate(self.nodes):
            current_topic = np.ones(self.vocab_size)
            heavy_vocab_indices = slice(node_index * 2, (node_index + 1) * 2)
            current_topic[heavy_vocab_indices] *= self.topic_sharpness
            current_topic /= current_topic.sum()
            self.topics_by_path[path] = current_topic
            self.topics_by_index.append(current_topic)
        self.topics_by_index = np.stack(self.topics_by_index)

    def generate(self):
        self.docs = []
        self.docs_aux = []

        for i in range(self.num_docs):
            leaf_distrib = np.random.dirichlet(self.alpha_leaves)
            depth_distrib = np.random.dirichlet(self.alpha_depths)
            node_distrib = np.concatenate([[depth_distrib[0]], depth_distrib[1] * leaf_distrib])
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
