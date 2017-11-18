from __future__ import print_function, division

import numpy as np
from scipy.io import loadmat
from scipy.special import digamma
import io
import logging


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

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

class SimpleHierarchicalTopicModel(object):
    def __init__(self, branching_factors, num_epochs, vocab):
        self.num_epochs = num_epochs
        self.branching_factors = branching_factors
        self.vocab = np.asarray(vocab, dtype='object')

        self.num_depths = len(self.branching_factors) + 1
        self.num_leaves = np.prod(self.branching_factors)
        self.num_nodes = 1 + np.sum(np.cumprod(self.branching_factors))

        self.init_paths()

    def init_paths(self):
        def explore(remaining_branching_factors, prefix):
            yield prefix
            if len(remaining_branching_factors) > 0:
                first = remaining_branching_factors[0]
                rest = remaining_branching_factors[1:]
                for i in range(first):
                    new_prefix = prefix + (i,)
                    for path in explore(rest, new_prefix):
                        yield path

        self.nodes = []
        self.path_to_node_index = dict()
        self.leaves = []
        self.path_to_leaf_index = dict()
        self.indicator_rlk = np.zeros((self.num_nodes, self.num_leaves, self.num_depths))
        # indicator_rlk[path r, leaf l, depth k] = 1 iff len(r) == k and r is prefix of l

        for path in explore(self.branching_factors, ()):
            node_index = len(self.nodes)
            self.path_to_node_index[path] = node_index
            self.nodes.append(path)

            if len(path) == len(self.branching_factors):
                leaf_index = len(self.leaves)
                self.path_to_leaf_index[path] = leaf_index
                self.leaves.append(path)

                for subpath_len in range(self.num_depths):
                    subpath = path[:subpath_len]
                    subpath_index = self.path_to_node_index[subpath]
                    self.indicator_rlk[subpath_index, leaf_index, subpath_len] = 1

        #_logger.debug("nodes:\n{}\n{}".format(self.nodes, self.path_to_node_index))
        #_logger.debug("leaves:\n{}\n{}".format(self.leaves, self.path_to_leaf_index))
        #_logger.debug("indicator_rlk:\n{}".format(self.indicator_rlk.transpose((2,0,1))))

    def fit(self, data):
        self.data = data
        self.vocab_size = data.shape[0]
        self.num_docs = data.shape[1]

        _logger.debug("Calculating total corpus length")
        self.total_corpus_length = int(data.sum())
        self.document_lengths = np.squeeze(np.asarray(data.sum(axis = 0))).astype('int')
        self.token_offsets_by_document = np.cumsum(np.concatenate([[0], self.document_lengths]))
        assert len(self.document_lengths) == self.num_docs
        assert len(self.token_offsets_by_document) == self.num_docs + 1
        _logger.debug("Total corpus length = {}".format(self.total_corpus_length))

        _logger.debug("Allocating prior params")
        self.prior_params_DL = np.ones(self.num_leaves)
        self.prior_params_DD = np.ones(self.num_depths)
        self.prior_params_DV = np.ones(self.vocab_size)

        def init_noisy(shape, softmax_normalize = False):
            X = np.ones(shape) + np.random.uniform(-0.1, 0.1, shape)
            if softmax_normalize:
                X = softmax(X, axis=-1)
            return X

        _logger.debug("Allocating variational params")
        self.var_params_DL = init_noisy((self.num_docs, self.num_leaves))
        self.var_params_DD = init_noisy((self.num_docs, self.num_depths))
        self.var_params_DV = init_noisy((self.num_nodes, self.vocab_size))
        self.var_params_L = init_noisy((self.total_corpus_length, self.num_leaves), softmax_normalize = True)
        self.var_params_D = init_noisy((self.total_corpus_length, self.num_depths), softmax_normalize = True)

        step_index = 0
        for epoch_index in range(self.num_epochs):
            ## TODO: pick a random permutation and iterate through dataset in that order
            for doc_index in np.random.permutation(self.num_docs):
                self.print_top_words_by_node(num_words = 5)
                self.update(epoch_index, step_index, doc_index)
                step_index += 1
            #doc_index = np.random.choice(self.num_docs)
            #self.update(step_index, doc_index)

        return self

    def step_size(self, step_index):
        return 0.5 / (1 + step_index)

    def update(self, epoch_index, step_index, doc_index):
        _logger.debug("Epoch {}, step {}, document {}".format(epoch_index, step_index, doc_index))

        lo = self.token_offsets_by_document[doc_index]
        hi = self.token_offsets_by_document[doc_index + 1]
        num_tokens = hi - lo

        _logger.debug("at step 1")
        doc = np.asarray(self.data[:, doc_index].todense()).squeeze().astype('int')
        offsets = np.concatenate([[0], np.cumsum(doc)])
        vocab_word_by_slot = np.empty(num_tokens, dtype='int')
        indicator_token_vocab = np.zeros((num_tokens, self.vocab_size), dtype='int')
        for vocab_word_index in range(self.vocab_size):
            start = offsets[vocab_word_index]
            end = offsets[vocab_word_index + 1]
            if end > start:
                vocab_word_by_slot[start:end] = vocab_word_index
                indicator_token_vocab[start:end, vocab_word_index] = 1

        _logger.debug("at step 2")
        expectation_log_DV = expectation_log_dirichlet(self.var_params_DV, axis = -1)

        # Convention for Einstein-summation (np.einsum) indices:
        # 0 = node, 1 = leaf, 2 = depth, 3 = word slot, 4 = vocab word

        _logger.debug("at step 3")
        log_L = expectation_log_dirichlet(self.var_params_DL[np.newaxis, doc_index, :], axis = -1) \
            + np.einsum(
                self.indicator_rlk, [0, 1, 2],
                self.var_params_D[lo:hi, :], [3, 2],
                indicator_token_vocab, [3, 4],
                expectation_log_DV, [0, 4],
                [3, 1])  # output indices are (word slot, leaf)
        _logger.debug("at step 4")
        self.var_params_L[lo:hi, :] = softmax(log_L, axis = -1)

        _logger.debug("at step 5")
        log_D = expectation_log_dirichlet(self.var_params_DD[np.newaxis, doc_index, :], axis = -1) \
            + np.einsum(
                self.indicator_rlk, [0, 1, 2],
                self.var_params_L[lo:hi, :], [3, 1],
                indicator_token_vocab, [3, 4],
                expectation_log_DV, [0, 4],
                [3, 2])  # output indices are (word slot, depth)
        _logger.debug("at step 6")
        self.var_params_D[lo:hi, :] = softmax(log_D, axis = -1)

        _logger.debug("at step 7")
        self.var_params_DL[doc_index, :] = self.prior_params_DL + np.sum(self.var_params_L[lo:hi, :], axis=0)
        _logger.debug("at step 8")
        self.var_params_DD[doc_index, :] = self.prior_params_DD + np.sum(self.var_params_D[lo:hi, :], axis=0)

        _logger.debug("at step 9")
        local_contrib_DV = np.einsum(
            self.indicator_rlk, [0, 1, 2],
            indicator_token_vocab, [3, 4],
            self.var_params_D[lo:hi, :], [3, 2],
            self.var_params_L[lo:hi, :], [3, 1],
            [0, 4])  # output indices are (node, vocab word)
        _logger.debug("at step 10")
        self.var_params_DV = (1 - self.step_size(step_index)) * self.var_params_DV + self.step_size(step_index) * (self.prior_params_DV[np.newaxis, :] + self.num_docs * local_contrib_DV)
        _logger.debug("at step 11")

    def get_expected_topic_vectors(self):
        return self.var_params_DV / self.var_params_DV.sum(axis = -1, keepdims = True)

    def get_top_words_by_node(self, num_words):
        topic_vectors = self.get_expected_topic_vectors()
        top_vocab_indices = np.argsort(-topic_vectors, axis=-1)[:, :num_words]
        result = dict()
        for node_index, path in enumerate(self.nodes):
            result[path] = self.vocab[top_vocab_indices[node_index]]
        return result

    def print_top_words_by_node(self, num_words):
        top_words = self.get_top_words_by_node(num_words = num_words)
        print("Top words by node:")
        for path in self.nodes:
            print("{}: {}".format(path, ", ".join(list(top_words[path]))))

def main():
    np.random.seed(1)

    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(message)s")

    print("Loading data...")
    data = loadmat("/Users/aleverentz/Code/anchor-word-recovery/M_nips.full_docs.mat.trunc.mat")["M"]
    #data = loadmat("/Users/aleverentz/ucsd-research/hier-topic-models/data/abstracts.mat")["M"]
    print("Data shape = {}".format(data.shape))
    print("Nonzero entries: {}".format(data.nnz))
    print("Loading vocab...")
    vocab = load_vocab("/Users/aleverentz/Code/anchor-word-recovery/vocab.nips.txt.trunc")
    #vocab = load_vocab("/Users/aleverentz/ucsd-research/hier-topic-models/data/abstracts_vocab.txt")
    print("Vocab size = {}".format(len(vocab)))
    assert data.shape[0] == len(vocab)
    branching_factors = [2, 2]
    model = SimpleHierarchicalTopicModel(branching_factors = branching_factors, num_epochs = 1, vocab = vocab)
    model.fit(data)
    top_words = model.get_top_words_by_node(num_words = 10)
    model.print_top_words_by_node(num_words = 10)

if __name__ == "__main__":
    main()
