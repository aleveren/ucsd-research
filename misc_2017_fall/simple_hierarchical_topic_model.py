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
    def __init__(self, branching_factors, num_iter):
        self.num_iter = num_iter
        self.branching_factors = branching_factors

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

        _logger.debug("Allocating variational params")
        self.var_params_DL = np.ones((self.num_docs, self.num_leaves))
        self.var_params_DD = np.ones((self.num_docs, self.num_depths))
        self.var_params_DV = np.ones((self.num_nodes, self.vocab_size))
        self.var_params_L = np.ones((self.total_corpus_length, self.num_leaves)) / self.num_leaves
        self.var_params_D = np.ones((self.total_corpus_length, self.num_depths)) / self.num_depths

        for iter_index in range(self.num_iter):
            ## TODO: pick a random permutation and iterate through dataset in that order
            #for doc_index in np.random.permutation(sef.num_docs):
            #    self.update(iter_index, doc_index)
            doc_index = np.random.choice(self.num_docs)
            self.update(iter_index, doc_index)

        return self

    def rho(self, iter_index):
        return 1. / (1 + iter_index)

    def update(self, iter_index, doc_index):
        _logger.debug("Iteration {}, document = {}".format(iter_index, doc_index))

        lo = self.token_offsets_by_document[doc_index]
        hi = self.token_offsets_by_document[doc_index + 1]
        num_tokens = hi - lo

        TODO = 0  # TODO

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

        #_logger.debug("doc: {}, offsets: {}, vocab_word_by_slot: {}".format(doc, offsets, vocab_word_by_slot))

        def fake_einsum(*args):
            from collections import defaultdict
            shapes = []
            sizes_by_index = defaultdict(list)
            for i, a in enumerate(args):
                if i % 2 == 1:
                    which_tensor = (i - 1) // 2
                    assert len(a) == len(current_shape), "len mismatch in tensor {}: {} vs {}".format(which_tensor, a, current_shape)
                    for dim, size in zip(a, current_shape):
                        sizes_by_index[dim].append((size, "tensor={}".format(which_tensor)))
                elif i < len(args) - 1:
                    current_shape = a.shape
                    shapes.append(current_shape)
                else:
                    output_indices = a

            sizes_final = dict()
            for k, v in sizes_by_index.items():
                sizes = set([vi[0] for vi in v])
                assert len(sizes) == 1
                sizes_final[k] = list(sizes)[0]

            output_shape = [sizes_final[dim] for dim in output_indices]

            print(shapes)
            print(sizes_final)
            print(sizes_by_index)
            print(output_indices)
            print(output_shape)

        expectation_log_DV = expectation_log_dirichlet(self.var_params_DV, axis = -1)

        # Convention for Einstein-summation (np.einsum) indices:
        # 0 = node, 1 = leaf, 2 = depth, 3 = word slot, 4 = vocab word

        log_L = expectation_log_dirichlet(self.var_params_DL[np.newaxis, doc_index, :], axis = -1) \
            + np.einsum(
                self.indicator_rlk, [0, 1, 2],
                self.var_params_D[lo:hi, :], [3, 2],
                indicator_token_vocab, [3, 4],
                expectation_log_DV, [0, 4],
                [3, 1])  # output indices are (word slot, leaf)
        self.var_params_L[lo:hi, :] = softmax(log_L, axis = -1)

        log_D = expectation_log_dirichlet(self.var_params_DD[np.newaxis, doc_index, :], axis = -1) \
            + np.einsum(
                self.indicator_rlk, [0, 1, 2],
                self.var_params_L[lo:hi, :], [3, 1],
                indicator_token_vocab, [3, 4],
                expectation_log_DV, [0, 4],
                [3, 2])  # output indices are (word slot, depth)
        self.var_params_D[lo:hi, :] = softmax(log_D, axis = -1)

        self.var_params_DL[doc_index, :] = self.prior_params_DL[np.newaxis, :] + np.sum(self.var_params_L[lo:hi, :], axis=0)
        self.var_params_DD[doc_index, :] = self.prior_params_DD[np.newaxis, :] + np.sum(self.var_params_D[lo:hi, :], axis=0)

        local_contrib_DV = np.einsum(
            self.indicator_rlk, [0, 1, 2],
            indicator_token_vocab, [3, 4],
            self.var_params_D[lo:hi, :], [3, 2],
            self.var_params_L[lo:hi, :], [3, 1],
            [0, 4])  # output indices are (node, vocab word)
        self.var_params_DV = (1 - self.rho(iter_index)) * self.var_params_DV + self.rho(iter_index) * (self.prior_params_DV[np.newaxis, :] + self.num_docs * local_contrib_DV)

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
    model = SimpleHierarchicalTopicModel(branching_factors = branching_factors, num_iter = 5)
    model.fit(data)
    #print("Results")

if __name__ == "__main__":
    main()
