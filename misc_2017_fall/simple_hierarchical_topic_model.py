from __future__ import print_function, division

import numpy as np
from scipy.io import loadmat
from scipy.special import digamma
from scipy.sparse import csc_matrix, isspmatrix_csc
import sys
import io
import logging
import pickle


try:
    from tqdm import tqdm as progress_bar
except:
    def progress_bar(*args, **kwargs):
        return args[0]
    progress_bar.update = lambda n=1: None
    progress_bar.set_postfix = lambda x: None

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

# Convention for Einstein-summation (np.einsum) indices:
NODE, LEAF, WORD_SLOT, VOCAB_WORD, DEPTH, DOC = list(range(6))

class SimpleHierarchicalTopicModel(object):
    def __init__(self, branching_factors, num_epochs, batch_size, vocab,
            do_compute_ELBO = True, save_params_history = False):
        self.num_epochs = num_epochs
        self.branching_factors = branching_factors
        self.vocab = np.asarray(vocab, dtype='object')
        self.batch_size = batch_size
        self.do_compute_ELBO = do_compute_ELBO
        self.save_params_history = save_params_history

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
        self.indicator_rl = np.zeros((self.num_nodes, self.num_leaves))
        # indicator_rl[path r, leaf l] = 1 iff r is prefix of l
        self.indicator_rk = np.zeros((self.num_nodes, self.num_depths))
        # indicator_rk[path r, depth k] = 1 iff len(r) == k

        for path in explore(self.branching_factors, ()):
            node_index = len(self.nodes)
            self.path_to_node_index[path] = node_index
            self.nodes.append(path)

            self.indicator_rk[node_index, len(path)] = 1

            if len(path) == len(self.branching_factors):
                leaf_index = len(self.leaves)
                self.path_to_leaf_index[path] = leaf_index
                self.leaves.append(path)

                for subpath_len in range(self.num_depths):
                    subpath = path[:subpath_len]
                    subpath_index = self.path_to_node_index[subpath]
                    self.indicator_rl[subpath_index, leaf_index] = 1

        self.depth_by_node = np.array([len(path) for path in self.nodes], dtype='int')

    def fit(self, data):
        if not isspmatrix_csc(data):
            _logger.debug("Converting input data to sparse format (CSC)")
            data = csc_matrix(data)

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

        '''
        Variable-naming convention:
        DV = distribution over vocab (global; these distributions constitute the "topics")
        DL = distribution over leaves (per document)
        DD = distribution over depths (per document)
        L = choice of leaf (per word-slot, per document)
        D = choice of depth (per word-slot, per document)

        Comparison to notation in paper:
        var_params_DL[d,i] = mu^lambda_{d,i}
        var_params_DD[d,k] = mu^phi_{d,k}
        var_params_DV[r,v] = mu^theta_{r,v}
        var_params_L[slot(d,n),i] = mu^l_{d,n,i}
        var_params_D[slot(d,n),k] = mu^z_{d,n,k}
        '''
        _logger.debug("Allocating prior params")
        self.prior_params_DL = 0.01 * np.ones(self.num_leaves)
        self.prior_params_DD = 1.0 * np.ones(self.num_depths)
        self.prior_params_DV = 0.1 * np.ones(self.vocab_size)

        _logger.debug("Allocating variational params")
        self.var_params_DL = np.random.uniform(0.99, 1.01, (self.num_docs, self.num_leaves))
        self.var_params_DD = np.random.uniform(0.99, 1.01, (self.num_docs, self.num_depths))
        self.var_params_DV = np.random.uniform(0.99, 1.01, (self.num_nodes, self.vocab_size))
        self.var_params_L = softmax(np.random.uniform(0.99, 1.01, (self.total_corpus_length, self.num_leaves)), axis = -1)
        self.var_params_D = softmax(np.random.uniform(0.99, 1.01, (self.total_corpus_length, self.num_depths)), axis = -1)

        _logger.debug("Generating per-document word-slot arrays")
        self.docs_expanded = []
        self.docs_by_word_slot = np.empty(self.total_corpus_length, dtype='int')
        self.overall_vocab_word_by_slot = []
        overall_token_index = 0
        for doc_index in range(self.num_docs):
            start = self.data.indptr[doc_index]
            end = self.data.indptr[doc_index + 1]
            counts = self.data.data[start:end].astype('int')
            vocab_indices = self.data.indices[start:end].astype('int')

            num_tokens = counts.sum()
            self.docs_by_word_slot[overall_token_index : overall_token_index + num_tokens] = doc_index

            vocab_word_by_slot = np.empty(num_tokens, dtype='int')
            token_index = 0
            for count, vocab_word_index in zip(counts, vocab_indices):
                vocab_word_by_slot[token_index : token_index + count] = vocab_word_index
                token_index += count
                overall_token_index += count
            self.docs_expanded.append(vocab_word_by_slot)
            self.overall_vocab_word_by_slot.extend(vocab_word_by_slot)

        if self.batch_size is None or self.batch_size <= 0:
            _batch_size = self.num_docs
        else:
            _batch_size = self.batch_size

        _logger.debug("Training model")
        self.stats_by_epoch = []
        self.update_stats_by_epoch(epoch_index = -1, step_index = 0)
        with progress_bar(total = self.num_epochs * self.num_docs, mininterval=1.0) as pbar:
            step_index = 0
            for epoch_index in range(self.num_epochs):
                pbar.set_postfix({"Status": "updating params"})

                # Pick a random permutation and iterate through dataset in that order
                doc_order = np.random.permutation(self.num_docs)
                while len(doc_order) > 0:
                    mini_batch_doc_indices = doc_order[:_batch_size]
                    doc_order = doc_order[_batch_size:]
                    self.update(epoch_index, step_index, mini_batch_doc_indices)
                    step_index += 1
                    pbar.update(n = len(mini_batch_doc_indices))

                pbar.set_postfix({"Status": "computing statistics"})
                self.update_stats_by_epoch(epoch_index, step_index)

        return self

    def update_stats_by_epoch(self, epoch_index, step_index):
        if epoch_index >= 0 and self.do_compute_ELBO:
            elbo = self.compute_ELBO()
        else:
            elbo = np.nan
        stats = dict(epoch_index = epoch_index, step_index = step_index,
            ELBO = elbo, step_size = self.step_size(step_index))
        if self.save_params_history:
            stats["var_params_L"] = self.var_params_L.copy()
            stats["var_params_D"] = self.var_params_D.copy()
            stats["var_params_DL"] = self.var_params_DL.copy()
            stats["var_params_DD"] = self.var_params_DD.copy()
            stats["var_params_DV"] = self.var_params_DV.copy()
        self.stats_by_epoch.append(stats)

    def get_stats_by_epoch(self, key, include_init = True, **kwargs):
        if include_init:
            stats = self.stats_by_epoch
        else:
            stats = (x for x in self.stats_by_epoch if x["epoch_index"] >= 0)
        return np.array([entry[key] for entry in stats], **kwargs)

    def step_size(self, step_index):
        return (1 + step_index) ** -0.5

    def update(self, epoch_index, step_index, doc_indices):
        word_slot_indices = []
        vocab_word_by_slot = []
        docs_by_word_slot = []
        for doc_index in doc_indices:
            lo = self.token_offsets_by_document[doc_index]
            hi = self.token_offsets_by_document[doc_index + 1]
            word_slot_indices.extend(range(lo, hi))
            vocab_word_by_slot.extend(self.docs_expanded[doc_index])
            docs_by_word_slot.extend([doc_index for _ in range(lo, hi)])
        del doc_index

        expectation_log_DV = expectation_log_dirichlet(self.var_params_DV, axis = -1)

        # Update L
        log_L = expectation_log_dirichlet(self.var_params_DL[docs_by_word_slot, :], axis = -1) \
            + np.einsum(
                self.indicator_rl, [NODE, LEAF],
                self.var_params_D[np.atleast_2d(word_slot_indices).transpose(), self.depth_by_node], [WORD_SLOT, NODE],
                expectation_log_DV[:, vocab_word_by_slot], [NODE, WORD_SLOT],
                [WORD_SLOT, LEAF])
        self.var_params_L[word_slot_indices, :] = softmax(log_L, axis = -1)

        # Update D
        log_D = expectation_log_dirichlet(self.var_params_DD[docs_by_word_slot, :], axis = -1) \
            + np.einsum(
                self.indicator_rl, [NODE, LEAF],
                self.indicator_rk, [NODE, DEPTH],
                self.var_params_L[word_slot_indices, :], [WORD_SLOT, LEAF],
                expectation_log_DV[:, vocab_word_by_slot], [NODE, WORD_SLOT],
                [WORD_SLOT, DEPTH])
        self.var_params_D[word_slot_indices, :] = softmax(log_D, axis = -1)

        # Update DL
        var_params_DL_by_word_slot = (self.prior_params_DL[np.newaxis, :]
            + self.var_params_L[word_slot_indices, :])
        np.add.at(self.var_params_DL, (docs_by_word_slot, slice(None)), var_params_DL_by_word_slot)

        # Update DD
        var_params_DD_by_word_slot = (self.prior_params_DD[np.newaxis, :]
            + self.var_params_D[word_slot_indices, :])
        np.add.at(self.var_params_DD, (docs_by_word_slot, slice(None)), var_params_DD_by_word_slot)

        # Update DV
        local_contrib_DV_by_word_slot = np.einsum(
            self.indicator_rl, [NODE, LEAF],
            self.var_params_D[np.atleast_2d(word_slot_indices).transpose(), self.depth_by_node], [WORD_SLOT, NODE],
            self.var_params_L[word_slot_indices, :], [WORD_SLOT, LEAF],
            [NODE, WORD_SLOT])
        # Sum local contribs by grouping word-slots according to vocab words
        local_contrib_DV = np.zeros((self.num_nodes, self.vocab_size))
        np.add.at(local_contrib_DV, (slice(None), vocab_word_by_slot), local_contrib_DV_by_word_slot)
        # Update topics according to stochastic update rule
        self.var_params_DV = (1 - self.step_size(step_index)) * self.var_params_DV \
            + self.step_size(step_index) * (self.prior_params_DV[np.newaxis, :] + local_contrib_DV * self.num_docs / len(doc_indices))

    def compute_ELBO(self):
        expectation_log_DV = expectation_log_dirichlet(self.var_params_DV, axis = -1)
        expectation_log_DL = expectation_log_dirichlet(self.var_params_DL, axis = -1)
        expectation_log_DD = expectation_log_dirichlet(self.var_params_DD, axis = -1)

        elbo = 0.0
        elbo += np.einsum(
            self.prior_params_DV[np.newaxis, :] - self.var_params_DV, [NODE, VOCAB_WORD],
            expectation_log_DV, [NODE, VOCAB_WORD],
            [])  # output is a scalar
        elbo += np.einsum(
            self.prior_params_DL[np.newaxis, :] - self.var_params_DL, [DOC, LEAF],
            expectation_log_DL, [DOC, LEAF],
            [])  # output is a scalar
        elbo += np.einsum(
            self.prior_params_DD[np.newaxis, :] - self.var_params_DD, [DOC, DEPTH],
            expectation_log_DD, [DOC, DEPTH],
            [])  # output is a scalar
        elbo += np.einsum(
            self.var_params_L, [WORD_SLOT, LEAF],
            expectation_log_DL[self.docs_by_word_slot, :] - np.log(self.var_params_L), [WORD_SLOT, LEAF],
            [])  # output is a scalar
        elbo += np.einsum(
            self.var_params_D, [WORD_SLOT, DEPTH],
            expectation_log_DD[self.docs_by_word_slot, :] - np.log(self.var_params_D), [WORD_SLOT, DEPTH],
            [])  # output is a scalar
        elbo += np.einsum(
            self.indicator_rl, [NODE, LEAF],
            self.var_params_D[:, self.depth_by_node], [WORD_SLOT, NODE],
            self.var_params_L, [WORD_SLOT, LEAF],
            expectation_log_DV[:, self.overall_vocab_word_by_slot], [NODE, WORD_SLOT],
            [])  # output is a scalar
        return elbo

    def get_expected_topic_vectors(self):
        return self.var_params_DV / self.var_params_DV.sum(axis = -1, keepdims = True)

    def get_top_words_by_node(self, num_words):
        topic_vectors = self.get_expected_topic_vectors()
        top_vocab_indices = np.argsort(-topic_vectors, axis=-1)[:, :num_words]
        result = dict()
        for node_index, path in enumerate(self.nodes):
            result[path] = self.vocab[top_vocab_indices[node_index]]
        return result

    def print_top_words_by_node(self, num_words, depth_first=False, file=None):
        if file is None:
            file = sys.stdout
        max_str_len_path = max([len(str(path)) for path in self.nodes])
        format_str = "{:" + str(max_str_len_path) + "}: {}"
        top_words = self.get_top_words_by_node(num_words = num_words)
        if depth_first:
            node_order = self.nodes
        else:
            node_order = sorted(self.nodes, key=lambda x: (len(x),) + x)
        print("Top words by node:", file=file)
        for path in node_order:
            print(format_str.format(str(path), ", ".join(list(top_words[path]))), file=file)
        return top_words

def main():
    np.random.seed(1)

    logging.basicConfig(
        level = logging.DEBUG,
        format = "%(asctime)s %(message)s")

    output_file = "output.txt"
    model_file = None  #"model.pkl"

    print("Loading data...")
    data = loadmat("/Users/aleverentz/Code/anchor-word-recovery/M_nips.full_docs.mat.trunc.mat")["M"]
    #data = loadmat("/Users/aleverentz/ucsd-research/hier-topic-models/data/abstracts.mat")["M"]
    print("Vocab size: {}".format(data.shape[0]))
    print("Num documents: {}".format(data.shape[1]))
    print("Nonzero entries: {}".format(data.nnz))
    print("Loading vocab...")
    vocab = load_vocab("/Users/aleverentz/Code/anchor-word-recovery/vocab.nips.txt.trunc")
    #vocab = load_vocab("/Users/aleverentz/ucsd-research/hier-topic-models/data/abstracts_vocab.txt")
    print("Vocab size = {}".format(len(vocab)))
    assert data.shape[0] == len(vocab)
    branching_factors = [5, 5]
    model = SimpleHierarchicalTopicModel(
        branching_factors = branching_factors, num_epochs = 20,
        batch_size = 100, vocab = vocab)
    model.fit(data)
    top_words = model.print_top_words_by_node(num_words = 10)
    if output_file is not None:
        print("Outputting to {}".format(output_file))
        with io.open(output_file, mode='w', encoding='utf8') as f:
            model.print_top_words_by_node(num_words = 10, file=f)
    if model_file is not None:
        print("Saving model to {}".format(model_file))
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    main()
