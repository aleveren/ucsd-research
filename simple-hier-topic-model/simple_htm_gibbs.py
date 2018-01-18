from __future__ import print_function, division

import numpy as np
from scipy.sparse import csc_matrix, isspmatrix_csc
import logging
import copy

try:
    from tqdm import tqdm as progress_bar
except:
    def progress_bar(*args, **kwargs):
        return args[0]
    progress_bar.update = lambda n=1: None
    progress_bar.set_postfix = lambda x: None


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())

def softmax(X, axis):
    X = np.asarray(X)
    eX = np.exp(X)
    return eX / eX.sum(axis = axis, keepdims = True)

EPSILON = 1e-7

class SimpleHTMGibbs(object):
    def __init__(self, branching_factors, num_epochs, burn_in, lag, reset_after_n_samples = None):
        self.branching_factors = copy.copy(branching_factors)
        self.num_epochs = num_epochs
        self.burn_in = burn_in
        self.lag = lag
        self.reset_after_n_samples = reset_after_n_samples

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
        self.indicator_node_leaf = np.zeros((self.num_nodes, self.num_leaves))
        # indicator_node_leaf[path r, leaf l] = 1 iff r is prefix of l
        self.indicator_node_depth = np.zeros((self.num_nodes, self.num_depths))
        # indicator_node_depth[path r, depth k] = 1 iff len(r) == k

        for path in explore(self.branching_factors, ()):
            node_index = len(self.nodes)
            self.path_to_node_index[path] = node_index
            self.nodes.append(path)

            self.indicator_node_depth[node_index, len(path)] = 1

            if len(path) == len(self.branching_factors):
                leaf_index = len(self.leaves)
                self.path_to_leaf_index[path] = leaf_index
                self.leaves.append(path)

                for subpath_len in range(self.num_depths):
                    subpath = path[:subpath_len]
                    subpath_index = self.path_to_node_index[subpath]
                    self.indicator_node_leaf[subpath_index, leaf_index] = 1

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

        self.prior_params = {
            "DL": 0.01 * np.ones(self.num_leaves),
            "DD": 1.0 * np.ones(self.num_depths),
            "DV": 0.1 * np.ones(self.vocab_size),
        }

        self.init_state()

        epochs_until_next_sample = self.burn_in
        samples_since_last_reset = 0

        self.stored_samples = []

        for epoch_index in progress_bar(range(self.num_epochs)):
            order = np.random.permutation(len(self.var_names_and_indices))
            for var_index in order:
                var_name, var_local_index = self.var_names_and_indices[var_index]
                self.sample(var_name, var_local_index)

            if epochs_until_next_sample <= 0:
                self.stored_samples.append(copy.deepcopy(self.var_state))
                samples_since_last_reset += 1

                if self.reset_after_n_samples is not None and \
                        samples_since_last_reset >= self.reset_after_n_samples:
                    samples_since_last_reset = 0
                    epochs_until_next_sample = self.burn_in
                    self.init_state()
                else:
                    epochs_until_next_sample = self.lag
            else:
                epochs_until_next_sample -= 1

    def init_state(self):
        self.var_state = {
            "L":  np.random.choice(self.num_leaves, size=self.total_corpus_length),
            "D":  np.random.choice(self.num_depths, size=self.total_corpus_length),
            "DL": np.random.dirichlet(np.ones(self.num_leaves), size=self.num_docs),
            "DD": np.random.dirichlet(np.ones(self.num_depths), size=self.num_docs),
            "DV": np.random.dirichlet(np.ones(self.vocab_size), size=self.num_nodes),
        }

        self.var_names_and_indices = []
        self.var_names_and_indices.extend([("L",  i) for i in range(self.total_corpus_length)])
        self.var_names_and_indices.extend([("D",  i) for i in range(self.total_corpus_length)])
        self.var_names_and_indices.extend([("DL", i) for i in range(self.num_docs)])
        self.var_names_and_indices.extend([("DD", i) for i in range(self.num_docs)])
        self.var_names_and_indices.extend([("DV", i) for i in range(self.num_nodes)])

    def sample(self, var_name, local_index):
        if var_name == "L":
            doc_index = self.docs_by_word_slot[local_index]
            params = np.log(EPSILON + self.var_state["DL"][doc_index])
            vocab_word_index = self.overall_vocab_word_by_slot[local_index]
            for i in range(self.num_leaves):
                depth = self.var_state["D"][local_index]
                node = self.leaves[i][:depth]
                node_index = self.path_to_node_index[node]
                params[i] += np.log(EPSILON + self.var_state["DV"][node_index, vocab_word_index])
            new_state = np.random.choice(len(params), p = softmax(params, axis=-1))

        elif var_name == "D":
            doc_index = self.docs_by_word_slot[local_index]
            params = np.log(EPSILON + self.var_state["DD"][doc_index])
            vocab_word_index = self.overall_vocab_word_by_slot[local_index]
            for k in range(self.num_depths):
                leaf_index = self.var_state["L"][local_index]
                node = self.leaves[leaf_index][:k]
                node_index = self.path_to_node_index[node]
                params[k] += np.log(EPSILON + self.var_state["DV"][node_index, vocab_word_index])
            new_state = np.random.choice(len(params), p = softmax(params, axis=-1))

        elif var_name == "DL":
            params = self.prior_params["DL"].copy()
            lo = self.token_offsets_by_document[local_index]
            hi = self.token_offsets_by_document[local_index + 1]
            leaf_assignments = self.var_state["L"][lo:hi]
            np.add.at(params, leaf_assignments, np.ones(hi - lo))
            new_state = np.random.dirichlet(params)

        elif var_name == "DD":
            params = self.prior_params["DD"].copy()
            lo = self.token_offsets_by_document[local_index]
            hi = self.token_offsets_by_document[local_index + 1]
            depth_assignments = self.var_state["D"][lo:hi]
            np.add.at(params, depth_assignments, np.ones(hi - lo))
            new_state = np.random.dirichlet(params)

        elif var_name == "DV":
            node_index = local_index
            params = self.prior_params["DV"].copy()
            for token_index, vocab_word_index in enumerate(self.overall_vocab_word_by_slot):
                leaf_index = self.var_state["L"][token_index]
                depth = self.var_state["D"][token_index]
                truncated_leaf_path = self.leaves[leaf_index][:depth]
                if node_index == self.path_to_node_index[truncated_leaf_path]:
                    params[vocab_word_index] += 1
            new_state = np.random.dirichlet(params)

        else:
            raise ValueError("Unrecognized var_name: {}".format(var_name))

        self.var_state[var_name][local_index] = new_state
