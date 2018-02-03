from __future__ import print_function, division

import numpy as np
from scipy.io import loadmat
from scipy.special import gammaln, digamma
from scipy.sparse import csc_matrix, isspmatrix_csc
import sys
import io
import logging
import pickle
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

_default_update_order = ["L", "D", "DL", "DD", "DV"]
_default_step_size_function = lambda step_index: (1 + step_index) ** -0.5

def _explore_branching_factors(factors, prefix):
    yield prefix
    if len(factors) > 0:
        first = factors[0]
        rest = factors[1:]
        for i in range(first):
            new_prefix = prefix + (i,)
            for path in _explore_branching_factors(rest, new_prefix):
                yield path

class SimpleHierarchicalTopicModel(object):
    def __init__(self, vocab, batch_size = None, stopping_condition = None, branching_factors = None,
            prior_params_DL = 0.01, prior_params_DD = 1.0, prior_params_DV = 0.1,
            do_compute_ELBO = True, save_params_history = False, paths = None,
            update_order = None, custom_initializer = None, step_size_function = None):
        if stopping_condition is None or stopping_condition == "default":
            stopping_condition = self._default_stopping_condition()
        self.stopping_condition = stopping_condition
        self.branching_factors = branching_factors
        self.vocab = np.asarray(vocab, dtype='object')
        self.given_prior_params_DL = prior_params_DL
        self.given_prior_params_DD = prior_params_DD
        self.given_prior_params_DV = prior_params_DV
        self.batch_size = batch_size
        self.do_compute_ELBO = do_compute_ELBO
        self.save_params_history = save_params_history
        self.custom_initializer = custom_initializer
        if custom_initializer is not None:
            assert set(custom_initializer.keys()) == set(_default_update_order)
        if update_order is None:
            self.update_order = copy.copy(_default_update_order)
        else:
            self.update_order = copy.copy(update_order)
        assert set(self.update_order) == set(_default_update_order)

        if step_size_function is None:
            step_size_function = _default_step_size_function
        self.step_size = step_size_function

        if self.branching_factors is None:
            assert paths is not None, "Must specify branching_factors or paths"
            self.nodes = [] + paths
        else:
            assert paths is None, "Cannot specify both branching_factors and paths"
            self.nodes = list(_explore_branching_factors(self.branching_factors, ()))

        self.init_paths()

    @classmethod
    def _default_stopping_condition(cls):
        return StoppingCondition(delay_epochs = 5, min_rel_increase = 1e-4, max_epochs = 500)

    def init_paths(self):
        self.num_nodes = len(self.nodes)

        self.depth_by_node = np.array([len(path) for path in self.nodes], dtype='int')
        self.max_depth = max(self.depth_by_node)
        self.unique_depths = np.unique(self.depth_by_node)
        self.depth_to_unique_index = {j: i for i, j in enumerate(self.unique_depths)}
        self.depth_index_by_node = np.array([self.depth_to_unique_index[d] for d in self.depth_by_node])
        self.num_depths = len(self.unique_depths)

        self.leaves = [x for x in self.nodes if len(x) == self.max_depth]
        self.num_leaves = len(self.leaves)

        self.path_to_node_index = dict()
        self.path_to_leaf_index = dict()
        self.indicator_node_leaf = np.zeros((self.num_nodes, self.num_leaves))
        # indicator_node_leaf[path r, leaf l] = 1 iff r is prefix of l
        self.indicator_node_depth = np.zeros((self.num_nodes, self.num_depths))
        # indicator_node_depth[path r, depth k] = 1 iff len(r) == k

        for node_index, path in enumerate(self.nodes):
            self.path_to_node_index[path] = node_index
            self.indicator_node_depth[node_index, self.depth_to_unique_index[len(path)]] = 1

        for leaf_index, path in enumerate(self.leaves):
            self.path_to_leaf_index[path] = leaf_index

            for subpath_len in self.unique_depths:
                subpath = path[:subpath_len]
                subpath_index = self.path_to_node_index.get(subpath, None)
                if subpath_index is not None:
                    self.indicator_node_leaf[subpath_index, leaf_index] = 1

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
        self.prior_params_DL = np.broadcast_to(self.given_prior_params_DL, self.num_leaves).astype('float').copy()
        self.prior_params_DD = np.broadcast_to(self.given_prior_params_DD, self.num_depths).astype('float').copy()
        self.prior_params_DV = np.broadcast_to(self.given_prior_params_DV, self.vocab_size).astype('float').copy()

        _logger.debug("Allocating variational params")
        self.var_params_DL = np.random.uniform(0.01, 1.99, (self.num_docs, self.num_leaves))
        self.var_params_DD = np.random.uniform(0.01, 1.99, (self.num_docs, self.num_depths))
        self.var_params_DV = np.random.uniform(0.01, 1.99, (self.num_nodes, self.vocab_size))
        self.var_params_L = softmax(np.random.uniform(0.01, 1.99, (self.total_corpus_length, self.num_leaves)), axis = -1)
        self.var_params_D = softmax(np.random.uniform(0.01, 1.99, (self.total_corpus_length, self.num_depths)), axis = -1)

        if self.custom_initializer is not None:
            for k, initializer in self.custom_initializer.items():
                setattr(self, "var_params_{}".format(k), initializer.copy())

        assert self.var_params_DL.shape == (self.num_docs, self.num_leaves)
        assert self.var_params_DD.shape == (self.num_docs, self.num_depths)
        assert self.var_params_L.shape == (self.total_corpus_length, self.num_leaves)
        assert self.var_params_D.shape == (self.total_corpus_length, self.num_depths)
        assert self.var_params_DV.shape == (self.num_nodes, self.vocab_size)

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
        with progress_bar(total = self.stopping_condition.max_epochs) as pbar:
            step_index = 0
            epoch_index = 0
            while not self.stopping_condition.should_stop(self):
                pbar.set_postfix({"Status": "updating params"})

                # Pick a random permutation and iterate through dataset in that order
                doc_order = np.random.permutation(self.num_docs)
                while len(doc_order) > 0:
                    mini_batch_doc_indices = doc_order[:_batch_size]
                    doc_order = doc_order[_batch_size:]
                    self.update(epoch_index, step_index, mini_batch_doc_indices)
                    step_index += 1

                pbar.update(n = 1)
                pbar.set_postfix({"Status": "computing statistics"})
                self.update_stats_by_epoch(epoch_index, step_index)
                epoch_index += 1

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

        for update_name in self.update_order:
            if update_name == "L":
                # Update L
                log_L = expectation_log_dirichlet(self.var_params_DL[docs_by_word_slot, :], axis = -1) \
                    + np.einsum(
                        self.indicator_node_leaf, [NODE, LEAF],
                        self.var_params_D[np.atleast_2d(word_slot_indices).transpose(), self.depth_index_by_node], [WORD_SLOT, NODE],
                        expectation_log_DV[:, vocab_word_by_slot], [NODE, WORD_SLOT],
                        [WORD_SLOT, LEAF])
                self.var_params_L[word_slot_indices, :] = softmax(log_L, axis = -1)

            elif update_name == "D":
                # Update D
                log_D = expectation_log_dirichlet(self.var_params_DD[docs_by_word_slot, :], axis = -1) \
                    + np.einsum(
                        self.indicator_node_leaf, [NODE, LEAF],
                        self.indicator_node_depth, [NODE, DEPTH],
                        self.var_params_L[word_slot_indices, :], [WORD_SLOT, LEAF],
                        expectation_log_DV[:, vocab_word_by_slot], [NODE, WORD_SLOT],
                        [WORD_SLOT, DEPTH])
                self.var_params_D[word_slot_indices, :] = softmax(log_D, axis = -1)

            elif update_name == "DL":
                # Update DL
                self.var_params_DL = np.broadcast_to(self.prior_params_DL, self.var_params_DL.shape).copy()
                np.add.at(self.var_params_DL, (docs_by_word_slot, slice(None)), self.var_params_L[word_slot_indices, :])

            elif update_name == "DD":
                # Update DD
                self.var_params_DD = np.broadcast_to(self.prior_params_DD, self.var_params_DD.shape).copy()
                np.add.at(self.var_params_DD, (docs_by_word_slot, slice(None)), self.var_params_D[word_slot_indices, :])

            elif update_name == "DV":
                # Update DV
                local_contrib_DV_by_word_slot = np.einsum(
                    self.indicator_node_leaf, [NODE, LEAF],
                    self.var_params_D[np.atleast_2d(word_slot_indices).transpose(), self.depth_index_by_node], [WORD_SLOT, NODE],
                    self.var_params_L[word_slot_indices, :], [WORD_SLOT, LEAF],
                    [NODE, WORD_SLOT])
                # Sum local contribs by grouping word-slots according to vocab words
                local_contrib_DV = np.zeros((self.num_nodes, self.vocab_size))
                np.add.at(local_contrib_DV, (slice(None), vocab_word_by_slot), local_contrib_DV_by_word_slot)

                if len(doc_indices) == self.num_docs:
                    # Use coordinate-ascent update rule
                    self.var_params_DV = self.prior_params_DV[np.newaxis, :] + local_contrib_DV
                else:
                    # Update topics according to stochastic update rule
                    self.var_params_DV = (1 - self.step_size(step_index)) * self.var_params_DV \
                        + self.step_size(step_index) * (self.prior_params_DV[np.newaxis, :] + local_contrib_DV * self.num_docs / len(doc_indices))

            else:
                raise ValueError("Unsupported update type: {}".format(update_name))

    def compute_ELBO(self):
        expectation_log_DV = expectation_log_dirichlet(self.var_params_DV, axis = -1)
        expectation_log_DL = expectation_log_dirichlet(self.var_params_DL, axis = -1)
        expectation_log_DD = expectation_log_dirichlet(self.var_params_DD, axis = -1)

        elbo = 0.0

        elbo += np.einsum(
            self.prior_params_DV[np.newaxis, :] - self.var_params_DV, [NODE, VOCAB_WORD],
            expectation_log_DV, [NODE, VOCAB_WORD],
            [])  # output is a scalar
        elbo += (gammaln(self.prior_params_DV[np.newaxis, :].sum(axis = -1))
            - gammaln(self.prior_params_DV[np.newaxis, :]).sum(axis = -1)
            - gammaln(self.var_params_DV.sum(axis = -1))
            + gammaln(self.var_params_DV).sum(axis = -1)).sum()

        elbo += np.einsum(
            self.prior_params_DL[np.newaxis, :] - self.var_params_DL, [DOC, LEAF],
            expectation_log_DL, [DOC, LEAF],
            [])  # output is a scalar
        elbo += (gammaln(self.prior_params_DL[np.newaxis, :].sum(axis = -1))
            - gammaln(self.prior_params_DL[np.newaxis, :]).sum(axis = -1)
            - gammaln(self.var_params_DL.sum(axis = -1))
            + gammaln(self.var_params_DL).sum(axis = -1)).sum()

        elbo += np.einsum(
            self.prior_params_DD[np.newaxis, :] - self.var_params_DD, [DOC, DEPTH],
            expectation_log_DD, [DOC, DEPTH],
            [])  # output is a scalar
        elbo += (gammaln(self.prior_params_DD[np.newaxis, :].sum(axis = -1))
            - gammaln(self.prior_params_DD[np.newaxis, :]).sum(axis = -1)
            - gammaln(self.var_params_DD.sum(axis = -1))
            + gammaln(self.var_params_DD).sum(axis = -1)).sum()

        elbo += np.einsum(
            self.var_params_L, [WORD_SLOT, LEAF],
            expectation_log_DL[self.docs_by_word_slot, :] - np.log(self.var_params_L), [WORD_SLOT, LEAF],
            [])  # output is a scalar

        elbo += np.einsum(
            self.var_params_D, [WORD_SLOT, DEPTH],
            expectation_log_DD[self.docs_by_word_slot, :] - np.log(self.var_params_D), [WORD_SLOT, DEPTH],
            [])  # output is a scalar

        elbo += np.einsum(
            self.indicator_node_leaf, [NODE, LEAF],
            self.var_params_D[:, self.depth_index_by_node], [WORD_SLOT, NODE],
            self.var_params_L, [WORD_SLOT, LEAF],
            expectation_log_DV[:, self.overall_vocab_word_by_slot], [NODE, WORD_SLOT],
            [])  # output is a scalar

        assert np.ndim(elbo) == 0, \
            "Internal error: ELBO should be scalar but has shape {}".format(np.shape(elbo))

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


class StoppingCondition(object):
    def __init__(self, delay_epochs = None, min_rel_increase = None, max_epochs = None):
        self.delay_epochs = delay_epochs
        self.min_rel_increase = min_rel_increase
        self.max_epochs = max_epochs
        if min_rel_increase is not None or delay_epochs is not None:
            assert delay_epochs is not None
            assert min_rel_increase is not None
        if max_epochs is None:
            assert delay_epochs is not None
            assert min_rel_increase is not None
        self.elbo_at_prev_continuation = None
        self.epoch_index_at_prev_continuation = None

    def should_stop(self, model):
        if len(model.stats_by_epoch) == 0:
            return False
        epoch_index = model.stats_by_epoch[-1]["epoch_index"]
        elbo = model.stats_by_epoch[-1]["ELBO"]
        if self.max_epochs is not None and epoch_index >= self.max_epochs:
            return True
        if self.min_rel_increase is None:
            return False
        if self.elbo_at_prev_continuation is None or np.isnan(self.elbo_at_prev_continuation) or elbo - self.elbo_at_prev_continuation > np.abs(elbo) * self.min_rel_increase:
            self.elbo_at_prev_continuation = elbo
            self.epoch_index_at_prev_continuation = epoch_index
        stop = epoch_index - self.epoch_index_at_prev_continuation > self.delay_epochs
        return stop
