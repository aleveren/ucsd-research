'''
Variational inference for the nested Chinese Restaurant Process topic model
'''

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from collections import namedtuple, OrderedDict

try:
    import tqdm
except ImportError:
    tqdm = None

class ErrIfNotFound(object):
    pass

_default_branch_factor = 5

class TreeNode(object):
    def __init__(self, children):
        self.children = children

    def depth(self):
        if len(self.children) == 0:
            return 0
        depths = map(lambda x: 1 + x.depth(), self.children)
        min_d, max_d = min(depths), max(depths)
        assert min_d == max_d
        return max_d

    def lookup_path(self, path, default = ErrIfNotFound()):
        if len(path) == 0:
            return self
        if path[0] < len(self.children):
            return self.children[path[0]].lookup_path(path[1:], default = default)
        if isinstance(default, ErrIfNotFound):
            raise ValueError("Path not found")
        return None

    def inner_and_full_paths(self, start_at = None, prefix_so_far = ()):
        if start_at is not None:
            prefix_so_far = start_at
            iterable = self.lookup_path(start_at).inner_and_full_paths(
                start_at = None, prefix_so_far = prefix_so_far)
            for p in iterable:
                yield p
        else:
            yield prefix_so_far
            for k, c in enumerate(self.children):
                new_prefix = tuple(prefix_so_far) + (k,)
                for p in c.inner_and_full_paths(start_at = None, prefix_so_far = new_prefix):
                    yield p

    @classmethod
    def from_branch_structure(cls, branch_structure):
        if len(branch_structure) == 0:
            return cls(children = [])
        children = [cls.from_branch_structure(branch_structure[1:])
            for i in range(branch_structure[0])]
        return cls(children = children)

class NCRP(object):
    def __init__(self, **kwargs):
        self._dict = dict(
            alphaW = 0.1,
            alphaV = 0.1,
            alphaTheta = None,
            branch_structure = None,
            depth = 2,
            iterations = 100,
            progress_bar = None,
        )
        for k, v in kwargs.items():
            assert k in self._dict, "Unrecognized parameter: {}".format(k)
            self._dict[k] = v
        # Set a few default values that depend on depth
        if self._dict["alphaTheta"] is None:
            self._dict["alphaTheta"] = np.ones((self._dict["depth"] + 1))
        if self._dict["branch_structure"] is None:
            self._dict["branch_structure"] = \
                [_default_branch_factor for i in range(self._dict["depth"])]

        assert len(self._dict["alphaTheta"]) == self._dict["depth"] + 1, \
            "Length of alphaTheta must be depth + 1"
        assert len(self._dict["branch_structure"]) == self._dict["depth"], \
            "Length of branch_structure must be equal to depth"

    def __repr__(self):
        return (self.__class__.__name__ + "("
            + ", ".join("{}={}".format(k, repr(v)) for k, v in self._dict.items())
            + ")")

    def fit(self, data):
        f = NCRPFit(config = self, data = data)
        return f.fit()

class NCRPFit(object):
    def __init__(self, config, data):
        assert isinstance(config, NCRP)
        self.config = config
        for k, v in config._dict.items():
            setattr(self, k, v)
        self.data = data
        self.n_obs = self.data.shape[0]
        self.vocab_size = self.data.shape[1]

    def init_stats(self):
        self.tree = TreeNode.from_branch_structure(self.branch_structure)

        self.count_paths = 0
        self.path_to_index = dict()
        self.index_to_path = dict()
        for i, p in enumerate(self.tree.inner_and_full_paths()):
            self.count_paths += 1
            self.path_to_index[p] = i
            self.index_to_path[i] = p
        self.j0 = np.ones((self.count_paths,), dtype='int')
        self.l0 = np.ones((self.count_paths,), dtype='int')
        for i, p in enumerate(self.tree.inner_and_full_paths()):
            self.j0[i] = len(self.tree.lookup_path(p).children)
            self.l0[i] = len(p)

        self.word_count_by_doc = np.squeeze(np.asarray(self.data.sum(axis = 1)))

        self.alphaW_var = self.alphaW * np.ones((self.count_paths, self.vocab_size,))
        self.EqlnW = np.ones((self.count_paths, self.vocab_size))

        self.alphaV_var = np.ones((self.count_paths,))
        self.betaV_var = self.alphaV * np.ones((self.count_paths,))
        self.EqlnV = np.ones((self.count_paths,))
        self.Eqln1_V = np.ones((self.count_paths,))

        self.z0 = np.ones((self.n_obs, self.count_paths))
        self.logS = np.ones((self.n_obs, self.count_paths)) + np.random.uniform(0, 0.01, self.z0.shape)
        self.path_prob = softmax(self.logS, axis=1)

        log_phi_var = np.ones((self.n_obs, self.depth + 1))
        log_phi_var += np.random.uniform(0, 0.01, log_phi_var.shape)
        self.phi_var = softmax(log_phi_var, axis=1)
        self.alphaTheta_var = np.ones((self.n_obs, self.depth + 1))
        # TODO: should either or both of these be 3-dimensional arrays?
        # - For example: phi_var[doc, vocab_word, depth]?
        #             or phi_var[doc, word_slot, depth]?

        # Define some expected values according to prior distrib p
        self.EplnV = digamma(1) - digamma(1 + self.alphaV)
        self.Epln1_V = digamma(self.alphaV) - digamma(1 + self.alphaV)
        self.EplnW = digamma(self.alphaW) - digamma(self.vocab_size * self.alphaW)

    def fit(self):
        self.init_stats()
        fit_desc = "Optimizing via coordinate ascent"
        for iter_index in self.get_progress_bar(range(self.iterations), desc=fit_desc):
            self.update()
        return self

    def update(self):
        # Update path-specific variational expectations
        self.EqlnV = digamma(self.alphaV_var) - digamma(self.alphaV_var + self.betaV_var)
        self.Eqln1_V = digamma(self.betaV_var) - digamma(self.alphaV_var + self.betaV_var)
        self.EqlnW = digamma(self.alphaW_var) - digamma(np.sum(self.alphaW_var, axis=1, keepdims=True))
        # The root node is a special case for the stick-breaking proportions
        root_path_index = self.path_to_index[()]
        self.EqlnV[root_path_index] = 0.0
        self.Eqln1_V[root_path_index] = 0.0  # TODO: Technically -infinity?  Should never be used?  OK b/c constant?

        # Compute z0
        self.z0 = np.zeros(self.z0.shape)
        for path_index, path in self.index_to_path.items():
            for k in range(self.depth+1):
                if k < len(path):
                    subpath_index = self.path_to_index[path[:k]]
                    self.z0[:, path_index] += self.phi_var[:, k] * self.data.dot(self.EqlnW[subpath_index, :])
                else:
                    self.z0[:, path_index] += self.phi_var[:, k] * self.word_count_by_doc * self.EplnW

        # Compute logS and path probabilities
        self.logS = (self.z0
            + (self.depth - self.l0[np.newaxis, :]) * self.EplnV
            - self.j0[np.newaxis, :] * self.Epln1_V
            - (self.depth - self.l0[np.newaxis, :]) * np.log(1 - np.exp(self.Epln1_V)))
        for path_index, path in self.index_to_path.items():
            for k in range(len(path) + 1):
                subpath_index = self.path_to_index[path[:k]]
                self.logS[:, path_index] += self.EqlnV[subpath_index]
                if k < len(path):
                    # TODO: double-check the indexing here
                    for j in range(path[k] - 1):
                        aux_path = tuple(path[:k]) + (j,)
                        aux_path_index = self.path_to_index[aux_path]
                        self.logS[:, path_index] += self.Eqln1_V[aux_path_index]
            for j in range(self.j0[path_index]):
                extended_path_index = self.path_to_index[tuple(path) + (j,)]
                self.logS[:, path_index] += self.Eqln1_V[extended_path_index]
        self.path_prob = softmax(self.logS, axis=1)

        # Update variational parameters for distribution over depths
        self.alphaTheta_var = self.alphaTheta[np.newaxis, :] + self.phi_var * self.word_count_by_doc[:, np.newaxis]

        # Update variational parameters for depth indicators
        # TODO: experiment with phi_var[d, n, k] instead of phi_var[d, k]
        EqlnTheta = digamma(self.alphaTheta_var) - digamma(self.alphaTheta_var.sum(axis=1, keepdims=True))
        log_phi_var = EqlnTheta.copy()
        for path_index, path in self.index_to_path.items():
            for k in range(self.depth + 1):
                if k < len(path):
                    subpath_index = self.path_to_index[path[:k]]
                    term = self.path_prob[:, path_index] * self.data.dot(self.EqlnW[subpath_index, :])
                else:
                    term = self.EplnW * self.path_prob[:, path_index] * self.word_count_by_doc
                log_phi_var[:, k] += term / self.word_count_by_doc
        self.phi_var = softmax(log_phi_var, axis=1)

        # Update variational parameters for stick-breaking proportions
        self.alphaV_var[:] = 1.0
        self.betaV_var[:] = self.alphaV
        for path_index, path in self.index_to_path.items():
            for child_path in self.tree.inner_and_full_paths(start_at = path):
                child_path_index = self.path_to_index[child_path]
                self.alphaV_var[path_index] += np.sum(self.path_prob[:, child_path_index])
            if len(path) == 0:
                self.betaV_var[path_index] += np.sum(self.path_prob[:, path_index])
            else:
                super_path = path[:-1]
                super_path_index = self.path_to_index[super_path]
                self.betaV_var[path_index] += np.sum(self.path_prob[:, super_path_index])
                for r_sibling in range(len(self.tree.lookup_path(super_path).children)):
                    if r_sibling <= path[-1]:
                        continue
                    r_sibling_path = tuple(super_path) + (r_sibling,)
                    for aux_path in self.tree.inner_and_full_paths(start_at = r_sibling_path):
                        aux_path_index = self.path_to_index[aux_path]
                        self.betaV_var[path_index] += np.sum(self.path_prob[:, aux_path_index])

        # Update variational parameters for topic distributions
        self.alphaW_var[:, :] = self.alphaW
        for path_index, path in self.index_to_path.items():
            current_prob = np.zeros((self.n_obs,))
            for child_path in self.tree.inner_and_full_paths(start_at = path):
                child_path_index = self.path_to_index[child_path]
                current_prob += self.path_prob[:, child_path_index]
            # TODO: perform this multiplication more efficiently, making use of sparse format
            prod = (np.asarray(self.data.todense())
                * self.phi_var[:, len(path), np.newaxis]
                * current_prob[:, np.newaxis])
            self.alphaW_var[path_index, :] += np.sum(prod, axis=0)

        # TODO: update tree structure (grow / prune / merge)

    def get_most_likely_paths(self):
        path_indices_by_doc = np.argmax(self.path_prob, axis=1)
        most_likely_paths = np.empty((self.n_obs,), dtype='object')
        for doc_index, p in enumerate(path_indices_by_doc):
            most_likely_paths[doc_index] = self.index_to_path[p]
        return most_likely_paths

    def get_top_words_per_node(self, k, vocab):
        assert k > 0
        result = OrderedDict()
        # Process shorter paths first
        sorted_paths = sorted(self.path_to_index.keys(),
            key = lambda x: (len(x), x))
        for path in sorted_paths:
            path_index = self.path_to_index[path]
            alpha = self.alphaW_var[path_index, :]
            word_indices = np.argpartition(alpha, -k)[-k:]
            top_alpha = alpha[word_indices]
            indirect_indices = np.argsort(-top_alpha)
            word_indices = word_indices[indirect_indices]
            top_alpha = top_alpha[indirect_indices]
            words = []
            for word_index in word_indices:
                words.append(vocab[word_index])
            result[path] = (words, word_indices, top_alpha)
        return result

    def get_progress_bar(self, iterable=None, **kwargs):
        if iterable is not None:
            kwargs["iterable"] = iterable

        if self.progress_bar == 'notebook' and tqdm is not None:
            return tqdm.tqdm_notebook(**kwargs)
        elif self.progress_bar == 'terminal' and tqdm is not None:
            return tqdm.tqdm(**kwargs)
        else:
            return iterable

def softmax(X, axis):
    '''Compute softmax using exp-normalize trick'''
    result = X - np.max(X, axis=axis, keepdims=True)
    np.exp(result, out=result)
    result /= np.sum(result, axis=axis, keepdims=True)
    return result

def main():
    ncrp = NCRP(alpha = 0.1)
    data = TODO  # TODO
    result = ncrp.fit(data)
    print(result)

if __name__ == "__main__":
    main()
