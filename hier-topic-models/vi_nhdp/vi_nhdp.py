'''
Variational inference for the Nested Hierarchical Dirichlet Process Model
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

class NHDP(object):
    def __init__(self, **kwargs):
        self._dict = dict(
            alphaU1 = 0.1,
            alphaU2 = 0.1,
            alphaV = 0.1,
            alphaTheta = 0.1,
            alphaVStar = 0.1,
            batch_size = 100,
            branch_structure = None,
            depth = 2,
            delay = 0,
            forgetting_rate = 0.75,
            iterations = 100,
            progress_bar = None,
        )
        for k, v in kwargs.items():
            assert k in self._dict, "Unrecognized parameter: {}".format(k)
            self._dict[k] = v
        # Set a few default values that depend on depth
        if self._dict["branch_structure"] is None:
            self._dict["branch_structure"] = \
                [_default_branch_factor for i in range(self._dict["depth"])]

        assert len(self._dict["branch_structure"]) == self._dict["depth"], \
            "Length of branch_structure must be equal to depth"

    def __repr__(self):
        return (self.__class__.__name__ + "("
            + ", ".join("{}={}".format(k, repr(v)) for k, v in self._dict.items())
            + ")")

    def fit(self, data):
        f = NHDPFit(config = self, data = data)
        return f.fit()

class NHDPFit(object):
    def __init__(self, config, data):
        assert isinstance(config, NHDP)
        self.config = config
        for k, v in config._dict.items():
            setattr(self, k, v)
        self.data = data
        self.n_obs = self.data.shape[0]
        self.vocab_size = self.data.shape[1]

    def init_stats(self):
        self.tree = TreeNode.from_branch_structure(self.branch_structure)
        self.num_nodes = 0
        self.index_to_path = dict()
        self.path_to_index = dict()
        for i, node in enumerate(self.tree.inner_and_full_paths()):
            self.num_nodes += 1
            self.index_to_path[i] = node
            self.path_to_index[node] = i
        self.U_var_param_1 = np.zeros((self.n_obs, self.num_nodes))
        self.U_var_param_2 = np.zeros((self.n_obs, self.num_nodes))
        self.Vdoc_var_param_1 = np.zeros((self.n_obs, self.num_nodes))
        self.Vdoc_var_param_2 = np.zeros((self.n_obs, self.num_nodes))
        self.path_probs = dict()

    def fit(self):
        self.init_stats()
        fit_desc = "Optimizing via coordinate ascent"
        for iter_index in self.get_progress_bar(range(self.iterations), desc=fit_desc):
            step_size = (iter_index + 1.0 + self.delay) ** (-self.forgetting_rate)
            self.update(step_size = step_size)
        return self

    def update(self, step_size):
        # Randomly subsample documents
        batch_indices = np.random.choice(np.arange(self.n_obs), size=self.batch_size, replace=False)

        lam_prime = np.zeros((self.num_nodes, self.vocab_size))
        tau_prime_1 = np.zeros((self.num_nodes))
        tau_prime_2 = np.zeros((self.num_nodes))

        for doc_index in batch_indices:
            doc_counts = self.data[doc_index, :]
            doc_size = np.sum(doc_counts)

            # Select a subtree according to greedy algorithm
            subtree, num_selected_nodes = self.select_subtree(doc_index)

            log_nu = np.zeros((doc_size, num_selected_nodes))  # path-probability variational parameter, before applying softmax

            # Optimize variational distributions
            for node in subtree:
                node_index = self.path_to_index[node]

                self.U_var_param_1[doc_index, node_index] = self.alphaU1
                self.U_var_param_2[doc_index, node_index] = self.alphaU2
                # TODO: finish computation

                log_nu = TODO

                word_slot_index = 0
                for vocab_word_index, count in enumerate(doc_counts):
                    for i in range(count):
                        lam_prime[node, vocab_word_index] += TODO
                        word_slot_index += 1

                if len(node) > 0:  # if node is not root
                    self.Vdoc_var_param_1[doc_index, node_index] = 1.0
                    self.Vdoc_var_param_2[doc_index, node_index] = self.alphaV
                    # TODO: finish computation

                    tau_prime_1[node_index] += TODO
                    num_siblings = len(self.tree.lookup_path(node[:-1]).children)
                    for j in range(node[-1] + 1, num_siblings):
                        tau_prime_2[node_index] += TODO

            self.path_probs[doc_index] = softmax(log_nu, axis=TODO)

        # Step in the direction of the natural gradient
        for node in full_tree:  # TODO: or use doc-specific subtree?
            self.theta_var_param[node, word] = (
                (1 - step_size) * self.theta_var_param[node, word]
                + step_size * (self.n_obs / self.batch_size) * lam_prime[node, word]
                + step_size * self.alphaTheta)

            if len(node) > 0:  # if node is not root
                self.V_var_param_1[node] = (
                    (1 - step_size) * self.V_var_param_1[node]
                    + step_size * (self.n_obs / self.batch_size) * tau_prime_1[node]
                    + step_size * 1)
                self.V_var_param_2[node] = (
                    (1 - step_size) * self.V_var_param_2[node]
                    + step_size * (self.n_obs / self.batch_size) * tau_prime_2[node]
                    + step_size * self.alphaVStar)

    def select_subtree(self, doc_index):
        subtree = list(self.tree.inner_and_full_paths())
        return subtree, len(subtree)  # TODO

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
    np.random.seed(1)
    nhdp = NHDP(alphaTheta = 0.1)
    data = np.random.poisson(lam = 2.0, size = (100, 10))  # TODO
    result = nhdp.fit(data)
    print(result)

if __name__ == "__main__":
    main()
