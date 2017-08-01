'''
Variational inference for the nested Chinese Restaurant Process topic model
'''

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
from collections import namedtuple

try:
    import tqdm
except ImportError:
    tqdm = None

class ErrIfNotFound(object):
    pass

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

    def is_leaf(self):
        return len(self.children) == 0

    def lookup_path(self, path, default = ErrIfNotFound()):
        if len(path) == 0:
            return self
        if path[0] < len(self.children):
            return self.children[path[0]].lookup_path(path[1:])
        if isinstance(default, ErrIfNotFound):
            raise ValueError("Path not found")
        return None

    def contains_path(self, path):
        lookup_result = self.lookup_path(path = path, default = None)
        return lookup_result is not None

    def inner_and_full_paths(self, prefix_so_far = ()):
        yield prefix_so_far
        for k, c in enumerate(self.children):
            new_prefix = tuple(prefix_so_far) + (k,)
            for p in c.inner_and_full_paths(prefix_so_far = new_prefix):
                yield p

class NCRP(object):
    def __init__(self, **kwargs):
        self._dict = dict(
            alphaW = 0.1,
            alphaV = 0.1,
            alphaTheta = np.ones((3,)),
            depth = 2,
            iterations = 100,
            progress_bar = None,
        )
        for k, v in kwargs.items():
            assert k in self._dict, "Unrecognized parameter: {}".format(k)
            self._dict[k] = v
        assert len(self._dict["alphaTheta"]) == self._dict["depth"] + 1, \
            "Length of alphaTheta must be depth + 1"

    def __repr__(self):
        return (self.__class__.__name__ + "("
            + ", ".join("{}={}".format(k, v) for k, v in self._dict.items())
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

        self.tree = TreeNode(children = [])
        for i in range(self.depth):
            self.tree = TreeNode(children = [self.tree])

        self.phi_var = (1.0 / self.depth) * np.ones((self.n_obs, self.depth + 1))
        self.alphaTheta_var = np.ones((self.n_obs, self.depth + 1))
        # TODO: should either or both of these be 3-dimensional arrays?
        # - For example: phi_var[doc, vocab_word, depth]?
        #             or phi_var[doc, word_slot, depth]?

        # Define some expected values according to prior distrib p
        self.EplnV = digamma(1) - digamma(1 + self.alphaV)
        self.Epln1_V = digamma(self.alphaV) - digamma(1 + self.alphaV)
        self.EplnW = np.ones((self.vocab_size,)) * \
            (digamma(self.alphaW) - digamma(self.vocab_size * self.alphaW))

        self.tree_var_params = dict()
        self.tree_stats = dict()

    def fit(self):
        pbar = self.get_progress_bar(total = self.iterations)
        self.init_tree_var_params()
        for iter_index in range(self.iterations):
            self.update_tree_stats()
            self.update_doc_stats()
            # TODO: update tree structure (grow / prune / merge)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        return self

    def init_tree_var_params(self):
        '''Initialize variational params for each node in the tree'''
        for path in self.tree.inner_and_full_paths():
            self.tree_var_params[path] = dict(
                alphaV_var = 1.0,
                betaV_var = 1.0,
                alphaW_var = np.ones((self.vocab_size,)),
            )

    def update_doc_stats(self):
        # Update level Dirichlet distributions (variational params for theta_d)
        self.alphaTheta_var = self.alphaTheta[np.newaxis, :] + self.phi_var

        # Update level categorical distributions (variational params for z_d)
        log_phi_var = np.zeros(self.phi_var.shape)
        for d in range(self.n_obs):
            for k in range(self.depth + 1):
                current = digamma(self.alphaTheta_var[d, k]) - digamma(np.sum(self.alphaTheta_var[d, :]))
                for path in self.tree.inner_and_full_paths():
                    for j in range(self.vocab_size):
                        count = self.data[d, j]
                        path_prob = self.tree_stats[path]["path_prob"][d]
                        subpath = path[:k]
                        eqw = self.tree_stats[subpath]["EqlnW"][j]
                        current += count * path_prob * eqw
                log_phi_var[d, k] = current
        self.phi_var = softmax(log_phi_var)

    def update_tree_stats(self):
        count_paths = 0
        for path in self.tree.inner_and_full_paths():
            count_paths += 1
            self.tree_stats[path] = dict()
            self.tree_stats[path]["j0"] = len(self.tree.lookup_path(path).children)
            self.tree_stats[path]["EqlnV"] = self.EqlnV(path)
            self.tree_stats[path]["Eqln1_V"] = self.Eqln1_V(path)
            self.tree_stats[path]["EqlnW"] = self.EqlnW(path)

        for path in self.tree.inner_and_full_paths():
            self.tree_stats[path]["z0"] = np.asarray([self.z0(d, path) for d in range(self.n_obs)])

        logS_matrix = np.zeros((count_paths, self.n_obs))
        for path_index, path in enumerate(self.tree.inner_and_full_paths()):
            ls = np.asarray([self.logS(d, path) for d in range(self.n_obs)])
            logS_matrix[path_index, :] = ls
            self.tree_stats[path]["logS"] = ls

        # Update probabilities of inner & full paths
        # (variational params for c_d)
        path_prob_matrix = softmax(logS_matrix, axis=0)
        for path_index, path in enumerate(self.tree.inner_and_full_paths()):
            self.tree_stats[path]["path_prob"] = path_prob_matrix[path_index, :]

        # Update topics (variational params for W_path)
        for path in self.tree.inner_and_full_paths():
            for j in range(self.vocab_size):
                current = self.tree_var_params[path]["alphaW_var"][j]
                for d in range(self.n_obs):
                    count = self.data[d, j]
                    term = 0
                    subtree = self.tree.lookup_path(path)
                    for path_prime in subtree.inner_and_full_paths(path):
                        path_prob = self.tree_stats[path_prime]["path_prob"][d]
                        term += self.phi_var[d, len(path)] * path_prob
                    current += count * term
                self.tree_var_params[path]["alphaW_var"][j] = current

        # Update stick-breaking params (variational params for V_path)
        # TODO: double check this logic
        for path in self.tree.inner_and_full_paths():
            if len(path) == 0:
                continue
            alpha = 2.0
            beta = 1.0 + self.alphaV
            for d in range(self.n_obs):
                short_path = path[:-1]
                subtree = self.tree.lookup_path(path)
                for path_prime in subtree.inner_and_full_paths(short_path):
                    path_prob = self.tree_stats[path]["path_prob"][d]
                    alpha += path_prob
                    if (len(path_prime) < len(path) or
                            path_prime[len(path) - 1] > path[-1]):
                        beta += path_prob
            self.tree_var_params[path]["alphaV_var"] = alpha
            self.tree_var_params[path]["betaV_var"] = beta

    def get_progress_bar(self, total):
        if self.progress_bar == 'notebook' and tqdm is not None:
            return tqdm.tqdm_notebook(total = total)
        elif self.progress_bar == 'terminal' and tqdm is not None:
            return tqdm.tqdm(total = total)
        else:
            return None

    def EqlnV(self, path):
        # Expectation of V[path], where V[path] ~ variational distrib q
        alpha = self.tree_var_params[path]["alphaV_var"]
        beta = self.tree_var_params[path]["betaV_var"]
        return digamma(alpha) - digamma(alpha + beta)

    def Eqln1_V(self, path):
        # Expectation of (1-V[path]), where V[path] ~ variational distrib q
        alpha = self.tree_var_params[path]["alphaV_var"]
        beta = self.tree_var_params[path]["betaV_var"]
        return digamma(beta) - digamma(alpha + beta)

    def EqlnW(self, path):
        # Expectation of W[path, :], where W[path, :] ~ variational distrib q
        alpha = self.tree_var_params[path]["alphaW_var"]
        return digamma(alpha) - digamma(np.sum(alpha))

    def z0(self, d, path):
        '''Compute E_q[log p(t_d | z_d, W, c_d = path)]'''
        L = self.depth
        l0 = len(path)
        if l0 < L:
            path = tuple(path) + tuple([0 for i in range(L - l0)])
        result = 0.0
        for j in range(self.vocab_size):
            count = self.data[d, j]
            term = 0.0
            for k in range(L+1):
                subpath = path[:k]
                term += self.phi_var[d, k] * self.tree_stats[subpath]["EqlnW"][j]
            result += count * term
        return result

    def logS(self, d, path):
        result = self.tree_stats[path]["z0"][d]
        L = self.depth
        l0 = len(path)
        for l in range(l0):
            result += self.tree_stats[path[:l]]["EqlnV"]
            for j in range(path[l]):
                new_path = tuple(path[:l-1]) + (j,)
                result += self.tree_stats[new_path]["Eqln1_V"]
        if l0 < L:
            # For inner paths, account for all full paths that leave the
            # truncated tree immediately after passing through `path`.
            j0 = self.tree_stats[path]["j0"]
            result += (L - l0) * self.EplnV
            result -= j0 * self.Epln1_V
            result -= (L - l0) * np.log(1 - np.exp(self.Epln1_V))
            for j in range(j0):
                new_path = tuple(path) + (j,)
                result += self.tree_stats[new_path]["Eqln1_V"]
        return result

def softmax(X, axis=0):
    '''Compute softmax using exp-normalize trick'''
    X = X.copy()
    X -= np.max(X, axis=axis, keepdims=True)
    np.exp(X, out=X)
    X /= np.sum(X, axis=axis, keepdims=True)
    return X

def main():
    ncrp = NCRP(alpha = 0.1)
    data = TODO  # TODO
    result = ncrp.fit(data)
    print(result)

if __name__ == "__main__":
    main()
