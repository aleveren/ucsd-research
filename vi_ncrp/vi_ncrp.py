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
    def __init__(self, children, vocab_size):
        self.children = children
        self.stats = dict(
            alphaV_var = 1.0,
            betaV_var = 1.0,
            alphaW_var = np.ones((vocab_size,)),
        )

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

class NCRP(object):
    def __init__(self, **kwargs):
        self._dict = dict(
            alphaW = 0.1,
            alphaV = 0.1,
            depth = 2,
            iterations = 100,
            progress_bar = None,
        )
        for k, v in kwargs.items():
            assert k in self._dict, "Unrecognized parameter: {}".format(k)
            self._dict[k] = v

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

        self.tree = TreeNode(children = [], vocab_size = self.vocab_size)
        for i in range(self.depth):
            self.tree = TreeNode(
                children = [self.tree], vocab_size = self.vocab_size)

        self.phi_var = (1.0 / self.depth) * np.ones((self.n_obs, self.depth + 1))

        # Define some expected values according to prior distrib p
        self.EplnV = digamma(1) - digamma(1 + self.alphaV)
        self.Epln1_V = digamma(self.alphaV) - digamma(1 + self.alphaV)
        self.EplnW = np.ones((self.vocab_size,)) * \
            (digamma(self.alphaW) - digamma(self.vocab_size * self.alphaW))

    def fit(self):
        pbar = self.get_progress_bar(total = self.iterations)
        for iter_index in range(self.iterations):
            self.update_logS()
            # TODO
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        return self

    def get_progress_bar(self, total):
        if self.progress_bar == 'notebook' and tqdm is not None:
            return tqdm.tqdm_notebook(total = total)
        elif self.progress_bar == 'terminal' and tqdm is not None:
            return tqdm.tqdm(total = total)
        else:
            return None

    def EqlnV(self, path):
        # Expectation of V[path], where V[path] ~ variational distrib q
        node = self.tree.lookup_path(path, default = None)
        if node is not None:
            alpha = node.stats["alphaV_var"]
            beta = node.stats["betaV_var"]
            return digamma(alpha) - digamma(alpha + beta)
        return self.EplnV

    def Eqln1_V(self, path):
        # Expectation of (1-V[path]), where V[path] ~ variational distrib q
        node = self.tree.lookup_path(path, default = None)
        if node is not None:
            alpha = node.stats["alphaV_var"]
            beta = node.stats["betaV_var"]
            return digamma(beta) - digamma(alpha + beta)
        return self.Epln1_V

    def EqlnW(self, path):
        # Expectation of W[path, :],
        # where W[path, :] ~ variational distrib q
        node = self.tree.lookup_path(path, default = None)
        if node is not None:
            alpha = node.stats["alphaW_var"]
            return digamma(alpha) - digamma(np.sum(alpha))
        return self.EplnW

    def z0(self, d, path):
        '''Compute E_q[log p(t_d | z_d, W, c_d = path)]'''
        L = self.depth
        l0 = len(path)
        if l0 < L:
            path = tuple(path) + tuple([0 for i in range(L - l0)])
        assert self.tree.contains_path(path), "Bad Z0 path: {}".format(path)
        result = 0.0
        for j in range(self.vocab_size):
            count = self.data[d, j]
            term = 0.0
            for k in range(L+1):
                subpath = path[:k]
                if self.tree.contains_path(subpath):
                    term += self.phi_var[d, k] * self.EqlnW(subpath)[j]
                else:
                    term += self.phi_var[d, k] * self.EplnW[j]
            result += count * term
        return result

    def logS(self, d, path):
        assert self.tree.contains_path(path)
        result = self.z0(d, path)
        L = self.depth
        l0 = len(path)
        for l in range(l0):
            result += self.EqlnV(path[:l])
            for j in range(path[l]):
                new_path = tuple(path[:l-1]) + (j,)
                result += self.Eqln1_V(new_path)
        if l0 < L:
            # For inner paths, account for all full paths that leave the
            # truncated tree immediately after passing through `path`.
            j0 = len(self.tree.lookup_path(path).children)
            result += (L - l0) * self.EplnV
            result -= j0 * self.Epln1_V
            result -= (L - l0) * np.log(1 - np.exp(self.Epln1_V))
            for j in range(j0):
                new_path = tuple(path) + (j,)
                result += self.Eqln1_V(path)
        return result

    def update_logS(self):
        for path in inner_and_full_paths(tree = self.tree, prefix_so_far = ()):
            logS_by_doc = []
            for d in range(self.n_obs):
                current_logS = self.logS(d, path)
                logS_by_doc.append(current_logS)
            self.tree.lookup_path(path).stats["logS"] = logS_by_doc

def inner_and_full_paths(tree, prefix_so_far):
    if len(tree.children) == 0:
        return [prefix_so_far]
    result = [prefix_so_far]
    for k, c in enumerate(tree.children):
        to_add = inner_and_full_paths(
            tree = c,
            prefix_so_far = tuple(prefix_so_far) + (k,))
        result.extend(to_add)
    return result

def main():
    ncrp = NCRP(alpha = 0.1)
    data = TODO  # TODO
    result = ncrp.fit(data)
    print(result)

if __name__ == "__main__":
    main()
