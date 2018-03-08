import numpy as np
import io
from scipy.special import digamma

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

def explore_branching_factors(factors):
    return list(_generator_explore_branching_factors(factors, prefix = ()))

def _generator_explore_branching_factors(factors, prefix):
    yield prefix
    if len(factors) > 0:
        first = factors[0]
        rest = factors[1:]
        for i in range(first):
            new_prefix = prefix + (i,)
            for path in _generator_explore_branching_factors(rest, new_prefix):
                yield path

def niceprint_str(X, precision = 4):
    fmt = "{{:.{}f}}".format(precision)
    formatter = dict(float = lambda x: fmt.format(x))
    result = np.array2string(X, max_line_width=1000, formatter=formatter)
    return result

def niceprint(*args, **kwargs):
    print(niceprint_str(*args, **kwargs))
