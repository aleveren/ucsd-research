from collections import namedtuple
from contextlib import contextmanager
import copy

import numpy as np
import networkx as nx

#import warnings
#warnings.filterwarnings('error')

class Model(object):
    def __init__(self):
        self.distribs = dict()
        self.signatures = dict()
        self.active_for_loop_variables = []

    def add_random_variable(self, var, distrib):
        assert isinstance(var, tuple)
        var_name = var[0]
        signature = Signature.from_spec(var)

        if var_name in self.signatures:
            expected = self.signatures[var_name]
            assert signature == expected, \
                "Inconsistent signature for {}: {} vs {}".format(var_name, signature, expected)
        else:
            self.signatures[var_name] = signature

        key, _ = signature.get_key_and_substitutions(var)
        self.distribs[key] = distrib

        indices_in_rv = set([w for i, w in enumerate(var) if i > 0 and isinstance(w, ForLoopVariable)])
        active_indices = set(self.active_for_loop_variables)

        rv_index_suffix = ("[" + ", ".join(str(x) for x in var[1:]) + "]") if len(var) > 1 else ""
        assert indices_in_rv <= active_indices, \
            "Out-of-scope indices in definition of '{}{}': {}".format(var[0], rv_index_suffix, indices_in_rv - active_indices)
        assert active_indices <= indices_in_rv, \
            "Unused indices in definition of '{}{}': {}".format(var[0], rv_index_suffix, active_indices - indices_in_rv)

    def lookup_distrib(self, var):
        var_name = var[0]
        signature = self.signatures[var_name]
        key, substitutions = signature.get_key_and_substitutions(var)
        result = self.distribs[key]
        for sub in substitutions:
            result = substitute(result, sub[0], sub[1])
        return result

    def matches_with_substitution(self, var, compare_var):
        if len(var) != len(compare_var):
            return None
        substitutions = []
        for i in range(len(var)):
            v = var[i]
            cv = compare_var[i]
            if (is_numpy_type(v) or is_numpy_type(cv)) and np.array_equal(v, cv):
                continue
            elif not is_numpy_type(v) and not is_numpy_type(cv) and v == cv:
                continue
            elif isinstance(cv, ForLoopVariable) and not isinstance(v, ForLoopVariable):
                substitutions.append((cv, v))
            else:
                return None
        return substitutions

    def active_iter_names(self):
        return set([x.iter_name for x in self.active_for_loop_variables])

    def loop_over(self, iter_name, set_name):
        return for_loop_variable_context(self, iter_name, set_name)

    def generate_data(self, output_vars, return_sampler = False, placeholders = None, sets = None, mappings = None):
        sampler = Sampler(model = self, placeholders = placeholders, sets = sets, mappings = mappings)
        result = sampler.sample(output_vars = output_vars)
        if return_sampler:
            return result, sampler
        else:
            return result

class Signature(namedtuple("Signature", ["for_loop_variables"])):
    def get_key_and_substitutions(self, subscripts):
        assert len(subscripts) == len(self.for_loop_variables)
        key = []
        substitutions = []
        for i, v in enumerate(self.for_loop_variables):
            s = subscripts[i]
            if v is None:
                key.append(s)
            else:
                key.append(v)  # NOTE: key uses fixed value v from for_loop_variables instead of s
                substitutions.append((v, s))
        return tuple(key), substitutions

    @classmethod
    def from_spec(cls, spec):
        for_loop_variables = tuple(s if isinstance(s, ForLoopVariable) else None for s in spec)
        return cls(for_loop_variables = for_loop_variables)

class Sampler(object):
    def __init__(self, model, placeholders, sets, mappings):
        self.model = model
        self.placeholders = placeholders
        self.sets = sets
        self.mappings = mappings
        self.cache = dict()

    def reset(self):
        self.cache.clear()

    def sample(self, output_vars):
        assert isinstance(output_vars, list)
        output_vars_expanded = []
        for var in output_vars:
            output_vars_expanded.extend(expand_rv_index_sets(var, self.sets))

        cache = dict()
        data = dict()
        for var in output_vars_expanded:
            data[var] = self.recursive_sample(var)
        return data

    def recursive_sample(self, var):
        '''Sample `var`.  If necessary, first sample from its dependencies.'''
        if var in self.cache:
            return self.cache[var]

        distrib = self.model.lookup_distrib(var)

        if isinstance(distrib, Constant):
            if is_simple_tuple(distrib.value):
                value = self.recursive_sample(distrib.value)
            else:
                value = distrib.value
            result = copy.deepcopy(value)
        elif isinstance(distrib, ConstantPlaceholder):
            result = self.placeholders[var]
        elif isinstance(distrib, Dirichlet):
            if is_simple_tuple(distrib.alpha):
                alpha = self.recursive_sample(distrib.alpha)
            else:
                alpha = distrib.alpha
            result = np.random.dirichlet(alpha)
        elif isinstance(distrib, Categorical):
            if is_simple_tuple(distrib.probs):
                probs = self.recursive_sample(distrib.probs)
            else:
                probs = distrib.probs
            result = np.random.choice(len(probs), p=probs)
        elif isinstance(distrib, Deterministic):
            if isinstance(distrib.func, SymbolicMapping):
                func = self.mappings[distrib.func.name]
            else:
                func = distrib.func
            args = []
            for arg in distrib.args:
                if is_simple_tuple(arg):
                    arg = self.recursive_sample(arg)
                args.append(arg)
            result = func(*args)
        elif isinstance(distrib, DeterministicLookup):
            indices = []
            for idx in distrib.indices:
                if is_simple_tuple(idx):
                    idx = self.recursive_sample(idx)
                indices.append(idx)
            indirect_var = (distrib.var,) + tuple(indices)
            self.recursive_sample(indirect_var)
            result = self.cache[indirect_var]
        else:
            raise ValueError("Unrecognized distribution: {}".format(distrib))

        self.cache[var] = result
        return result


def is_simple_tuple(X):
    '''Return whether X is a tuple but not a namedtuple'''
    return isinstance(X, tuple) and not hasattr(X, "_make")

def is_numpy_type(X):
    return isinstance(X, (np.ndarray, np.generic))

def substitute(X, src, dest):
    if X == src:
        return dest
    elif isinstance(X, tuple):
        if hasattr(X, "_make"):
            newval = substitute(tuple(X), src, dest)
            return X._make(newval)
        else:
            return tuple(substitute(x, src, dest) for x in X)
    elif isinstance(X, list):
        return [substitute(x, src, dest) for x in X]
    else:
        return X

def expand_rv_index_sets(var, sets):
    first_expansion_slot = None
    for i, v in enumerate(var):
        if isinstance(v, IndexSet):
            first_expansion_slot = i
            break

    if first_expansion_slot is None:
        return [var]

    expanded = []
    for val in sets[var[first_expansion_slot].name]:
        new_var = var[:first_expansion_slot] + (val,) + var[first_expansion_slot+1:]
        expanded.extend(expand_rv_index_sets(new_var, sets))

    return expanded

@contextmanager
def for_loop_variable_context(model, iter_name, set_name):
    spec = ForLoopVariable(iter_name, set_name)
    assert spec.iter_name not in model.active_iter_names(), \
        "Cannot reuse iter_name '{}'".format(spec.iter_name)
    model.active_for_loop_variables.append(spec)
    yield spec
    model.active_for_loop_variables.pop()

class ForLoopVariable(namedtuple("ForLoopVariable", ["iter_name", "set_name"])):
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "for {} in {}".format(self.iter_name, self.set_name)

IndexSet = namedtuple("IndexSet", ["name"])
Constant = namedtuple("Constant", ["value"])
Dirichlet = namedtuple("Dirichlet", ["alpha"])
Categorical = namedtuple("Categorical", ["probs"])
Deterministic = namedtuple("Deterministic", ["func", "args"])
ConstantPlaceholder = namedtuple("ConstantPlaceholder", ["shape"])
SymbolicMapping = namedtuple("SymbolicMapping", ["domain", "codomain", "name"])
DeterministicLookup = namedtuple("DeterministicLookup", ["var", "indices"])
