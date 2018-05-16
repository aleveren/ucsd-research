from collections import namedtuple
from contextlib import contextmanager
import copy

import numpy as np
import networkx as nx

class Model(object):
    def __init__(self):
        self.distribs = dict()
        self.signatures = dict()
        self.active_for_loop_variables = []

    def add_random_variable(self, var, distrib):
        assert isinstance(var, tuple)
        var_name = var[0]
        signature = len(var) - 1  # For now, signature is just # of indices following the var name
        if var_name in self.signatures:
            expected = self.signatures[var_name]
            assert signature == expected, \
                "Inconsistent signature for {}: {} vs {}".format(var_name, signature, expected)
        else:
            self.signatures[var_name] = signature
        self.distribs[var_name] = distrib

        indices_in_rv = set([w for i, w in enumerate(var) if i > 0 and isinstance(w, ForLoopVariable)])
        active_indices = set(self.active_for_loop_variables)

        rv_index_suffix = ("[" + ", ".join(str(x) for x in var[1:]) + "]") if len(var) > 1 else ""
        assert indices_in_rv <= active_indices, \
            "Out-of-scope indices in definition of '{}{}': {}".format(var[0], rv_index_suffix, indices_in_rv - active_indices)
        assert active_indices <= indices_in_rv, \
            "Unused indices in definition of '{}{}': {}".format(var[0], rv_index_suffix, active_indices - indices_in_rv)

    def active_iter_names(self):
        return set([x.iter_name for x in self.active_for_loop_variables])

    def loop_over(self, iter_name, set_name):
        return for_loop_variable_context(self, iter_name, set_name)

    def generate_data(self, output_vars, placeholders = None, sets = None, mappings = None):
        sampler = Sampler(model = self, placeholders = placeholders, sets = sets, mappings = mappings)
        return sampler.sample(output_vars = output_vars, return_cache = return_cache)

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
        output_vars_expanded = set()
        for var in output_vars:
            output_vars_expanded |= expand_rv_index_sets(var, self.sets)

        cache = dict()
        data = dict()
        for var in output_vars_expanded:
            self.recursive_sample(var)
            data[var] = self.cache[var]
        return data

    def recursive_sample(self, var):
        '''Sample `var`.  If necessary, first sample from its dependencies.'''
        if var not in cache:
            distrib = self.get_sampleable_distrib(var)
            self.cache[var] = sample(distrib)

    def get_sampleable_distrib(self, var):

        pass  # TODO

def prepare_to_sample(var, distrib, mappings, sets, data):
    if isinstance(distrib, Deterministic) and isinstance(distrib.func, SymbolicMapping):
        distrib = distrib._replace(func = mappings[distrib.func.name])
    distrib = substitute_rvs(distrib, data, sets)
    return distrib

def sample(distrib):
    if isinstance(distrib, Constant):
        return copy.deepcopy(distrib.value)
    elif isinstance(distrib, Dirichlet):
        return np.random.dirichlet(distrib.alpha)
    elif isinstance(distrib, Categorical):
        return np.random.choice(len(distrib.probs), p=distrib.probs)
    elif isinstance(distrib, Deterministic):
        return distrib.func(*distrib.args)
    elif isinstance(distrib, ConstantPlaceholder):
        raise ValueError("Internal error: found ConstantPlaceholder: {}".format(distrib))
    else:
        raise ValueError("Unrecognized distribution: {}".format(distrib))

class UnspecifiedRandomVariable(object):
    pass

def expand_rv(var, distrib, sets):
    result = {var: distrib}
    for i, idx in enumerate(var):
        if isinstance(idx, ForLoopVariable):
            set_to_expand = sets[idx.set_name]
            expanded_result = dict()
            for cur_var, cur_distrib in result.items():
                for val in set_to_expand:
                    # Replace idx -> val in cur_var and distrib
                    mod_cur_var = substitute(cur_var, idx, val)
                    mod_distrib = substitute(distrib, idx, val)
                    expanded_result[mod_cur_var] = mod_distrib
            result = expanded_result
    return result

def expand_placeholders(var, distrib, placeholders):
    if isinstance(distrib, ConstantPlaceholder):
        return Constant(placeholders[var])
    else:
        return distrib

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

def get_dependencies(X, sets):
    if isinstance(X, tuple) and not hasattr(X, "_make"):
        # Plain tuple (not namedtuple)
        return expand_rv_index_sets(X, sets)
    elif isinstance(X, tuple) or isinstance(X, list):
        result = set()
        for x in X:
            result |= get_dependencies(x, sets)
        return result
    else:
        return set()

def substitute_rvs(X, data, sets):
    if isinstance(X, tuple) and not hasattr(X, "_make") and X in data:
       # Plain tuple (not namedtuple) appearing in the data
       return data[X]
    elif isinstance(X, tuple):
        if hasattr(X, "_make"):
            newval = tuple(substitute_rvs(x, data, sets) for x in X)
            return X._make(newval)
        else:
            index_set_positions = [i for i, idx in enumerate(X) if isinstance(idx, IndexSet)]
            if len(index_set_positions) == 0:
                return tuple(substitute_rvs(x, data, sets) for x in X)
            else:
                intermed = expand_rv_index_sets_dict(X, sets)
                return substitute_rvs(intermed, data, sets)
    elif isinstance(X, list):
        return [substitute_rvs(x, data, sets) for x in X]
    elif isinstance(X, dict):
        return {k: substitute_rvs(v, data, sets) for k, v in X.items()}
    else:
        return X

def expand_rv_index_sets(var, sets):
    first_expansion_slot = None
    for i, v in enumerate(var):
        if isinstance(v, IndexSet):
            first_expansion_slot = i
            break

    if first_expansion_slot is None:
        return set([var])

    expanded = set()
    for val in sets[var[first_expansion_slot].name]:
        new_var = var[:first_expansion_slot] + (val,) + var[first_expansion_slot+1:]
        expanded |= expand_rv_index_sets(new_var, sets)

    return expanded

def expand_rv_index_sets_dict(var, sets):
    expansion_slot = None
    for i in range(len(var)):
        if isinstance(var[i], IndexSet):
            expansion_slot = i
            break

    if expansion_slot is None:
        return var

    expanded = dict()
    idx_set_name = var[expansion_slot].name
    for val in sets[idx_set_name]:
        new_var = var[:expansion_slot] + (val,) + var[expansion_slot+1:]
        expanded[val] = expand_rv_index_sets_dict(new_var, sets)

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

def lookup(var_collection, indices):
    result = var_collection
    for idx in indices:
        result = result[idx]
    return result

def DeterministicLookup(var, indices):
    return Deterministic(lookup, args = [var, indices])
