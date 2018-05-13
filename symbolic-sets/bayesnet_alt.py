from collections import namedtuple
from contextlib import contextmanager

class Model(object):
    def __init__(self):
        self.vars_and_distribs = []
        self.active_for_loop_variables = []

    def add_random_variable(self, var, distrib):
        assert isinstance(var, tuple)
        self.vars_and_distribs.append((var, distrib))

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

    def generate_data(self, placeholders = None, sets = None, mappings = None):
        data = dict()
        pass  # TODO
        return data

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
DeterministicLookup = namedtuple("DeterministicLookup", ["var", "indices"])
Placeholder = namedtuple("Placeholder", ["shape"])
SymbolicMapping = namedtuple("SymbolicMapping", ["domain", "codomain", "name"])
