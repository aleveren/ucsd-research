from collections import namedtuple

def index_over(name, subscript = None):
    if subscript is None:
        subscript = ()
    return SymbolicIndex(name = name, subscript = subscript)

SymbolicIndex = namedtuple("SymbolicIndex", ["name", "subscript"])

class RandomVariable(object):
    def __init__(self, name, distribution, subscript):
        if subscript is None:
            subscript = ()
        self.name = name
        self.distribution = distribution

    def __getitem__(self, index):
        print("Got called with index = {}".format(index))
        if index is None:
            index = ()
        elif not isinstance(index, tuple):
            index = (index,)
        print("Now index = {}".format(index))

class Model(object):
    def __init__(self):
        self.params = dict()
        self.random_vars = dict()

    def add_parameter(self, name, indexed_by = None, subscript = None):
        distrib = ConstantDistrib(indexed_by = indexed_by)
        return self.add_random_variable(name = name, distribution = distrib, subscript = subscript)

    def add_random_variable(self, name, distribution, subscript = None):
        rv = RandomVariable(name = name, distribution = distribution, subscript = subscript)
        self.random_vars[name] = rv
        return rv

    def log_joint_probability(self, ignore_constants_relative_to = None):
        pass

###################
# Distributions

class ConstantDistrib(object):
    def __init__(self, indexed_by = None):
        if indexed_by is None:
            indexed_by = ()
        self.indexed_by = indexed_by

class CategoricalDistrib(object):
    def __init__(self, probabilities):
        self.probabilities = probabilities

class DeterministicDistrib(object):
    def __init__(self, function, args, range_indexed_by = None):
        self.function = function
        self.args = args
        self.range_indexed_by = None

class DirichletDistrib(object):
    def __init__(self, alphas):
        self.alphas = alphas
