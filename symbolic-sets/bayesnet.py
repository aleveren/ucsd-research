def index_over(var_name, set_name, subscript = None):
    return SymbolicIndex(var_name = var_name, set_name = set_name, subscript = subscript)

class SymbolicIndex(object):
    def __init__(self, var_name, set_name, subscript):
        if subscript is None:
            subscript = ()
        self.var_name = var_name
        self.set_name = set_name
        self.subscript = subscript

    def __repr__(self):
        set_and_subscript = self.set_name
        if self.subscript:
            set_and_subscript += "[{}]".format(self.subscript)
        result = "index {} over {}".format(self.var_name, set_and_subscript)
        return result

    def __str__(self):
        return self.__repr__()

    @property
    def range_indexed_by(self):
        return self.set_name

class RandomVariable(object):
    def __init__(self, name, distribution, subscript):
        if subscript is None:
            subscript = ()
        self.name = name
        self.distribution = distribution
        self.subscript = subscript

    def __getitem__(self, indices):
        #print("Got called with indices = {}".format(indices))
        if indices is None:
            indices = ()
        elif isinstance(indices, SymbolicIndex):
            indices = (indices,)
        elif not isinstance(indices, tuple):
            indices = (indices,)
        #print("### subscript = {}".format(self.subscript))
        #print("### indices = {}".format(indices))
        for sub, ix in zip(self.subscript, indices):
            #print("processing {} <--> {}".format(sub, ix))
            assert hasattr(ix, "range_indexed_by")
            assert sub.range_indexed_by == ix.range_indexed_by, \
                "Mismatch between {} and {}".format(sub.range_indexed_by, ix.range_indexed_by)
        return DereferencedRV(self, indices)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

    @property
    def range_indexed_by(self):
        return self.distribution.range_indexed_by

class DereferencedRV(object):
    # TODO: consider making this a subclass of RandomVariable?

    def __init__(self, random_var, indices):
        self.random_var = random_var
        self.indices = indices

    def __repr__(self):
        str_indices = ", ".join(str(x) for x in self.indices)
        return "{}[{}]".format(self.random_var.name, str_indices)

    def __str__(self):
        return self.__repr__()

    @property
    def distribution(self):
        return self.random_var.distribution

    @property
    def range_indexed_by(self):
        return self.distribution.range_indexed_by

class Model(object):
    def __init__(self):
        self.params = dict()
        self.random_vars = []

    def add_parameter(self, name, indexed_by = None, subscript = None):
        distrib = ConstantDistrib(indexed_by = indexed_by)
        return self.add_random_variable(name = name, distribution = distrib, subscript = subscript)

    def add_random_variable(self, name, distribution, subscript = None):
        rv = RandomVariable(name = name, distribution = distribution, subscript = subscript)
        self.random_vars.append(rv)
        return rv

    def log_joint_probability(self):
        # TODO: add param ignore_constants_relative_to (default: None)

        def create_term(rv):
            return ("TODO_term_for", rv)

        result = None
        for rv in self.random_vars:
            term = create_term(rv)
            assert term is not None
            if term == 0:
                continue
            if result is None:
                result = term
            elif isinstance(result, tuple) and result[0] == "add":
                result = ("add",) + result[1:] + (term,)
            else:
                result = ("add", result, term)
        if result is None:
            result = 0
        return result

class Indicator(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return "Indicator[{} == {}]".format(self.a, self.b)

    def __str__(self):
        return self.__repr__()

_dummy_count = 0
def dummy_index_over(idx):
    global _dummy_count
    var_name = "dummy{}".format(_dummy_count)
    _dummy_count += 1
    return index_over(var_name = var_name, set_name = idx.set_name, subscript = idx.subscript)

###################
# Distributions

class ConstantDistrib(object):
    def __init__(self, indexed_by = None):
        if indexed_by is None:
            indexed_by = ()
        self.indexed_by = indexed_by

    def log_probability(self, value):
        return 0  # TODO: should this use an indicator function?

class CategoricalDistrib(object):
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def log_probability(self, value):
        idx = self.probabilities.indexed_by
        dummy_idx = dummy_index_over(idx)
        return ("sum", dummy_idx,
            ("multiply",
                ("indicator", value, dummy_idx.name),
                ("log", ("index", probabilities, dummy_idx.name))))

class DeterministicDistrib(object):
    def __init__(self, function, args, range_indexed_by = None):
        self.function = function
        self.args = args
        self.range_indexed_by = range_indexed_by

    def log_probability(self, value):
        # TODO: FIXME - probably shouldn't use self.args directly?
        dummy_idx = dummy_index_over(self.range_indexed_by)
        return ("sum", dummy_idx,
            ("indicator", (self.function,) + tuple(self.args), value))

class DirichletDistrib(object):
    def __init__(self, alphas):
        self.alphas = alphas

    def log_probability(self, value):
        idx = self.alphas.indexed_by
        dummy_idx = dummy_index_over(idx)
        return ("add",
            ("log_gamma", ("sum", dummy_idx, ("index", self.alphas, dummy_idx.name))),
            ("multiply", -1, ("sum", dummy_idx, ("log_gamma", ("index", self.alphas, dummy_idx.name)))),
            ("sum", dummy_idx, ("multiply",
                ("add", ("index", self.alphas, dummy_idx.name), -1),
                ("log", ("index", value, dummy_idx.name)))))
