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

        def get_terms(rv):
            value = ("index", rv, rv.subscript) if rv.subscript else rv
            result = rv.distribution.log_probability(value)
            if result == 0:
                result = []
            elif isinstance(result, tuple) and result[0] == "add":
                result = list(result[1:])
            else:
                result = [result]
            remaining_subscripts = rv.subscript
            while len(remaining_subscripts) > 0:
                sub = remaining_subscripts[-1]
                result = [("sum", sub, term) for term in result]
                remaining_subscripts = remaining_subscripts[:-1]
            return result

        terms = []
        for rv in self.random_vars:
            terms.extend(get_terms(rv))
        if len(terms) == 0:
            return 0
        else:
            return ("add",) + tuple(terms)

class Indicator(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __repr__(self):
        return "Indicator[{} == {}]".format(self.a, self.b)

    def __str__(self):
        return self.__repr__()

_dummy_count = 0
def dummy_var():
    global _dummy_count
    var_name = "dummy{}".format(_dummy_count)
    _dummy_count += 1
    return var_name

def pretty_print(obj):
    print(pretty_print_str(obj))

def pretty_print_str(obj, indent = 0):
    def simple(x):
        # TODO: handle namedtuples differently?
        return not isinstance(x, tuple)
    if simple(obj) or all(simple(x) for x in obj):
        return " " * indent + str(obj)
    else:
        first_line = str(obj[0])
        remaining_lines = [pretty_print_str(x, indent = indent + 2) for x in obj[1:]]
        return "{}({},\n{}\n{})".format(
            " " * indent, first_line, ",\n".join(remaining_lines), " " * indent)

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
        try:
            indexed_by = self.probabilities.distribution.indexed_by
        except:  # TODO: REMOVE
            raise Exception("Tried to get indexed_by from {}".format(self.probabilities.distribution))
        dummy_idx = index_over(dummy_var(), indexed_by)
        return ("sum", dummy_idx,
            ("multiply",
                ("indicator", value, dummy_idx.var_name),
                ("log", ("index", self.probabilities, dummy_idx.var_name))))

class DeterministicDistrib(object):
    def __init__(self, function, args, range_indexed_by = None):
        self.function = function
        self.args = args
        self.range_indexed_by = range_indexed_by

    def log_probability(self, value):
        # TODO: FIXME - probably shouldn't use self.args directly?
        dummy_idx = index_over(dummy_var(), self.range_indexed_by)
        return ("sum", dummy_idx,
            ("indicator", (self.function,) + tuple(self.args), value))

class DirichletDistrib(object):
    def __init__(self, alphas):
        self.alphas = alphas

    def log_probability(self, value):
        dummy_idx = index_over(dummy_var(), self.indexed_by)
        return ("add",
            ("log_gamma", ("sum", dummy_idx, ("index", self.alphas, dummy_idx.var_name))),
            ("multiply", -1, ("sum", dummy_idx, ("log_gamma", ("index", self.alphas, dummy_idx.var_name)))),
            ("sum", dummy_idx, ("multiply",
                ("add", ("index", self.alphas, dummy_idx.var_name), -1),
                ("log", ("index", value, dummy_idx.var_name)))))

    @property
    def indexed_by(self):
        return self.alphas.distribution.indexed_by
