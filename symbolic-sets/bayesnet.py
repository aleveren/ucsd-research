class IndexSet(object):
    def __init__(self, var_name, set_name, collection_indexed_by = None):
        if collection_indexed_by is None:
            collection_indexed_by = ()
        self.var_name = var_name
        self.set_name = set_name
        self.collection_indexed_by = collection_indexed_by

    def __repr__(self):
        set_and_subscript = self.set_name
        if self.collection_indexed_by:
            set_and_subscript += "[{}]".format(", ".join(str(x) for x in self.collection_indexed_by))
        result = "{} in {}".format(self.var_name, set_and_subscript)
        return result

    def __str__(self):
        return self.__repr__()

    def copy_with_fresh_variable(self):
        return IndexSet(
            var_name = dummy_var(prefix = self.var_name),
            set_name = self.set_name,
            collection_indexed_by = self.collection_indexed_by)

    @property
    def range_indexed_by(self):
        return self

class RandomVariableCollection(object):
    def __init__(self, name, distribution, collection_indexed_by, dereferenced_subscripts):
        if collection_indexed_by is None:
            collection_indexed_by = ()
        elif isinstance(collection_indexed_by, IndexSet):
            collection_indexed_by = (collection_indexed_by,)
        if dereferenced_subscripts is None:
            dereferenced_subscripts = ()
        self.name = name
        self.distribution = distribution
        self.collection_indexed_by = collection_indexed_by
        self.dereferenced_subscripts = dereferenced_subscripts

    def __getitem__(self, indices):
        if indices is None:
            indices = ()
        elif isinstance(indices, IndexSet):
            indices = (indices,)
        elif not isinstance(indices, tuple):
            indices = (indices,)

        if len(indices) == 0:
            return self
        elif len(self.collection_indexed_by) == 0:
            assert len(indices) == 1
            assert indices[0].range_indexed_by.set_name == self.components_indexed_by.set_name
            return RandomVariableCollection(
                name = self.name,
                distribution = self.distribution,
                collection_indexed_by = (),
                dereferenced_subscripts = self.dereferenced_subscripts + (indices[0].range_indexed_by.var_name,))
        else:
            first_ci = self.collection_indexed_by[0]
            first_i = indices[0]
            assert hasattr(first_i, "range_indexed_by")
            assert first_ci.range_indexed_by.set_name == first_i.range_indexed_by.set_name, \
                "Mismatch between {} and {}".format(first_ci.range_indexed_by.set_name, first_i.range_indexed_by.set_name)
            indirect = RandomVariableCollection(
                name = self.name,
                distribution = self.distribution,
                collection_indexed_by = self.collection_indexed_by[1:],
                dereferenced_subscripts = self.dereferenced_subscripts + (first_i.range_indexed_by.var_name,))
            return indirect[indices[1:]]

    def __repr__(self):
        if len(self.dereferenced_subscripts) == 0:
            subscript_suffix = ''
        else:
            subscript_suffix = "[{}]".format(", ".join(str(x) for x in self.dereferenced_subscripts))
        return self.name + subscript_suffix

    def __str__(self):
        return self.__repr__()

    @property
    def components_indexed_by(self):
        return self.distribution.components_indexed_by

    @property
    def range_indexed_by(self):
        return self.distribution.range_indexed_by

class Model(object):
    def __init__(self):
        self.params = dict()
        self.random_vars = []

    def add_parameter(self, name, components_indexed_by = None, collection_indexed_by = None):
        distrib = ConstantDistrib(components_indexed_by = components_indexed_by)
        return self.add_random_variable(name = name, distribution = distrib, collection_indexed_by = collection_indexed_by)

    def add_random_variable(self, name, distribution, collection_indexed_by = None):
        rv = RandomVariableCollection(
            name = name,
            distribution = distribution,
            collection_indexed_by = collection_indexed_by,
            dereferenced_subscripts = ())
        self.random_vars.append(rv)
        return rv

    def log_joint_probability(self):
        # TODO: add param ignore_constants_relative_to (default: None)

        def get_terms(rv):
            orig_collection_indexed_by = rv.collection_indexed_by
            if len(rv.collection_indexed_by) > 0:
                rv = rv[rv.collection_indexed_by]
            result = rv.distribution.log_probability(rv)
            if result == 0:
                result = []
            elif isinstance(result, tuple) and result[0] == "add":
                result = list(result[1:])
            else:
                result = [result]
            remaining_indices = orig_collection_indexed_by
            while len(remaining_indices) > 0:
                idx = remaining_indices[-1]
                result = [("sum", idx, term) for term in result]
                remaining_indices = remaining_indices[:-1]
            return result

        terms = []
        for rv in self.random_vars:
            terms.extend(get_terms(rv))
        if len(terms) == 0:
            return 0
        elif len(terms) == 1:
            return terms[0]
        else:
            return ("add",) + tuple(terms)

_dummy_count = 0
def dummy_var(prefix = "dummy"):
    global _dummy_count
    var_name = "{}{}".format(prefix, _dummy_count)
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
    def __init__(self, components_indexed_by = None):
        if components_indexed_by is None:
            components_indexed_by = ()
        self.components_indexed_by = components_indexed_by

    def log_probability(self, value):
        return 0  # TODO: should this use an indicator function?

class CategoricalDistrib(object):
    def __init__(self, probabilities):
        self.probabilities = probabilities

    def log_probability(self, value):
        components_indexed_by = self.probabilities.distribution.components_indexed_by
        dummy_idx = components_indexed_by.copy_with_fresh_variable()
        try:
            foo = self.probabilities[dummy_idx]
        except:
            raise Exception("DEBUGGING: self.probabilities = {} ({})".format(self.probabilities, type(self.probabilities)))
        return ("sum", dummy_idx,
            ("multiply",
                ("indicator", value, dummy_idx.var_name),
                ("log", self.probabilities[dummy_idx])))
                #("log", ("index", self.probabilities, dummy_idx.var_name))))

class DeterministicDistrib(object):
    def __init__(self, function, args, range_indexed_by = None):
        self.function = function
        self.args = args
        self.range_indexed_by = range_indexed_by

    def log_probability(self, value):
        # TODO: FIXME - probably shouldn't use self.args directly?
        dummy_idx = self.range_indexed_by.copy_with_fresh_variable()
        return ("sum", dummy_idx,
            ("indicator", (self.function,) + tuple(self.args), dummy_idx.var_name))

class DirichletDistrib(object):
    def __init__(self, alphas):
        self.alphas = alphas

    def log_probability(self, value):
        dummy_idx = self.components_indexed_by.copy_with_fresh_variable()
        return ("add",
            ("log_gamma", ("sum", dummy_idx, self.alphas[dummy_idx])),
            ("multiply", -1, ("sum", dummy_idx, ("log_gamma", self.alphas[dummy_idx]))),
            ("sum", dummy_idx, ("multiply",
                ("add", self.alphas[dummy_idx], -1),
                ("log", value[dummy_idx]))))

    @property
    def components_indexed_by(self):
        return self.alphas.distribution.components_indexed_by
