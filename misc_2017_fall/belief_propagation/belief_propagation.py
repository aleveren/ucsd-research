from __future__ import division, print_function

import numpy as np
from collections import namedtuple

########################
# Directed graph inference

RandomVariable = namedtuple("RandomVariable", [
    "name",
    "domain",
])

class QueryMixin(object):
    @property
    def variable_names(self):
        return list(self.variables.keys())

    def query_mixin(self, event, evidence = None):
        print("DEBUGGING: query(event = {}, evidence = {})".format(event, evidence))

        if evidence is not None:
            numer_event = dict()
            numer_event.update(event)
            for k, v in evidence.items():
                numer_event[k] = v
                if k in event and event[k] != v:
                    return 0  # Evidence contradicts the query event, so probability is 0
            numer = self.query(numer_event, evidence = None)
            denom = self.query(evidence, evidence = None)
            return numer / denom

        def query_recursive(event, unassigned_combinations):
            print("DEBUGGING: query_recursive(event = {}, unassigned_combinations = {})".format(event, unassigned_combinations))
            if len(unassigned_combinations) == 0:
                return self.joint_probability(event)

            first_var = list(unassigned_combinations.keys())[0]
            new_unassigned_combinations = dict()
            for k in unassigned_combinations.keys():
                if k != first_var:
                    new_unassigned_combinations[k] = unassigned_combinations[k]

            result = 0.0
            for val in self.variables[first_var].domain:
                new_event = dict()
                new_event.update(event)
                new_event[first_var] = val
                result += query_recursive(new_event, new_unassigned_combinations)
            return result

        nuisance_vars = set(self.variable_names) - set(event.keys())
        unassigned_combinations = {nv: self.variables[nv].domain for nv in nuisance_vars}

        return query_recursive(event, unassigned_combinations)

class DirectedGraphModel(namedtuple("DirectedGraphModel", [
        "tables"]), QueryMixin):
    @property
    def variables(self):
        result = dict()
        for t in self.tables:
            result[t.var.name] = t.var
            for p in t.parents:
                result[p.name] = p
        return result

    @property
    def table_by_name(self):
        result = dict()
        for t in self.tables:
            result[t.var.name] = t
        return result

    def joint_probability(self, event):
        assert set(event.keys()) == set(self.variable_names)
        result = 1.0
        for v in self.variable_names:
            d = {v: event[v]}
            t = self.table_by_name[v]
            for p in t.parents:
                d[p.name] = event[p.name]
            result *= t.query(d)
        return result

    def query(self, event, evidence = None):
        return self.query_mixin(event = event, evidence = evidence)

class ConditionalTable(object):
    def __init__(self, var, table, parents = None):
        self.parents = parents if parents is not None else []
        self.var = var
        self.table = table
        self.parent_names = [p.name for p in self.parents]
        first_key = list(self.table.keys())[0]
        self.key_order = None
        for key in self.table:
            if self.key_order is None:
                self.key_order = _get_var_names(key)
            else:
                assert self.key_order == _get_var_names(key), \
                    "Inconsistent keys in table definition"
        assert (set(self.parent_names) | set([self.var.name])) == set(self.key_order)

    def query(self, event):
        print("DEBUGGING: {}.query(event = {})".format(self, event))
        key = self._convert_to_assignment(event)
        print("DEBUGGING: key = {}".format(key))
        return self.table[key]

    def __str__(self):
        return "ConditionalTable(var = {}, parents = {}, table = {})".format(self.var.name, [p.name for p in self.parents], self.table)

    def __repr__(self):
        return self.__str__()

    def _convert_to_assignment(self, event):
        assert hasattr(event, "keys")
        key = []
        for var in self.key_order:
            key.append(var)
            key.append(event[var])
        return tuple(key)

def _get_var_names(key):
    result = []
    assert len(key) % 2 == 0
    for i in range(len(key) // 2):
        result.append(key[i * 2])
    return tuple(result)

def _get_var_values(key):
    result = []
    assert len(key) % 2 == 0
    for i in range(len(key) // 2):
        result.append(key[i * 2 + 1])
    return tuple(result)

########################
# Factor graph inference

class Factor(namedtuple("Factor", [
        "variables",
        "function",
        "cond_table"])):
    def evaluate(self, event):
        a = tuple([v.name for v in self.variables])
        b = self.cond_table.key_order
        assert a == b, "{} != {}".format(a, b)
        assert hasattr(event, "keys")
        return self.function(event)

    def __str__(self):
        return "Factor([{}])".format(", ".join(v.name for v in self.variables))

    def __repr__(self):
        return self.__str__()

class FactorGraph(namedtuple("FactorGraph", [
        "factors"]), QueryMixin):
    @property
    def variables(self):
        result = dict()
        for f in self.factors:
            for v in f.variables:
                result[v.name] = v
        return result

    def joint_probability(self, event):
        print("DEBUGGING: event = {}".format(event))
        assert hasattr(event, "keys")
        result = 1.0
        for f in self.factors:
            sub_event = {v.name: event[v.name] for v in f.variables}
            print("DEBUGGING: f = {}, sub_event = {}".format(f, sub_event))
            factor_result = f.evaluate(sub_event)
            print("DEBUGGING: factor_result = {}".format(factor_result))
            result *= factor_result
        return result

    def query(self, event, evidence = None):
        # TODO: replace with belief propagation
        return self.query_mixin(event = event, evidence = evidence)

    @classmethod
    def from_directed_graph_model(cls, model):
        factors = []
        tables = model.table_by_name
        for t in tables.values():
            variables = [model.variables[v] for v in t.key_order]
            current_factor = Factor(variables = variables, function = lambda x: t.query(x), cond_table = t)
            factors.append(current_factor)
        return cls(factors = factors)

########################
# Examples

def build_grass_network():
    r = RandomVariable("Rain", [0, 1])
    s = RandomVariable("Sprinkler", [0, 1])
    w = RandomVariable("GrassWet", [0, 1])
    network = DirectedGraphModel(tables = [
        ConditionalTable(var = r, table = {
            ("Rain", 0): 4./5,
            ("Rain", 1): 1./5,
        }),
        ConditionalTable(var = s, parents = [r], table = {
            ("Rain", 0, "Sprinkler", 0): 3./5,
            ("Rain", 0, "Sprinkler", 1): 2./5,
            ("Rain", 1, "Sprinkler", 0): 99./100,
            ("Rain", 1, "Sprinkler", 1): 1./100,
        }),
        ConditionalTable(var = w, parents = [r, s], table = {
            ("Rain", 0, "Sprinkler", 0, "GrassWet", 0): 1.,
            ("Rain", 0, "Sprinkler", 0, "GrassWet", 1): 0.,
            ("Rain", 1, "Sprinkler", 0, "GrassWet", 0): 1./5,
            ("Rain", 1, "Sprinkler", 0, "GrassWet", 1): 4./5,
            ("Rain", 0, "Sprinkler", 1, "GrassWet", 0): 1./10,
            ("Rain", 0, "Sprinkler", 1, "GrassWet", 1): 9./10,
            ("Rain", 1, "Sprinkler", 1, "GrassWet", 0): 1./100,
            ("Rain", 1, "Sprinkler", 1, "GrassWet", 1): 99./100,
        }),
    ])
    return network

def main():
    grass_network = build_grass_network()
    graph = FactorGraph.from_directed_graph_model(grass_network)

    # Example query: probability that it rained today, given that the grass is wet
    # P(Rain = 1 | GrassWet = 1)
    expected = 891./2491
    print("Expected result: {}".format(expected))

    #result1 = grass_network.query(event = {"Rain": 1}, evidence = {"GrassWet": 1})
    #print("Result 1:        {}".format(result1))

    print(graph)
    try:
        result2 = graph.query(event = {"Rain": 1}, evidence = {"GrassWet": 1})
    except Exception as e:
        import traceback
        import sys
        traceback.print_tb(e.__traceback__, file=sys.stdout)
        print(repr(e))
        result2 = -1.
    print("Result 2:        {}".format(result2))

    # TODO: reinstate
    #np.testing.assert_allclose(result1, expected)
    #np.testing.assert_allclose(result2, expected)

if __name__ == "__main__":
    main()
