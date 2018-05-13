import bayesnet as bn
import networkx as nx

# Pachinko Allocation Model (PAM)

# This is a semi-constrained version of the PAM -- not as constrained as the
# 3-layer or 4-layer PAM, since it allows an arbitrary number of layers
# ("N-layer PAM").  However, it does not yet allow an arbitrary DAG structure.

# Define a directed acyclic graph of topics, with exactly one source (the root), and with all leaves at the same depth (?)
g = nx.DiGraph()
g.add_edges_from([
  (0, 1),
  (0, 2),
  (1, 3),
  (1, 4),
  (1, 5),
  (2, 3),
  (2, 4),
  (2, 5),
])
g.graph["root"] = 0

bottom_layer_shared_across_corpus = True

def all_paths(g, start=None, prefix=()):
  if start is None:
    start = g.graph["root"]
  assert nx.algorithms.is_directed_acyclic_graph(g)
  pass # TODO

def depth(g):
  pass # TODO

def convert_path_to_node(g, path):
  pass # TODO

def num_children(g, node):
  pass # TODO

m = bn.Model()
prior_params = dict()
for path in all_paths(g):
  nc = num_children(g, convert_path_to_node(g, path))
  if nc == 0:
    prior_params[path] = bn.vector(("alpha", path), shape=(m.index_set("vocab"),))
  else:
    prior_params[path] = bn.vector(("alpha", path), shape=(nc,))
documents_by_word_slot = bn.symbolic_mapping("word_slots", "documents")

if bottom_layer_shared_across_corpus:
  for path in all_paths(g):
    if num_children(g, convert_path_to_node(path)) == 0:
      m.add_random_variable(("theta", path), bn.Dirichlet(prior_params[path]))

with m.for_loop_variable("documents") as d:
  for path in all_paths(g):
    if num_children(g, convert_path_to_node(path)) > 0 or not bottom_layer_shared_across_corpus:
      m.add_random_variable(("theta", d, path), bn.Dirichlet(prior_params[path]))

with m.for_loop_variable("word_slots") as w:
  m.add_random_variable(("r", w, 0), bn.Constant(()))
  m.add_random_variable(("current_doc", w), bn.Deterministic(documents_by_word_slot, args=[w])
  for i in range(depth(g)):
    if i == depth(g) - 1 and bottom_layer_shared_across_corpus:
      m.add_random_variable(("current_theta"), w, i), bn.DeterministicLookup("theta", ("r", w, i))
    else:
      m.add_random_variable(("current_theta", w, i), bn.DeterministicLookup("theta", ("current_doc", w), ("r", w, i)))
    m.add_random_variable(("z", w, i), bn.Multinomial(("current_theta", w, i)))
    m.add_random_variable(("r", w, i+1), bn.Deterministic(lambda xs, x: xs + [x], args = [("r", w, i), ("z", w, i)]))
  m.add_random_variable(("current_theta", w, depth(g)), bn.DeterministicLookup("theta", ("current_doc", w), ("r", w, depth(g))))
  m.add_random_variable(("t", w), bn.Multinomial(("current_theta", w, depth(g))))  # <-- observed data



# Operations we might want to perform on the model, m:
# 1. Symbolic expressions for log-joint probability, complete-conditional (or collapsed complete-conditional) probability
# 2. Given observed dataset, infer hidden variables (ie, "train the model") using automatically derived Gibbs sampling or stochastic variational inference



# TODO: how to get a symbolic while-loop or symbolic if-statement?  (needed for less constrained graph structures, like hPAM model 2?)
