import bayesnet as bn

print("=" * 20)

m = bn.Model()
a = bn.IndexSet("a", "a_domain")
x = m.add_parameter("x", components_indexed_by = a)
y = m.add_random_variable("y", distribution = bn.CategoricalDistrib(x))
bn.pretty_print(m.log_joint_probability())

print("=" * 20)

m = bn.Model()
a = bn.IndexSet("a", "a_domain")
b = bn.IndexSet("b", "b_domain")
x = m.add_parameter("x", components_indexed_by = a)
y = m.add_random_variable("y", distribution = bn.CategoricalDistrib(x), collection_indexed_by = b)
bn.pretty_print(m.log_joint_probability())

print("=" * 20)

m = bn.Model()

nodes = bn.IndexSet("r", "nodes")
depths = bn.IndexSet("k", "depths")
leaves = bn.IndexSet("l", "leaves")
vocab = bn.IndexSet("v", "vocab")
documents = bn.IndexSet("d", "documents")
word_slots = bn.IndexSet("n", "word_slots", collection_indexed_by = (documents,))

alpha_phi = m.add_parameter("alpha_phi", components_indexed_by = depths)
alpha_lam = m.add_parameter("alpha_lam", components_indexed_by = leaves)
alpha_theta = m.add_parameter("alpha_theta", components_indexed_by = vocab)

theta = m.add_random_variable("theta", collection_indexed_by = (nodes,),
    distribution = bn.DirichletDistrib(alpha_theta))

phi = m.add_random_variable("phi", collection_indexed_by = (documents,),
    distribution = bn.DirichletDistrib(alpha_phi))
lam = m.add_random_variable("lam", collection_indexed_by = (documents,),
    distribution = bn.DirichletDistrib(alpha_lam))

z = m.add_random_variable("z", collection_indexed_by = (documents, word_slots),
    distribution = bn.CategoricalDistrib(phi[documents]))
l = m.add_random_variable("l", collection_indexed_by = (documents, word_slots),
    distribution = bn.CategoricalDistrib(lam[documents]))
node_selected = m.add_random_variable("node_selected",
    collection_indexed_by = (documents, word_slots),
    distribution = bn.DeterministicDistrib(
        function = "node_by_depth_and_leaf",  # purely symbolic functional dependence?
        args = [z[documents, word_slots], l[documents, word_slots]],
        range_indexed_by = nodes))
t = m.add_random_variable("t", collection_indexed_by = (documents, word_slots),
    distribution = bn.CategoricalDistrib(theta[node_selected[documents, word_slots]]))

log_joint = m.log_joint_probability()
bn.pretty_print(log_joint)
print(bn.to_latex(log_joint))
#bn.pretty_print(m.log_joint_probability(
#    ignore_constants_relative_to = [phi[d]]))
#bn.pretty_print(m.log_joint_probability(
#    ignore_constants_relative_to = [z[d, n]]))

# Q: is it OK to reuse indices (d, n) like this?  seems like it could yield a naming clash...

# TODO: for variational inference (conjugate exponential models only?), try searching for 
