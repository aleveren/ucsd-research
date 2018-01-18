import bayesnet as bn

m = bn.Model()

alpha_phi = m.add_parameter("alpha_phi", indexed_by = "depths")
alpha_lam = m.add_parameter("alpha_lam", indexed_by = "leaves")
alpha_theta = m.add_parameter("alpha_theta", indexed_by = "nodes")

r = bn.index_over("nodes")
theta = m.add_random_variable("theta", subscript = (r,),
    distribution = bn.DirichletDistrib(alpha_theta))

d = bn.index_over("documents")
phi = m.add_random_variable("phi", subscript = (d,),
    distribution = bn.DirichletDistrib(alpha_phi))
lam = m.add_random_variable("lam", subscript = (d,),
    distribution = bn.DirichletDistrib(alpha_lam))

n = bn.index_over("word_slots", subscript = (d,))
z = m.add_random_variable("z", subscript = (d, n),
    distribution = bn.CategoricalDistrib(phi))
l = m.add_random_variable("l", subscript = (d, n),
    distribution = bn.CategoricalDistrib(lam))
r = m.add_random_variable("r", subscript = (d, n),
    distribution = bn.DeterministicDistrib(
        function = "node_by_depth_and_leaf",  # purely symbolic functional dependence?
        args = [z, l],
        range_indexed_by = "nodes"))
t = m.add_random_variable("t", subscript = (d, n),
    distribution = bn.CategoricalDistrib(theta[r[d, n]]))

print(m.log_joint_probability())
print(m.log_joint_probability(
    ignore_constants_relative_to = [phi[d]]))
print(m.log_joint_probability(
    ignore_constants_relative_to = [z[d, n]]))

# Q: is it OK to reuse indices (d, n) like this?  seems like it could yield a naming clash...

# TODO: for variational inference (conjugate exponential models only?), try searching for 
