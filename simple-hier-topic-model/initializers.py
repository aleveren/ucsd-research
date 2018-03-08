import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import issparse

from utils import softmax


class UniformInitializer(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def init_var_params(self, var, shape):
        if var == "DL": return np.random.uniform(self.low, self.high, shape)
        if var == "DD": return np.random.uniform(self.low, self.high, shape)
        if var == "DV": return np.random.uniform(self.low, self.high, shape)
        if var == "L": return softmax(np.random.uniform(self.low, self.high, shape), axis = -1)
        if var == "D": return softmax(np.random.uniform(self.low, self.high, shape), axis = -1)


class KMeansInitializer(object):
    '''
    An initializer based on the KMeans initialization procedure described in
    Paisley et al, "Nested Hierarchical Dirichlet Processes", Section 5.1
    '''

    def __init__(self, data, paths, low, high,
            kappa = 0.5, noise_factor = 100.0, subset_size = None,
            noise_proportion = 0.5, const_proportion = 0.5):
        self.data = data
        self.subset_size = subset_size
        self.paths = paths
        self.path_to_index = {p: i for i, p in enumerate(paths)}
        self.kappa = kappa
        self.noise_factor = noise_factor
        self.noise_proportion = noise_proportion
        self.const_proportion = const_proportion
        self.low = low
        self.high = high

    def init_var_params(self, var, shape):
        if var == "DL": return np.random.uniform(self.low, self.high, shape)
        if var == "DD": return np.random.uniform(self.low, self.high, shape)
        if var == "L": return softmax(np.random.uniform(self.low, self.high, shape), axis = -1)
        if var == "D": return softmax(np.random.uniform(self.low, self.high, shape), axis = -1)
        if var == "DV":
            num_docs = self.data.shape[1]
            if not self.subset_size:
                data_subset = self.data
            else:
                data_subset_mask = np.random.permutation([True for i in range(self.subset_size)] + [False for i in range(num_docs - self.subset_size)])
                data_subset = self.data[:, data_subset_mask]
            if issparse(data_subset):
                data_subset = np.asarray(data_subset.todense())
            data_subset = data_subset.astype('float').copy()
            data_subset /= data_subset.sum(axis=0, keepdims=True)
            init_DV = np.zeros(shape)
            self._helper(init_DV, data_subset, prefix=())
            return init_DV

    def _helper(self, init_DV, data_subset, prefix):
        vocab_size = self.data.shape[0]
        num_docs = self.data.shape[1]
        prefix_index = self.path_to_index[prefix]
        mean = data_subset.mean(axis = -1)
        noise = np.random.dirichlet(self.noise_factor * np.ones(vocab_size) / vocab_size)
        current_DV_param = num_docs * (self.kappa * mean + (1 - self.kappa) * (self.const_proportion / vocab_size + self.noise_proportion * noise))
        init_DV[prefix_index, :] = current_DV_param
        child_paths = [x for x in self.paths if x[:len(prefix)] == prefix and len(x) == len(prefix) + 1]
        if len(child_paths) > 0:
            data_subset -= mean[:, np.newaxis]
            data_subset = np.maximum(0, data_subset)
            data_subset /= data_subset.sum(axis=0, keepdims=True)
            kmeans = KMeans(n_clusters = len(child_paths))
            clustering = kmeans.fit_predict(data_subset.transpose())
            # TODO: switch to a KMeans implementation that allows us to use L1 distance
            for i, p in enumerate(child_paths):
                current_subset_mask = (clustering == i)
                current_subset = data_subset[:, current_subset_mask].copy()
                self._helper(init_DV, current_subset, prefix = p)


class CustomInitializer(object):
    """
    A custom initializer that can be used to "cheat" and nudge the results closer
    to the desired values, based on stats from the simulation.
    We can use this to verify that variational inference can find the desired value
    if it starts sufficiently close to it.
    """
    def __init__(self, sim, topics_noise = 0.0, permutation = None):
        self.sim = sim
        self.topics_noise = topics_noise
        self.permutation = permutation

    def init_var_params(self, var, shape):
        sim = self.sim
        if var == "L": return np.ones((sim.num_docs * sim.doc_length, sim.num_leaves))
        if var == "D": return np.ones((sim.num_docs * sim.doc_length, sim.num_depths))
        if var == "DL": return np.broadcast_to(sim.alpha_leaves, (sim.num_docs, sim.num_leaves))
        if var == "DD": return np.broadcast_to(sim.alpha_depths, (sim.num_docs, sim.num_depths))
        if var == "DV":
            if self.permutation is not None:
                _permutation = self.permutation
            else:
                _permutation = np.arange(sim.num_nodes)
            init_DV = sim.topics_by_index[_permutation, :] + np.random.uniform(0.0, self.topics_noise, sim.topics_by_index.shape)
            return init_DV / np.min(init_DV)
