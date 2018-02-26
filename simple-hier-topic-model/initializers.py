import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import issparse

from simple_hierarchical_topic_model import softmax


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
