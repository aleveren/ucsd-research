import unittest
import numpy as np

from context import hier_clust


class Tests(unittest.TestCase):
    def test_similarity(self):
        hc = hier_clust.HierClust(n_neighbors = 3, sigma_similarity = 1.0)
        d = [[1,0,0],[0,1,0],[0,0,1]]
        sim1 = hc._get_sparse_similarity(d).todense()
        sim2 = hc._get_dense_similarity(d)
        assert np.allclose(sim1, sim2)
