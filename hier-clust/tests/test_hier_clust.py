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

    def test_clust_large_data(self):
        theta = np.linspace(0, 2*np.pi, 20)
        points = []
        for x in [0, 5, 10, 15, 20]:
            for th in theta:
                points.append(np.array([[x + np.cos(th), np.sin(th)]]))
        data = np.vstack(points)
        assert data.shape == (100, 2)
        hc = hier_clust.HierClust(
            n_neighbors = 10,
            threshold_for_subset = 50,
            sigma_similarity = 1.0)
        tree, assignments = hc.fit(data)
        assert len(tree.data["orig_indices"]) == 100
        assert len(assignments) == 100
