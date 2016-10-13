import unittest
import numpy as np

from context import hier_clust


class Tests(unittest.TestCase):
    def test_similarity_sparse(self):
        # Test case where sparseness makes a difference
        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0)
        d = [[0,0], [1,0], [0,2], [1,2], [5,5], [5,5]]
        sim1 = hc._get_similarity(d, sparse = 'auto')
        sim2 = hc._get_similarity(d, sparse = 'never')
        x = np.exp(-0.5)
        expected = np.array([
            [1, x, 0, 0, 0, 0],
            [x, 1, 0, 0, 0, 0],
            [0, 0, 1, x, 0, 0],
            [0, 0, x, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1]])
        assert np.allclose(sim1.todense(), expected)
        assert sim1.nnz == 12
        assert not np.allclose(sim1.todense(), sim2)
        assert not np.allclose(sim2, expected)

    def test_similarity_sparse_mutual(self):
        # Test case where mutual KNN makes a difference
        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0)
        d = [[0,0], [1,0], [0,2]]
        sim1 = hc._get_similarity(d, sparse = 'auto')

        x = np.exp(-0.5)
        expected = np.array([
            [1, x, 0],
            [x, 1, 0],
            [0, 0, 1]])
        assert np.allclose(sim1.todense(), expected)
        assert sim1.nnz == 5

        # Recompute similarities, allowing non-mutual neighbors
        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0,
            mutual_neighbors = False)
        sim2 = hc._get_similarity(d, sparse = 'auto')

        x = np.exp(-0.5)
        y = 0.5 * np.exp(-2)
        expected = np.array([
            [1, x, y],
            [x, 1, 0],
            [y, 0, 1]])
        assert np.allclose(sim2.todense(), expected)
        assert sim2.nnz == 7

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
