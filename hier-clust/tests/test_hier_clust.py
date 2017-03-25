import unittest
from mock import patch
import numpy as np
import pandas as pd

from context import hier_clust


class Tests(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def sim_data(self):
        theta = np.linspace(0, 2*np.pi, 20)
        points = []
        for x in [0, 5, 10, 15, 20]:
            for th in theta:
                points.append(np.array([[x + np.cos(th), np.sin(th)]]))
        data = np.vstack(points)
        assert data.shape == (100, 2)
        return data

    def test_similarity_sparse(self):
        # Test case where sparseness makes a difference
        d = [[0,0], [1,0], [0,2], [1,2], [5,5], [5,5]]

        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0,
            sparse_similarity = 'auto')
        dist1 = hc._get_distances(d)

        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0,
            sparse_similarity = 'never')
        dist2 = hc._get_distances(d)

        sim1 = hc._get_similarity(dist1)
        sim2 = hc._get_similarity(dist2)
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
        assert not np.allclose(sim1.todense(), sim2.todense())
        assert not np.allclose(sim2.todense(), expected)

    def test_similarity_sparse_mutual(self):
        # Test case where mutual KNN makes a difference
        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0,
            mutual_neighbors = True)
        d = [[0,0], [1,0], [0,2]]
        dist1 = hc._get_distances(d)
        sim1 = hc._get_similarity(dist1)

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
        dist2 = hc._get_distances(d)
        sim2 = hc._get_similarity(dist2)

        x = np.exp(-0.5)
        y = 0.5 * np.exp(-2)
        expected = np.array([
            [1, x, y],
            [x, 1, 0],
            [y, 0, 1]])
        assert np.allclose(sim2.todense(), expected)
        assert sim2.nnz == 7

    def test_cluster(self):
        data = self.sim_data()
        hc = hier_clust.HierClust(
            n_neighbors = 10,
            sigma_similarity = 'auto')
        tree, assignments = hc.fit(data)
        assert len(tree.data["orig_indices"]) == 100
        assert len(assignments) == 100

    def test_median(self):
        assert hier_clust.HierClust()._get_median([]) is None
        assert hier_clust.HierClust()._get_median([7]) == 7
        assert hier_clust.HierClust()._get_median([8, 7]) == 7
        assert hier_clust.HierClust()._get_median([6, 8, 7]) == 7

        xs = [3, 1, 5, 2, 6, 4, 9, 8, 7]
        assert hier_clust.HierClust()._get_median(xs) == 5

        xs = [3, 1, 5, 2, 6, 4, 8, 7]
        assert hier_clust.HierClust()._get_median(xs) == 4

        xs = [3, 2, 3, 1, 1, 1, 2, 3, 2]
        assert hier_clust.HierClust()._get_median(xs) == 2

        xs = [2, 1, 1, 3, 1, 1, 3, 1, 3]
        assert hier_clust.HierClust()._get_median(xs) == 1

        xs = [2, 3, 2, 3, 2, 3, 3, 1, 3]
        assert hier_clust.HierClust()._get_median(xs) == 3

    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.read_csv')
    def test_main(self, mock_read_csv, mock_to_csv):
        mock_read_csv.return_value = pd.DataFrame(self.sim_data(),
            columns = ['a1', 'a2'])

        hier_clust.main(['--input', 'bogus.csv'])
        assert not mock_to_csv.called

        hier_clust.main(['--input', 'bogus.csv', '--feature_columns', 'a.*',
            '--random_seed', '1', '--output', 'bogus_out.csv',
            '--constructor_json', '{"alpha": 0.75}'])
        assert mock_to_csv.called
