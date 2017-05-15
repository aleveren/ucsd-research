import unittest
from mock import patch
import numpy as np
import pandas as pd
from scipy.sparse import issparse

import context
import hier_clust


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
        d = np.array([[0,0], [1,0], [0,2], [1,2], [5,5], [5,5]])

        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0,
            sparse_similarity = 'auto')
        dist1 = hc._get_distances(d)
        assert issparse(dist1)

        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0,
            sparse_similarity = 'never')
        dist2 = hc._get_distances(d)
        assert not issparse(dist2)

        # Check distance calculations
        z = np.nan
        expected = np.array([
            [ z,  1,  0,  0,  0,  0],
            [ 1,  z,  0,  0,  0,  0],
            [ 0,  0,  z,  1,  0,  0],
            [ 0,  0,  1,  z,  0,  0],
            [ 0,  0,  0,  0,  z,  z],
            [ 0,  0,  0,  0,  z,  z],
        ], dtype='float')
        dist1_squared = np.asarray(np.square(dist1.todense()))
        np.testing.assert_allclose(dist1_squared, expected, equal_nan=True)

        expected = np.array([
            [ 0,  1,  4,  5, 50, 50],
            [ 1,  0,  5,  4, 41, 41],
            [ 4,  5,  0,  1, 34, 34],
            [ 5,  4,  1,  0, 25, 25],
            [50, 41, 34, 25,  0,  0],
            [50, 41, 34, 25,  0,  0],
        ], dtype='float')
        np.testing.assert_allclose(np.square(dist2), expected)

        # Check similarity calculations
        sim1 = hc._get_similarity(dist1)
        assert issparse(sim1)
        sim2 = hc._get_similarity(dist2)
        assert not issparse(sim2)

        x = np.exp(-0.5)
        expected = np.array([
            [1, x, 0, 0, 0, 0],
            [x, 1, 0, 0, 0, 0],
            [0, 0, 1, x, 0, 0],
            [0, 0, x, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1]])
        np.testing.assert_allclose(sim1.todense(), expected)
        assert sim1.nnz == 12
        assert not np.allclose(sim1.todense(), sim2)
        assert not np.allclose(sim2, expected)

        expected = np.exp(-dist2 ** 2 / 2.0)
        np.testing.assert_allclose(sim2, expected)

    def test_similarity_sparse_mutual(self):
        # Test case where mutual KNN makes a difference
        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0,
            mutual_neighbors = True)
        d = np.array([[0,0], [1,0], [0,2]])

        dist1 = hc._get_distances(d)

        z = np.nan
        expected = np.array([
            [z, 1, 0],
            [1, z, 0],
            [0, 0, z]])
        dist1_squared = np.asarray(np.square(dist1.todense()))
        np.testing.assert_allclose(dist1_squared, expected, equal_nan=True)
        assert dist1.nnz == 5

        sim1 = hc._get_similarity(dist1)

        x = np.exp(-0.5)
        expected = np.array([
            [1, x, 0],
            [x, 1, 0],
            [0, 0, 1]])
        np.testing.assert_allclose(sim1.todense(), expected)
        assert sim1.nnz == 5

        # Recompute similarities, allowing non-mutual neighbors
        hc = hier_clust.HierClust(n_neighbors = 2, sigma_similarity = 1.0,
            mutual_neighbors = False)

        dist2 = hc._get_distances(d)

        z = np.nan
        expected = np.array([
            [z, 1, 4],
            [1, z, 0],
            [4, 0, z]])
        dist2_squared = np.asarray(np.square(dist2.todense()))
        np.testing.assert_allclose(dist2_squared, expected, equal_nan=True)
        assert dist2.nnz == 7

        sim2 = hc._get_similarity(dist2)

        x = np.exp(-0.5)
        y = np.exp(-2.0)
        expected = np.array([
            [1, x, y],
            [x, 1, 0],
            [y, 0, 1]])
        np.testing.assert_allclose(sim2.todense(), expected)
        assert sim2.nnz == 7

    def test_cluster_balltree(self):
        data = self.sim_data()
        hc = hier_clust.HierClust(
            n_neighbors = 10,
            sigma_similarity = 'auto',
            neighbor_graph_strategy = 'balltree')
        tree, assignments = hc.fit(data)
        assert len(tree.data["orig_indices"]) == 100
        assert len(assignments) == 100

    def test_cluster_rptree(self):
        data = self.sim_data()
        hc = hier_clust.HierClust(
            n_neighbors = 10,
            sigma_similarity = 'auto',
            neighbor_graph_strategy = 'rptree')
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

    def test_conjugate_gradient(self):
        hc = hier_clust.HierClust()
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([34, 79, 124])
        expected = np.array([3, 5, 7])
        result = hc._solve_conjugate_gradient(A, b)
        np.testing.assert_allclose(result, expected, atol=1e-2)

    def test_fiedler_vector(self):
        # Fiedler computation via full eigendecomposition
        hc = hier_clust.HierClust(full_eigen_threshold = 10)
        W = np.array([
            [1.0, 0.5, 0.1, 0.1],
            [0.5, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.5],
            [0.1, 0.1, 0.5, 1.0],
        ])
        diag = np.diag(W.sum(axis = 0))
        L = diag - W
        expected = np.array([0.5, 0.5, -0.5, -0.5])
        result = hc._get_fiedler_vector(L)
        np.testing.assert_allclose(result, expected, atol=1e-6)

        # Fiedler computation via power iteration
        hc = hier_clust.HierClust(full_eigen_threshold = 2)
        result = hc._get_fiedler_vector(L)
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_connected_components(self):
        hc = hier_clust.HierClust()

        z = lambda n, m: np.zeros((n, m))
        a = lambda n: np.ones((n, n))
        dist = np.bmat([
            [3*a(3), z(3,2), z(3,2)],
            [z(2,3), 5*a(2), z(2,2)],
            [z(2,3), z(2,2), 7*a(2)],
        ])
        components, num_components = hc._get_connected_components(dist)

        assert np.array_equal(np.sort(np.unique(components)), [0, 1, 2])
        assert len(np.unique(components[:3])) == 1
        assert len(np.unique(components[3:5])) == 1
        assert len(np.unique(components[5:])) == 1
        assert num_components == 3

    def test_custom_metric(self):
        def metric(x, y):
            x, y = min(x, y), max(x, y)
            if x == y: return 0.0
            elif x == 1 and y == 2: return 3.0
            elif x == 1 and y == 3: return 5.0
            elif x == 2 and y == 3: return 7.0
            else: return 10.0

        hc = hier_clust.HierClust(metric = metric)

        data = np.array([1, 2, 3]).reshape((-1, 1))
        dist = hc._get_distances(data)

        expected_dist = np.array([
            [0, 3, 5],
            [3, 0, 7],
            [5, 7, 0],
        ])
        assert np.array_equal(dist, expected_dist)

        # Test again with rptree
        hc = hier_clust.HierClust(metric = metric, n_neighbors = 2,
            neighbor_graph_strategy = 'rptree')

        data = np.array([1, 2, 3]).reshape((-1, 1))
        dist = hc._get_distances(data)
        result = hc._partition(data)

        z = np.nan
        expected_dist = np.array([
            [z, 3, 5],
            [3, z, 0],
            [5, 0, z],
        ])
        np.testing.assert_allclose(dist.todense(), expected_dist, equal_nan = True)

    def test_partition_end_to_end(self):
        hc = hier_clust.HierClust(sigma_similarity = 1.0)

        a = np.sqrt(-2*np.log(0.5))
        b = np.sqrt(-2*np.log(0.1))

        q = a / 2.
        r = np.sqrt(b**2 - q**2)
        y = (-b**2 + 5*a**2/4. + r**2) / (2.*r)
        z = np.sqrt(a**2 - y**2)

        # Construct a dataset that leads to a simple similarity matrix
        data = np.array([
            [-q, 0, 0],
            [q, 0, 0],
            [0, r, 0],
            [0, r-y, z],
        ])

        expected_dist = np.array([
            [0, a, b, b],
            [a, 0, b, b],
            [b, b, 0, a],
            [b, b, a, 0],
        ])

        dist = hc._get_distances(data)
        np.testing.assert_allclose(dist, expected_dist)

        expected_sim = np.array([
            [1.0, 0.5, 0.1, 0.1],
            [0.5, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.5],
            [0.1, 0.1, 0.5, 1.0],
        ])

        sim = hc._get_similarity(dist)
        np.testing.assert_allclose(sim, expected_sim)

        expected_fiedler = np.array([0.5, 0.5, -0.5, -0.5])

        diag = np.diag(sim.sum(axis=0))
        L = diag - sim
        fiedler = hc._get_fiedler_vector(L)
        np.testing.assert_allclose(fiedler, expected_fiedler)

        expected_partition = np.array([1, 1, 0, 0])
        partition = hc._partition(data)
        assert np.array_equal(partition, expected_partition)

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
