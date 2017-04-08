#!/usr/bin/env python

import unittest
import numpy as np

import context
from rptree.nearestNeighbor import (selectRank, makeForest, euclidean,
    NearestNeighborForest, Node, Leaf, Rule, repopulateTree)

class TestNearestNeighbor(unittest.TestCase):

  def test_nearest(self):
    np.random.seed(1)
    dims = 5
    M = np.sqrt(dims) + 1
    nrows = dims * 100
    data = np.ones((1, dims))
    for i in range(nrows):
      row = np.random.uniform(0, 1, [1, dims])
      row[0, i % dims] = M
      data = np.vstack((data, row))

    forest = makeForest(data, maxLeafSize = 100, numTrees = 20,
        distanceFunction = euclidean)
    query = np.zeros(dims)
    result = forest.nearestNeighbor(query)
    expected = np.ones(dims)

    self.assertTrue(np.array_equal(expected, result))

  def test_kneighbors(self):
    np.random.seed(1)

    isolated_points = np.array([
        [110, 110],
        [113, 114],
        [119, 122],
    ], dtype='float')

    n_obs = 100
    n_col = isolated_points.shape[1]
    data = np.vstack([
        isolated_points,
        np.random.normal(0, 1, (n_obs - len(isolated_points), n_col)),
    ])
    perm = np.random.permutation(n_obs)
    data = data[perm, :]  # shuffle rows

    forest = makeForest(data, maxLeafSize = 10, numTrees = 20,
        distanceFunction = euclidean)

    distances, indices = forest.kneighbors(isolated_points, k = 3)

    expected = np.array([
        [0, 5, 15],
        [0, 5, 10],
        [0, 10, 15],
    ], dtype='float')
    np.testing.assert_allclose(distances, expected)

    # Extract indices corresponding to isolated points
    i1, i2, i3 = [list(perm).index(i) for i in range(3)]

    expected = np.array([
        [i1, i2, i3],
        [i2, i1, i3],
        [i3, i2, i1],
    ], dtype='int')
    np.testing.assert_array_equal(indices, expected)

  def test_kneighbors_trees_disagree(self):
    # Test case where different trees return different sets of nearest neighbors
    data = np.array([[-1, -2], [-1, 2], [1, -2], [1, 2]], dtype='float')

    empty_leaf = lambda: Leaf([], [], euclidean)
    t1 = Node(Rule(np.array([1., 0.]), 0.), empty_leaf(), empty_leaf())
    t2 = Node(Rule(np.array([0., 1.]), 0.), empty_leaf(), empty_leaf())

    repopulateTree(tree = t1, data = data)
    repopulateTree(tree = t2, data = data)

    np.testing.assert_array_equal(t1.getLeaf(data[0]).data, data[[0, 1], :])
    np.testing.assert_array_equal(t1.getLeaf(data[1]).data, data[[0, 1], :])
    np.testing.assert_array_equal(t1.getLeaf(data[2]).data, data[[2, 3], :])
    np.testing.assert_array_equal(t1.getLeaf(data[3]).data, data[[2, 3], :])

    np.testing.assert_array_equal(t2.getLeaf(data[0]).data, data[[0, 2], :])
    np.testing.assert_array_equal(t2.getLeaf(data[1]).data, data[[1, 3], :])
    np.testing.assert_array_equal(t2.getLeaf(data[2]).data, data[[0, 2], :])
    np.testing.assert_array_equal(t2.getLeaf(data[3]).data, data[[1, 3], :])

    # Make sure kneighbors is working as expected for each individual tree
    d, i = t1.kneighbors(data[0], k = 2)
    np.testing.assert_array_equal(i, [[0, 1]])
    d, i = t2.kneighbors(data[0], k = 2)
    np.testing.assert_array_equal(i, [[0, 2]])

    # Try getting nearest neighbors from a combination of trees
    forest = NearestNeighborForest([t1, t2], euclidean)
    d, i = forest.kneighbors(data[0], k = 2)
    np.testing.assert_array_equal(i, [[0, 2]])

  def test_kneighbors_extra_neighbors(self):
    np.random.seed(1)

    data = np.array([
        [110, 110],
        [113, 114],
        [119, 122],
    ], dtype='float')

    forest = makeForest(data, maxLeafSize = 2, numTrees = 100,
        distanceFunction = euclidean)

    distances, indices = forest.kneighbors(data, k = 4)

    expected = np.ma.masked_invalid([
        [0, 5, 15, np.nan],
        [0, 5, 10, np.nan],
        [0, 10, 15, np.nan],
    ])
    np.testing.assert_allclose(distances, expected)

    expected = np.ma.masked_equal([
        [0, 1, 2, -999],
        [1, 0, 2, -999],
        [2, 1, 0, -999],
    ], value = -999)
    np.testing.assert_array_equal(indices, expected)

  def test_selection(self):
    np.random.seed(1)
    a = np.arange(10, 110, 10)
    original = np.copy(a)
    np.random.shuffle(a)
    self.assertFalse(np.array_equal(a, original))
    for i in range(-1, len(a) + 1):
      result = selectRank(a, i)
      if i < 0:
        self.assertEqual(10, result)
      elif i >= 9:
        self.assertEqual(100, result)
      else:
        self.assertEqual(10 * (i+1), result)

if __name__ == "__main__":
  unittest.main()
