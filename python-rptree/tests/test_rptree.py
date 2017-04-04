#!/usr/bin/env python

import unittest
import numpy as np

import context
from rptree.nearestNeighbor import selectRank, makeForest, euclidean

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
