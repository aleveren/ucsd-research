#!/usr/bin/python

import unittest
import numpy as np
from nearestNeighbor import selectRank, makeForest, euclidean

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
