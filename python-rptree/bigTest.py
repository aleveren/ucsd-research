#!/usr/bin/python

import unittest
import numpy as np
from bigNearestNeighbor import (
    selectRank, makeForest, euclidean, Node, Rule,
    projectionsToRulesTree)

class TestNearestNeighbor(unittest.TestCase):

  def test_projectionsToRulesTree_simple(self):
    projections = np.array([
        [0, 0],
        [1, 70],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50],
        [6, 60],
        [7, 10],
    ])
    directions = [np.ones(10)]
    quantiles = [0.5]
    maxLeafSize = 1
    result = projectionsToRulesTree(
        projections, directions, quantiles, maxLeafSize, columnIndex = 1)
    expected = Node(Rule(np.ones(10), 30),
        np.array([0, 2, 3, 7]),
        np.array([1, 4, 5, 6]))

    np.testing.assert_equal(expected, result)

  def test_projectionsToRulesTree_withDepth(self):
    projections = np.array([
        [0,  0, 24,  0],
        [1, 70,  0, 31],
        [2, 20, 23,  0],
        [3, 30, 22,  0],
        [4, 40,  0, 34],
        [5, 50,  0, 32],
        [6, 60,  0, 33],
        [7, 10, 21,  0],
    ])
    directions = [np.ones(3), np.ones(3)*2, np.ones(3)*3]
    quantiles = [0.5, 0.25, 0.75]
    maxLeafSize = 1
    result = projectionsToRulesTree(
        projections, directions, quantiles, maxLeafSize, columnIndex = 1)
    expected = Node(Rule(np.ones(3), 30),
        Node(Rule(np.ones(3)*2, 21),
            np.array([7]),        # <= 50 pctile on dim1, <= 25 pctile on dim2
            np.array([0, 2, 3])), # <= 50 pctile on dim1,  > 25 pctile on dim2
        Node(Rule(np.ones(3)*3, 33),
            np.array([1, 5, 6]),  #  > 50 pctile on dim1, <= 75 pctile on dim3
            np.array([4])))       #  > 50 pctile on dim1,  > 75 pctile on dim3

    np.testing.assert_equal(expected, result)

  def test_mapPathsToLeaves(self):
    tree = Node(None, 10, 20)
    expected = {"L": 10, "R": 20}
    result = tree.mapPathsToLeaves()
    self.assertEqual(result, expected)

    tree = Node(None, Node(None, 10, 11), 20)
    expected = {"LL": 10, "LR": 11, "R": 20}
    result = tree.mapPathsToLeaves()
    self.assertEqual(result, expected)

    tree = Node(None,
        Node(None,
            10,
            Node(None, 11, 12)),
        Node(None,
            Node(None, 20, 21),
            22))
    expected = {"LL": 10, "LRL": 11, "LRR": 12,
        "RLL": 20, "RLR": 21, "RR": 22}
    result = tree.mapPathsToLeaves()
    self.assertEqual(result, expected)

  def test_replaceLeaves(self):
    tree = Node(None, 10, 20)
    expected = Node(None, 11, 21)
    result = tree.replaceLeaves(lambda p, v: v+1)
    self.assertEqual(result, expected)

    tree = Node(None, Node(None, 10, 11), 20)
    expected = Node(None, Node(None, 11, 12), 21)
    result = tree.replaceLeaves(lambda p, v: v+1)
    self.assertEqual(result, expected)

    tree = Node(None,
        Node(None,
            10,
            Node(None, 11, 12)),
        Node(None,
            Node(None, 20, 21),
            22))
    expected = Node(None,
        Node(None,
            "LL",
            Node(None, "LRL", "LRR")),
        Node(None,
            Node(None, "RLL", "RLR"),
            "RR"))
    result = tree.replaceLeaves(lambda p, v: p)
    self.assertEqual(result, expected)

  def test_replaceLeaves_expandTree(self):
    tree = Node(None, Node(None, 10, 20), 30)
    expected = Node(None,
      Node(None,
        Node(None, 11, Node(None, 12, 13)),
        Node(None, 21, Node(None, 22, 23))),
      Node(None, 31, Node(None, 32, 33)))
    result = tree.replaceLeaves(
        lambda p, v: Node(None, v+1, Node(None, v+2, v+3)))
    self.assertEqual(result, expected)

if __name__ == "__main__":
  unittest.main()
