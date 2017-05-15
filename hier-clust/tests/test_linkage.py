import unittest
import numpy as np

import context
from hier_clust import tree_util, linkage_util

Tree = tree_util.Tree


class Tests(unittest.TestCase):
    def get_tree(self):
        return Tree(data = {"orig_indices": [0, 1, 2, 3, 4]}, children = [
            Tree(data = {"orig_indices": [0, 1]}, children = [
                Tree.leaf(data = {"orig_indices": [0]}),
                Tree.leaf(data = {"orig_indices": [1]}),
            ]),
            Tree(data = {"orig_indices": [2, 3, 4]}, children = [
                Tree.leaf(data = {"orig_indices": [3]}),
                Tree(data = {"orig_indices": [2, 4]}, children = [
                    Tree.leaf(data = {"orig_indices": [2]}),
                    Tree.leaf(data = {"orig_indices": [4]}),
                ]),
            ]),
        ])

    def test_linkage(self):
        tree = self.get_tree()
        result = linkage_util.get_linkage(tree)
        assert len(result) == 4  # should match number of internal nodes
        np.testing.assert_array_equal(result[0], [0, 1, 2, 2])
        np.testing.assert_array_equal(result[1], [2, 4, 1, 2])
        np.testing.assert_array_equal(result[2], [3, 6, 2, 3])
        np.testing.assert_array_equal(result[3], [5, 7, 3, 5])

    def test_linkage_to_tree(self):
        tree = self.get_tree()
        link = linkage_util.get_linkage(tree)
        tree2 = linkage_util.linkage_to_tree(link)
        assert tree == tree2
