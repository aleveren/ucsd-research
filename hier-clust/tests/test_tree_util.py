import unittest
import numpy as np
import textwrap

import context
from hier_clust import tree_util

Tree = tree_util.Tree


class Tests(unittest.TestCase):
    def large_tree(self):
        return Tree(data = 1, children = [
            Tree(data = 2, children = [
                Tree.leaf(4),
                Tree.leaf(5),
            ]),
            Tree(data = 3, children = [
                Tree.leaf(6),
                Tree(data = 7, children = [
                    Tree.leaf(8),
                    Tree.leaf(9),
                ]),
            ]),
        ])

    def test_depth(self):
        leaf1 = Tree.leaf(data = 42)
        assert leaf1.depth() == 0

        leaf2 = Tree(data = 7, children = [])
        assert leaf2.depth() == 0

        tree1 = Tree(data = 10, children = [leaf1, leaf2])
        assert tree1.depth() == 1

        tree2 = Tree(data = 3, children = [
            Tree.leaf(data = 6),
            tree1,
        ])
        assert tree2.depth() == 2

        assert self.large_tree().depth() == 3

    def test_equal(self):
        assert Tree.leaf(42) == Tree.leaf(42)
        assert Tree.leaf(42) != Tree.leaf(7)

    def test_prune(self):
        expected = Tree(data = 1, children = [
            Tree.leaf(data = 2),
            Tree.leaf(data = 3),
        ])
        assert self.large_tree().prune(1) == expected

    def test_subtree(self):
        expected = Tree(data = 7, children = [
            Tree.leaf(data = 8),
            Tree.leaf(data = 9),
        ])
        assert self.large_tree().subtree('RR') == expected

        assert self.large_tree().subtree('RL') == Tree.leaf(6)

    def test_map_data(self):
        expected = Tree(data = 10, children = [
            Tree(data = 20, children = [
                Tree.leaf(40),
                Tree.leaf(50),
            ]),
            Tree(data = 30, children = [
                Tree.leaf(60),
                Tree(data = 70, children = [
                    Tree.leaf(80),
                    Tree.leaf(90),
                ]),
            ]),
        ])
        result = self.large_tree().map_data(lambda x: x * 10)
        assert result == expected

    def test_reduce_leaf_data(self):
        result = self.large_tree().reduce_leaf_data(
            combine = lambda x, y: x + y,
            leaf_func = lambda x: 1)
        assert result == 5

        result = self.large_tree().reduce_leaf_data(
            combine = sum,
            leaf_func = lambda x: 1,
            list_arg = True)
        assert result == 5

        result = self.large_tree().reduce_leaf_data(
            combine = lambda x, y: x + y,
            leaf_func = lambda x: x)
        expected = 32  # (4 + 5) + (6 + (8 + 9))
        assert result == expected

        # Try omitting leaf_func (equiv. to identity function)
        result = self.large_tree().reduce_leaf_data(
            combine = lambda x, y: x + y)
        expected = 32  # (4 + 5) + (6 + (8 + 9))
        assert result == expected

    def test_num_leaves(self):
        result = self.large_tree().num_leaves()
        assert result == 5

    def test_str_display(self):
        result = self.large_tree().str_display()
        expected = textwrap.dedent("""\
            Tree(data = 1, children = [
              Tree(data = 2, children = [
                Tree(data = 4, children = [])
                Tree(data = 5, children = [])
              ])
              Tree(data = 3, children = [
                Tree(data = 6, children = [])
                Tree(data = 7, children = [
                  Tree(data = 8, children = [])
                  Tree(data = 9, children = [])
                ])
              ])
            ])""")
        assert result == expected

    def test_reconstruct_tree(self):
        result = tree_util.reconstruct_tree(['LLL', 'LLR', 'LR', 'RL', 'RR']) \
            .map_data(list)
        expected = Tree(data = [0, 1, 2, 3, 4], children = [
            Tree(data = [0, 1, 2], children = [
                Tree(data = [0, 1], children = [
                    Tree(data = [0], children = []),
                    Tree(data = [1], children = []),
                ]),
                Tree(data = [2], children = []),
            ]),
            Tree(data = [3, 4], children = [
                Tree(data = [3], children = []),
                Tree(data = [4], children = []),
            ]),
        ])
        assert result == expected
