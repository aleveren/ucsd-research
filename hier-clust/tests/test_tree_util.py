import unittest
import numpy as np
import textwrap

from context import tree_util

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
        assert self.large_tree().subtree('11') == expected

        assert self.large_tree().subtree('10') == Tree.leaf(6)

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
        result = tree_util.reconstruct_tree(['000', '001', '01', '10', '11']) \
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
