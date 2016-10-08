import numpy as np
from collections import namedtuple, defaultdict


class Tree(namedtuple("Tree", ["data", "children"])):
    @classmethod
    def leaf(cls, data = None):
        return cls(data = data, children = [])

    def depth(self):
        if len(self.children) == 0:
            return 0
        return 1 + max([c.depth() for c in self.children])

    def prune(self, depth):
        if depth <= 0:
            return Tree.leaf(data = self.data)
        pruned_children = []
        for c in self.children:
            pruned_children.append(c.prune(depth - 1))
        return Tree(data = self.data, children = pruned_children)

    def subtree(self, path):
        if len(path) == 0:
            return self
        child_index = int(path[0])
        assert child_index >= 0 and child_index < len(self.children), \
            "Cannot find requested subtree"
        return self.children[child_index].subtree(path[1:])

    def map_data(self, f):
        mapped_children = [c.map_data(f) for c in self.children]
        mapped_data = f(self.data)
        return Tree(data = mapped_data, children = mapped_children)

    def reduce_leaves(self, f_leaf, f_combine):
        if len(self.children) == 0:
            return f_leaf(self.data)
        children_results = [c.reduce_leaves(f_leaf, f_combine)
            for c in self.children]
        result = f_combine(*children_results)
        return result

    def reduce_all(self, f_node, f_combine):
        if len(self.children) == 0:
            return f_node(self.data)
        children_results = [c.reduce_all(f_node, f_combine)
            for c in self.children]
        result = f_combine(f_node(self.data), *children_results)
        return result

    def str_display(self, indent = 0):
        result = ' ' * indent + 'Tree(data = {}, children = ['.format(self.data)
        if len(self.children) > 0:
            result += '\n'
            for c in self.children:
                result += c.str_display(indent + 2)
                result += '\n'
            result += ' ' * indent
        result += '])'
        return result


def reconstruct_tree(leaf_ids, orig_indices = None, tree_path = ''):
    '''
    Given a sequence of leaf ids, reconstruct the corresponding tree
    '''
    if orig_indices is None:
        orig_indices = np.arange(len(leaf_ids))
    leaf_ids = np.asarray(leaf_ids)

    depth = len(tree_path)
    indices = defaultdict(list)

    for row_index, c in enumerate(leaf_ids):
        if not c.startswith(tree_path):
            raise Exception("Found misplaced data")
        elif depth >= len(c):
            indices["leaf"].append(row_index)
        elif c[depth] == '0':
            indices["left"].append(row_index)
        elif c[depth] == '1':
            indices["right"].append(row_index)
        else:
            raise Exception(
                "Found unexpected cluster id: {}".format(c))

    if len(leaf_ids) == 0 or len(indices["leaf"]) > 0:
        assert len(indices["left"]) == 0 and len(indices["right"]) == 0, \
            "Invalid tree-path structure (found leaf and non-leaf data)"
        return Tree.leaf(data = orig_indices[indices["leaf"]])

    c_left = leaf_ids[indices["left"]]
    c_right = leaf_ids[indices["right"]]

    i_left = orig_indices[indices["left"]]
    i_right = orig_indices[indices["right"]]

    if len(leaf_ids) > 0:
        assert len(c_left) > 0 and len(c_right) > 0, \
            "Invalid tree-path structure (left or right is missing)"

    tree_left = reconstruct_tree(c_left, i_left, tree_path + '0')
    tree_right = reconstruct_tree(c_right, i_right, tree_path + '1')

    return Tree(children = [tree_left, tree_right], data = orig_indices)
