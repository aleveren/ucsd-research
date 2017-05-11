import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc


def gen_data(depth, n_dim, depth_labels = None):
    def helper(depth, sigma, offset, path):
        if depth == 0:
            x = np.random.normal(offset, sigma, offset.shape)
            y = 0
            if depth_labels is not None:
                path = path[:depth_labels]
            for digit in path:
                y *= 2
                y += digit
            return x, np.array([y])

        direction = np.random.normal(0, 1, offset.shape)
        direction /= np.linalg.norm(direction)

        x1, y1 = helper(depth - 1, sigma = sigma / 2.0, offset = offset + direction * sigma, path = path + [0])
        x2, y2 = helper(depth - 1, sigma = sigma / 2.0, offset = offset - direction * sigma, path = path + [1])
        
        return np.vstack([x1, x2]), np.concatenate([y1, y2])

    origin = np.zeros((n_dim,))
    return helper(depth, sigma = 1.0, offset = origin, path = [])


def plot_tree_overlay(data, tree, max_depth, dims = None, ax = None):
    if dims is None:
        dims = [0, 1]
    if ax is None:
        fig, ax = plt.subplots()

    def annotate(tree):
        indices = tree.data["orig_indices"]
        tree.data["mean"] = np.mean(data[indices], axis=0)
        for c in tree.children:
            annotate(c)

    def get_lines(tree):
        start = tree.data["mean"][dims]
        lines = []
        for c in tree.children:
            end = c.data["mean"][dims]
            lines.append([start, end])
            lines.extend(get_lines(c))
        return lines

    tree = tree.prune(max_depth)
    annotate(tree)
    lines = mc.LineCollection(get_lines(tree))
    ax.add_collection(lines)