#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.sparse import (coo_matrix, dia_matrix, csr_matrix,
    dok_matrix, issparse)
from sklearn.neighbors import NearestNeighbors, DistanceMetric
import sklearn.metrics
from collections import defaultdict, Counter, deque
import sys
import os
import re
import json
import argparse
import logging

from tree_util import Tree, get_path_element


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


class HierClust(object):
    def __init__(self,
            n_neighbors = 20,
            mutual_neighbors = False,
            sigma_similarity = 'auto',
            alpha = 0.8,
            leaf_size = 1,
            neighbor_graph_strategy = 'balltree',
            metric = 'euclidean',
            full_eigen_threshold = 10,
            convergence_iterations = 10):
        self.n_neighbors = n_neighbors
        self.mutual_neighbors = mutual_neighbors
        self.sigma_similarity = sigma_similarity
        self.alpha = alpha
        self.leaf_size = leaf_size
        self.neighbor_graph_strategy = neighbor_graph_strategy
        self.metric = metric
        self.full_eigen_threshold = full_eigen_threshold
        self.convergence_iterations = convergence_iterations

    def fit(self, data):
        '''
        Generates a hierarchical clustering on the given data
        '''
        data = np.asarray(data)
        orig_indices = np.arange(len(data))

        tree = self._clustering(data, orig_indices, tree_path = '')
        assignments = self._get_assignments(tree)
        tree_paths = np.array([p for i, p in assignments], dtype='object')

        return tree, tree_paths

    def _clustering(self, data, orig_indices, tree_path):
        '''
        Recursive helper function for partitioning a dataset into 2 parts
        '''
        log_msg = "Partitioning {} observations (tree_path = {})" \
            .format(len(data), tree_path)
        _logger.debug(log_msg)

        distances = self._get_distances(data)
        components = self._get_connected_components(distances)
        grouped_by_component = self._group_data_by_component(
            data = data,
            distances = distances,
            components = components,
            orig_indices = orig_indices)

        return self._cluster_multiple_components(
            groups = grouped_by_component,
            tree_path = tree_path)

    def _cluster_multiple_components(self, groups, tree_path):
        if len(groups) == 1:
            only_group = groups[groups.keys()[0]]
            return self._cluster_single_component(
                data = only_group["data"],
                distances = only_group["distances"],
                orig_indices = only_group["orig_indices"],
                tree_path = tree_path)

        unique_components = []
        component_sizes = []
        all_orig_indices = []
        for comp in groups:
            unique_components.append(comp)
            component_sizes.append(groups[comp]["size"])
            all_orig_indices.extend(groups[comp]["orig_indices"])

        # Heuristic: put largest 2 components in separate branches
        top_component_indices = np.argpartition(component_sizes, -2)[-2:]
        top_components = np.array(unique_components)[top_component_indices]

        new_groups = {"left": dict(), "right": dict()}
        new_sizes = {"left": 0, "right": 0}
        for i, name in enumerate(["left", "right"]):
            comp = top_components[i]
            new_groups[name][comp] = groups[comp]
            new_sizes[name] += groups[comp]["size"]

        # Divide the remaining components naively
        for comp, current_group in groups.items():
            if comp not in top_components:
                if new_sizes["left"] < new_sizes["right"]:
                    side = "left"
                else:
                    side = "right"
                new_groups[side][comp] = current_group
                new_sizes[side] += current_group["size"]

        assert len(new_groups["left"]) > 0
        assert len(new_groups["right"]) > 0

        left_tree = self._cluster_multiple_components(
            groups = new_groups["left"],
            tree_path = tree_path + get_path_element(0))
        right_tree = self._cluster_multiple_components(
            groups = new_groups["right"],
            tree_path = tree_path + get_path_element(0))
        children = [left_tree, right_tree]

        return Tree(children = children, data = {
            "orig_indices": all_orig_indices,
            "tree_path": tree_path,
        })

    def _cluster_single_component(self, data, distances,
            orig_indices, tree_path):
        if len(data) <= 1 or len(data) <= self.leaf_size:
            return Tree.leaf(data = {
                "orig_indices": orig_indices,
                "tree_path": tree_path,
            })

        partition = self._partition_within_component(
            data = data, distances = distances)

        assert partition.shape == (len(data),)
        partition_sizes = Counter(partition)
        assert len(partition_sizes) == 2 and partition_sizes[0] > 0 and \
            partition_sizes[1] > 0,  \
            "Bad partition sizes: {}".format(partition_sizes)
        _logger.debug("Partition result: {}".format(partition_sizes))

        children = []
        for label in [0, 1]:
            partition_mask = partition == label
            data_subset = data[partition_mask]
            orig_indices_subset = orig_indices[partition_mask]
            subtree = self._clustering(
                data_subset,
                orig_indices_subset,
                tree_path + get_path_element(label))
            children.append(subtree)

        return Tree(children = children, data = {
            "orig_indices": orig_indices,
            "tree_path": tree_path,
        })

    def _partition_within_component(self, data, distances):
        '''
        Partitions a connected component into two pieces
        '''
        n_obs = data.shape[0]
        assert n_obs >= 2

        if n_obs == 2:
            return np.array([0, 1])

        similarity = self._get_similarity(distances)
        diag = similarity.sum(axis = 0)
        diag = dia_matrix((diag, [0]), (n_obs, n_obs)).tocsr()
        laplacian = diag - similarity
        fiedler_vector = self._get_fiedler_vector(laplacian)

        stats = Counter()
        for f in fiedler_vector:
            if f < 0: stats['neg'] += 1
            elif f == 0: stats['eq'] += 1
            elif f > 0: stats['pos'] += 1

        if stats['neg'] > 0 and stats['pos'] + stats['eq'] > 0:
            partition = (fiedler_vector >= 0).astype('int')
        else:  # pragma: no cover
            # TODO: fall back to some other partitioning method?
            raise Exception("Couldn't properly partition data; "
                "eigenvector components: {}".format(stats))

        assert partition.shape == (n_obs,)
        return partition

    def _get_connected_components(self, A):
        '''
        Use breadth-first search to label all connected components
        '''
        if not isinstance(A, csr_matrix):
            A = csr_matrix(A)
        n_obs = A.shape[0]
        num_components = 0
        unvisited = set(range(n_obs))
        components = -1 * np.ones((n_obs,))

        def bfs(start_index, component):
            q = deque([start_index])

            while len(q) > 0:
                i = q.popleft()
                if i in unvisited:
                    unvisited.remove(i)
                    components[i] = component

                    # Iterate over unvisited neighbors
                    for offset in range(A.indptr[i], A.indptr[i + 1]):
                        j = A.indices[offset]
                        if j in unvisited:
                            q.append(j)

        while len(unvisited) > 0:
            # Find an unvisited vertex
            for current in unvisited:
                break  # Hacky way to get an arbitrary element from a set
            bfs(current, num_components)
            num_components += 1

        return components

    def _group_data_by_component(self, data, distances, orig_indices, components):
        indices_by_component = defaultdict(list)
        for i, c in enumerate(components):
            indices_by_component[c].append(i)
        grouped_by_component = dict()
        for c, indices in indices_by_component.items():
            grouped_by_component[c] = {
                "size": len(indices),
                "data": data[indices],
                "orig_indices": orig_indices[indices],
                "distances": self._distance_subset(distances, indices),
            }
        return grouped_by_component

    def _distance_subset(self, distances, indices):
        '''
        Extract the submatrix of a sparse distance matrix corresponding
        to the given indices, slicing along both rows and columns.
        '''
        assert isinstance(distances, csr_matrix)
        indices = np.asarray(indices)
        # First, slice CSR by rows
        result = distances[indices, :]
        # Then, slice CSC by columns and convert back to CSR
        result = result.tocsc()[:, indices].tocsr()
        return result

    def _get_fiedler_vector(self, A):
        '''
        Compute the eigenvector corresponding to the 2nd smallest eigenvalue
        '''
        if A.shape[0] < self.full_eigen_threshold:
            if issparse(A):  # pragma: no cover
                A = np.asarray(A.todense())
            ws, vs = np.linalg.eigh(A)
            return np.asarray(vs[:, 1]).flatten()

        if not isinstance(A, csr_matrix):
            A = csr_matrix(A)

        n_obs = A.shape[0]
        const_vector = np.ones((n_obs,)) / np.sqrt(n_obs)
        x = np.random.uniform(-1, 1, const_vector.shape)
        x -= np.dot(const_vector, x) * const_vector
        x /= np.linalg.norm(x)

        for i in range(self.convergence_iterations):
            x = self._solve_conjugate_gradient(A, x)
            x -= np.dot(const_vector, x) * const_vector
            x /= np.linalg.norm(x)
            if -np.min(x) > np.max(x):
                x *= -1

        return x

    def _solve_conjugate_gradient(self, A, b):
        '''
        Solve Ax = b using conjugate gradient.  For best performance,
        A should be a sparse matrix in the CSR format (Compressed Sparse Row).
        '''
        input_shape = b.shape
        b = b.reshape((-1, 1))
        x = np.zeros_like(b)
        r = b - A.dot(x)
        p = r
        for k in range(self.convergence_iterations):
            prev_r = r
            alpha = np.asscalar(r.T.dot(r) / float(p.T.dot(A.dot(p))))
            x = x + alpha * p
            r = r - alpha * A.dot(p)
            beta = np.asscalar(r.T.dot(r) / prev_r.T.dot(prev_r))
            p = r + beta * p
        return x.reshape(input_shape)

    def _get_assignments(self, tree):
        '''
        Turns a tree into a column of cluster id's ("tree paths")
        '''
        if len(tree.children) == 0:
            indices = tree.data["orig_indices"]
            path = tree.data["tree_path"]
            indices = sorted(indices)
            return [(i, path) for i in indices]

        # Get assignments recursively on children
        children_results = [self._get_assignments(c) for c in tree.children]

        # Merge the results
        merge_status = [0 for c in tree.children]
        result = []
        while True:
            source = None
            lowest_index = None
            lowest_index_result = None

            for k in range(len(tree.children)):
                if merge_status[k] < len(children_results[k]):
                    candidate = children_results[k][merge_status[k]]
                    candidate_index, candidate_path = candidate
                    if source is None or candidate_index < lowest_index:
                        source = k
                        lowest_index = candidate_index
                        lowest_index_result = candidate

            if source is None:
                break

            merge_status[source] += 1
            result.append(lowest_index_result)

        return result

    def _get_median(self, elements, k = None):
        '''
        Computes the median of an array using quickselect.
        '''
        if len(elements) == 0:
            return None
        if k is None:
            k = int(np.ceil(len(elements) / 2.)) - 1
        elements = np.asarray(elements)
        pivot_index = np.random.randint(len(elements))
        pivot_value = elements[pivot_index]
        below = elements[elements < pivot_value]
        equal = elements[elements == pivot_value]
        above = elements[elements > pivot_value]
        if k >= len(below) and k < len(below) + len(equal):
            return pivot_value
        elif k < len(below):
            return self._get_median(below, k)
        else:
            return self._get_median(above, k - len(below) - len(equal))

    def _get_similarity(self, dist):
        '''
        Generate a similarity matrix for the given distance matrix
        '''
        assert isinstance(dist, csr_matrix)

        if self.sigma_similarity == 'auto':
            # Choose the value of sigma that maps the
            # median distance to self.alpha
            flat_dist = np.asarray(dist.data)
            nontrivial_dist = flat_dist[(flat_dist != 0) & np.isfinite(flat_dist)]
            _logger.debug("Computing median distance")
            med_dist = self._get_median(nontrivial_dist)
            _logger.debug("Median distance = {}".format(med_dist))
            if med_dist is not None:
                sigma = med_dist * np.sqrt(1./(2. * np.log(1. / self.alpha)))
            else:  # pragma: no cover
                _logger.warning("Median calculation failed")
                sigma = 1.0
        else:
            sigma = self.sigma_similarity
        scaling_factor = 2. * sigma ** 2

        sim_data = np.ones_like(dist.data)
        mask = np.isfinite(dist.data)
        sim_data[mask] = np.exp(-dist.data[mask] ** 2 / scaling_factor)

        similarity = dist.copy()
        similarity.data = sim_data

        # Enforce symmetry
        # TODO: make sure that this operation is still efficient when sparse
        similarity = 0.5 * similarity + 0.5 * similarity.T

        flat_sim = np.asarray(similarity.data)
        nontrivial_sim = flat_sim[(flat_sim != 0) & (flat_sim != 1.0) & np.isfinite(flat_sim)]
        _logger.debug("Computing median similarity")
        med_sim = self._get_median(nontrivial_sim)
        _logger.debug("Median similarity = {}".format(med_sim))

        assert isinstance(similarity, csr_matrix)

        return similarity

    def _get_distances(self, data):
        '''
        Gets a distance matrix based on K-nearest neighbors.
        '''
        n_obs = data.shape[0]
        n_neighbors = min(self.n_neighbors, n_obs)

        if self.neighbor_graph_strategy == 'balltree':
            nn_obj = NearestNeighbors(
                n_neighbors = n_neighbors,
                algorithm = 'ball_tree',
                metric = self.metric,
            ).fit(data)
            distances, indices = nn_obj.kneighbors(data)

        else:  # pragma: no cover
            raise Exception("Unrecognized neighbor_graph_strategy: {}".format(self.neighbor_graph_strategy))

        entries = defaultdict(list)
        for row_index in xrange(len(data)):
            for nbr_index in xrange(n_neighbors):
               col_index = indices[row_index, nbr_index]

               dist = distances[row_index, nbr_index]
               if dist == 0.0:
                   # Workaround: prevents sparse format from ignoring zeros
                   dist = np.nan

               a, b = min(row_index, col_index), max(row_index, col_index)
               entries[a, b].append(dist)

        row_new = []
        col_new = []
        val_new = []
        for (a, b), dists in entries.items():
            count = len(dists)
            assert count == 1 or count == 2

            if a == b:
                assert count == 1
                row_new.append(a)
                col_new.append(b)
                val_new.append(dists[0])

            if a != b and (not self.mutual_neighbors or count == 2):
                row_new.append(a)
                col_new.append(b)
                val_new.append(dists[0])
                row_new.append(b)
                col_new.append(a)
                val_new.append(dists[0])

        result = coo_matrix((val_new, (row_new, col_new)), shape=(len(data), len(data)))
        result = result.tocsr()
        return result


def numeric_logging_level(level_string):
    num_level = getattr(logging, level_string.upper(), None)
    if not isinstance(num_level, int):  # pragma: no cover
        raise ValueError('Invalid log level: {}'.format(level_string))
    return num_level


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required = True)
    parser.add_argument('--feature_columns', default = None)
    parser.add_argument('--log', default = 'WARNING')
    parser.add_argument('--random_seed', type = long, default = None)
    parser.add_argument('--output_column', default = "cluster_id")
    parser.add_argument('--output', default = None)
    parser.add_argument('--constructor_json', default = None)
    args = parser.parse_args(argv)

    data = pd.read_csv(args.input)
    if args.feature_columns is not None:
        col_indices = [i for i in range(len(data.columns))
            if re.match(args.feature_columns, data.columns[i])]
    else:
        col_indices = slice(None, None, None)

    logging.basicConfig(
        level = numeric_logging_level(args.log),
        format = '%(asctime)s %(message)s')

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    if args.constructor_json is not None:
        constructor_args = json.loads(args.constructor_json)
    else:
        constructor_args = dict()

    hc = HierClust(**constructor_args)
    tree, assignments = hc.fit(data = data.iloc[:, col_indices])

    if args.output is None:
        for i, path in enumerate(assignments):
            print("{}: {}".format(i, path))
    else:
        _logger.info("Outputting to '%s' (column = '%s')",
            args.output, args.output_column)
        data[args.output_column] = assignments
        data.to_csv(args.output, index = False)


if __name__ == '__main__':  # pragma: no cover
    main(sys.argv[1:])
