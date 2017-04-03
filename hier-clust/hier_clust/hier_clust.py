#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, dia_matrix, csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors, DistanceMetric
import sklearn.metrics
from collections import Counter, deque
import sys
import os
import re
import json
import argparse
import logging

from tree_util import Tree, get_path_element

_dirname = os.path.dirname(__file__)
_newpath = os.path.abspath(os.path.join(_dirname, '..', '..', 'python-rptree'))
print("Adding path {}".format(_newpath))
sys.path.append(_newpath)
from nearestNeighbor import makeForest


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


class HierClust(object):
    def __init__(self,
            n_neighbors = 20,
            mutual_neighbors = False,
            sigma_similarity = 'auto',
            sparse_similarity = 'auto',
            alpha = 0.8,
            leaf_size = 1,
            neighbor_graph_strategy = 'rptree',
            metric = 'euclidean',
            full_eigen_threshold = 10,
            convergence_iterations = 10):
        self.n_neighbors = n_neighbors
        self.mutual_neighbors = mutual_neighbors
        self.sigma_similarity = sigma_similarity
        self.sparse_similarity = sparse_similarity
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

        tree = self._fit_helper(data, orig_indices, tree_path = '',
            num_leaves_done = 0)
        assignments = self._get_assignments(tree)
        tree_paths = np.array([p for i, p in assignments], dtype='object')

        return tree, tree_paths

    def _fit_helper(self, data, orig_indices, tree_path, num_leaves_done):
        '''
        Recursive helper function for partitioning a dataset into 2 parts
        '''
        log_msg = "Partitioning {} observations " \
            "(tree_path = {}, num_leaves_done = {})" \
            .format(len(data), tree_path, num_leaves_done)
        _logger.debug(log_msg)

        if len(data) <= 1 or len(data) <= self.leaf_size:
            return Tree.leaf(data = {
                "orig_indices": orig_indices,
                "tree_path": tree_path,
            })

        partition = self._partition(data)
        assert partition.shape == (len(data),)

        data_subsets = []
        for label in [0, 1]:
            data_subset = data[partition == label]
            data_subsets.append(data_subset)
            size = len(data_subset)
            if size == 0 or size == len(data): # pragma: no cover
                raise Exception("Bad partition: ({} of {})" \
                    .format(size, len(data)))

        _logger.debug("Partition result: #0: {}, #1: {}" \
            .format(len(data_subsets[0]), len(data_subsets[1])))

        children = []
        for label in [0, 1]:
            data_subset = data_subsets[label]
            orig_indices_subset = orig_indices[partition == label]
            subtree = self._fit_helper(
                data_subset,
                orig_indices_subset,
                tree_path + get_path_element(label),
                num_leaves_done = num_leaves_done)
            children.append(subtree)
            num_leaves_done += len(data_subset)

        return Tree(children = children, data = {
            "orig_indices": orig_indices,
            "tree_path": tree_path,
        })

    def _partition(self, data):
        '''
        Partitions a dataset into two pieces
        '''
        n_obs = data.shape[0]
        assert n_obs >= 2

        if n_obs == 2:
            return np.array([0, 1])

        distances = self._get_distances(data)
        components, num_components = self._get_connected_components(distances)
        if num_components == 1:
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
                raise Exception("Couldn't properly partition data; "
                    "eigenvector components: {}".format(stats))

            assert partition.shape == (n_obs,)
            return partition

        else:
            _logger.debug("Found {} components".format(num_components))
            partition = (components > 0).astype('int')
            assert partition.shape == (n_obs,)
            return partition

    def _get_connected_components(self, A):
        '''
        Use depth-first search to label all connected components
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

        return components, num_components

    def _get_fiedler_vector(self, A):
        '''
        Compute the eigenvector corresponding to the 2nd smallest eigenvalue
        '''
        if A.shape[0] < self.full_eigen_threshold:
            if issparse(A):  # pragma: no cover
                A = A.todense()
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
        if self.sigma_similarity == 'auto':
            # Choose the value of sigma that maps the
            # median distance to self.alpha
            if issparse(dist):
                flat_dist = np.asarray(dist.data)
            else:
                flat_dist = dist.flatten()
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

        if issparse(dist):
            row_new = []
            col_new = []
            val_new = []
            for i in xrange(len(dist.data)):
                row = dist.row[i]
                col = dist.col[i]
                val = dist.data[i]
                row_new.append(row)
                col_new.append(col)
                if row == col or np.isnan(val):
                    # Workaround: NaN prevents sparse format from ignoring zeros
                    val_new.append(1.0)
                else:
                    sim = np.exp(-val ** 2 / scaling_factor)
                    val_new.append(sim)
            similarity = coo_matrix((val_new, (row_new, col_new)), shape=dist.shape)
        else:
            similarity = np.exp(-dist ** 2 / scaling_factor)

        # Enforce symmetry
        similarity = 0.5 * similarity + 0.5 * similarity.T

        if issparse(dist):
            flat_sim = np.asarray(similarity.data)
        else:
            flat_sim = similarity.flatten()
        nontrivial_sim = flat_sim[(flat_sim != 0) & (flat_sim != 1.0) & np.isfinite(flat_sim)]
        _logger.debug("Computing median similarity")
        med_sim = self._get_median(nontrivial_sim)
        _logger.debug("Median similarity = {}".format(med_sim))

        if issparse(similarity):
            similarity = csr_matrix(similarity)

        return similarity

    def _get_distances(self, data):
        '''
        Gets a distance matrix.
        For small datasets, return a dense matrix, otherwise return
        a sparse matrix based on K-nearest-neighbors.
        '''
        sparse = self.sparse_similarity
        if sparse == 'never':
            sparse = False
        elif sparse == 'auto' or sparse is None:
            sparse = (len(data) > self.n_neighbors)

        if not sparse:
            return sklearn.metrics.pairwise.pairwise_distances(
                data, metric = self.metric)

        if self.neighbor_graph_strategy == 'rptree':
            if isinstance(self.metric, basestring):  # TODO: python3 compatibility
                local_metric = DistanceMetric.get_metric(self.metric)
                def func(x, y):
                    result = local_metric.pairwise([x], [y])
                    return np.asscalar(result)
                distanceFunction = func
            else:
                distanceFunction = self.metric
            forest = makeForest(data, maxLeafSize = self.n_neighbors * 2, numTrees = 10, distanceFunction = distanceFunction)
            distances, indices = forest.kneighbors(data, k = self.n_neighbors)
        else:
            nn_obj = NearestNeighbors(
                n_neighbors = self.n_neighbors,
                algorithm = 'ball_tree',
                metric = self.metric,
            ).fit(data)
            distances, indices = nn_obj.kneighbors(data)

        row_new = []
        col_new = []
        val_new = []
        entries = Counter()
        for row_index in xrange(len(data)):
            for nbr_index in xrange(self.n_neighbors):
               col_index = indices[row_index, nbr_index]

               dist = distances[row_index, nbr_index]
               if dist == 0.0:
                   # Workaround: prevents sparse format from ignoring zeros
                   dist = np.nan

               a, b = min(row_index, col_index), max(row_index, col_index)
               entries[a, b] += 1

               if a == b or not self.mutual_neighbors:
                   row_new.append(row_index)
                   col_new.append(col_index)
                   val_new.append(dist)

               elif entries[a, b] == 2:
                   row_new.append(row_index)
                   col_new.append(col_index)
                   val_new.append(dist)

                   row_new.append(col_index)
                   col_new.append(row_index)
                   val_new.append(dist)

        result = coo_matrix((val_new, (row_new, col_new)), shape=(len(data), len(data)))
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
