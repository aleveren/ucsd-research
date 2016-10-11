#!/usr/bin/env python

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import SpectralClustering
import sklearn.metrics
import sys
import re
import argparse
import logging

from tree_util import Tree


_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())


class HierClust(object):
    def __init__(self,
            n_neighbors = 20,
            threshold_for_subset = 500,
            representative_growth_exponent = 1/3.0,
            sigma_similarity = 1.0,
            leaf_size = 1):
        self.n_neighbors = n_neighbors
        self.threshold_for_subset = threshold_for_subset
        self.representative_growth_exponent = representative_growth_exponent
        self.sigma_similarity = sigma_similarity
        self.leaf_size = leaf_size

    def fit(self, data, feature_columns = None):
        '''
        Generates a hierarchical clustering on the given data
        '''
        if feature_columns is None:
            feature_columns = slice(None, None, None)
        elif isinstance(feature_columns, basestring):
            feature_columns = [i for i in range(len(data.columns))
                if re.match(feature_columns, data.columns[i])]
        data = np.asarray(data)[:, feature_columns]
        orig_indices = np.arange(len(data))

        tree = self._fit_helper(data, orig_indices, tree_path = '',
            num_leaves_done = 0)
        assignments = self._get_assignments(tree)
        tree_paths = np.array([p for i, p in assignments])

        return tree, tree_paths

    def _fit_helper(self, data, orig_indices, tree_path, num_leaves_done):
        '''
        Recursive helper function for partitioning a dataset into 2 parts
        '''
        log_msg = "Partitioning {} observations " \
            "(tree_path = {}, num_leaves_done = {})" \
            .format(len(data), tree_path, num_leaves_done)
        if len(data) > self.threshold_for_subset:
            _logger.info(log_msg)
        else:
            _logger.debug(log_msg)

        if len(data) <= 1 or len(data) <= self.leaf_size:
            return Tree.leaf(data = {
                "orig_indices": orig_indices,
                "tree_path": tree_path,
            })

        if len(data) == 2:
            partition = np.array([0, 1])
        elif len(data) <= self.threshold_for_subset:
            partition = self._small_partition(data)
        else:
            partition = self._large_partition(data)

        num0 = len(partition[partition == 0])
        num1 = len(partition[partition == 1])
        _logger.debug("Partition result: #0: {}, #1: {}".format(num0, num1))

        children = []
        for label in [0, 1]:
            data_subset = data[partition == label]
            orig_indices_subset = orig_indices[partition == label]
            subtree = self._fit_helper(
                data_subset,
                orig_indices_subset,
                tree_path + str(label),
                num_leaves_done = num_leaves_done)
            children.append(subtree)
            num_leaves_done += len(data_subset)

        return Tree(children = children, data = {
            "orig_indices": orig_indices,
            "tree_path": tree_path,
        })

    def _small_partition(self, data):
        _logger.debug("Running _small_partition on %s observations", len(data))

        similarity = self._get_similarity(data)
        spc_obj = SpectralClustering(n_clusters = 2, affinity = 'precomputed',
            assign_labels = 'discretize')
        partition = spc_obj.fit_predict(similarity)

        return partition

    def _large_partition(self, data):
        _logger.debug("Running _large_partition on %s observations", len(data))

        num_representatives = self._num_reps(len(data))
        if num_representatives < len(data):
            # Generate a random subset
            mask = [True for i in range(num_representatives)] + \
                [False for i in range(len(data) - num_representatives)]
            mask = np.random.permutation(mask)
            subset = data[mask, :]
        else:
            subset = data

        # Partition the small set
        small_partition = self._small_partition(subset)

        # Use KNN classifier to extend partition to full data
        _logger.debug("Running KNN classifier on %s observations", len(data))
        nn_classifier = KNeighborsClassifier(
            n_neighbors = self.n_neighbors,
            algorithm = 'ball_tree',
            metric = 'euclidean',
        ).fit(subset, small_partition)
        full_partition = nn_classifier.predict(data)
        return full_partition

    def _get_assignments(self, tree):
        '''
        Turns a tree into a column of cluster id's ("tree paths")
        '''
        if len(tree.children) == 0:
            indices = tree.data["orig_indices"]
            path = tree.data["tree_path"]
            if len(indices) > 1:
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

    def _get_similarity(self, data):
        '''
        Generate a similarity matrix for the given data
        '''
        if len(data) <= self.n_neighbors:
            return self._get_dense_similarity(data)
        else:
            return self._get_sparse_similarity(data)

    def _nn_result_to_sparse_similarity(self, distances, indices):
        '''
        Convert the result of K-nearest-neighbors into a sparse similarity
        matrix
        '''
        num_obs = len(indices)
        rows_nested = [[i for k in range(self.n_neighbors)]
            for i in range(num_obs)]
        rows = np.array(rows_nested).flatten()
        cols = indices.flatten()
        scaling_factor = 2.0 * self.sigma_similarity ** 2
        similarities = np.exp(-distances ** 2 / scaling_factor).flatten()

        # Store result in a sparse, symmetric matrix
        similarities = coo_matrix((similarities, (rows, cols)),
            shape = (len(indices), len(indices)))
        similarities = 0.5 * similarities + 0.5 * similarities.T
        return similarities

    def _get_sparse_similarity(self, data):
        '''
        Generate a sparse similarity matrix via nearest neighbors
        '''
        nn_obj = NearestNeighbors(
            n_neighbors = self.n_neighbors,
            algorithm = 'ball_tree',
            metric = 'euclidean',
        ).fit(data)
        distances, indices = nn_obj.kneighbors(data)
        return self._nn_result_to_sparse_similarity(distances, indices)

    def _get_dense_similarity(self, data):
        '''
        Generate a sparse similarity matrix via nearest neighbors
        '''
        scaling_factor = 2.0 * self.sigma_similarity ** 2
        distances = sklearn.metrics.pairwise.pairwise_distances(
            data, metric = 'euclidean')
        dense_similarity = np.exp(-distances ** 2 / scaling_factor)
        return dense_similarity

    def _num_reps(self, n):
        '''
        A function that grows like f(n) but transitions to n below n0.
        Used for picking the number of representatives for spectral clustering.
        '''
        alpha = float(self.representative_growth_exponent)
        n0 = float(self.threshold_for_subset)
        def f(x): return x ** alpha
        def f_prime(x): return alpha * x ** (alpha - 1)
        a = n0 - f(n0) / f_prime(n0)
        b = 1.0 / f_prime(n0)
        if n < n0:
            return int(n)
        else:
            return int(np.ceil(a + b * f(n)))


def numeric_logging_level(level_string):
    num_level = getattr(logging, level_string.upper(), None)
    if not isinstance(num_level, int):
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
    args = parser.parse_args(argv)

    data = pd.read_csv(args.input)

    logging.basicConfig(
        level = numeric_logging_level(args.log),
        format = '%(asctime)s %(message)s')

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    hc = HierClust()
    tree, assignments = hc.fit(
        data = data,
        feature_columns = args.feature_columns)

    if args.output is None:
        for i, path in enumerate(assignments):
            print("{}: {}".format(i, path))
    else:
        _logger.info("Outputting to '%s' (column = '%s')",
            args.output, args.output_column)
        data[args.output_column] = assignments
        data.to_csv(args.output, index = False)


if __name__ == '__main__':
    main(sys.argv[1:])