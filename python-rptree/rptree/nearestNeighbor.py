#!/usr/bin/env python

import numpy as np
from collections import namedtuple

def selectQuantile(values, alpha):
  rank = int(round(len(values) * alpha))
  return selectRank(values, rank)

def selectRank(values, rank):
  if rank <= 0:
    return min(values)
  elif rank >= len(values) - 1:
    return max(values)
  else:
    partition = np.partition(values, rank)
    return partition[rank]

def euclidean(a, b):
  return np.linalg.norm(a - b)

def linearScanNearestNeighbor(query, data, distanceFunction):
  nearest = None
  minDistance = None
  for row in data:
    currentDistance = distanceFunction(query, row)
    if minDistance == None or currentDistance < minDistance:
      minDistance = currentDistance
      nearest = row
  return nearest
  
def randomUnitVector(n):
  v = np.random.normal(0.0, 1.0, n)
  unit = v / np.linalg.norm(v)
  return unit

def makeTree(data, orig_indices, maxLeafSize, distanceFunction):
  if len(data) <= maxLeafSize:
    return Leaf(data, orig_indices, distanceFunction)
  rule = chooseRule(data)
  leftSelections = rule(data)
  leftTree = makeTree(
      data[leftSelections],
      orig_indices[leftSelections],
      maxLeafSize,
      distanceFunction)
  rightTree = makeTree(
      data[~leftSelections],
      orig_indices[~leftSelections],
      maxLeafSize,
      distanceFunction)
  return Node(rule, leftTree, rightTree)

def repopulateTree(tree, data, orig_indices = None):
  if orig_indices is None:
    orig_indices = np.arange(len(data))
  if isinstance(tree, Leaf):
    tree.data = data
    tree.orig_indices = orig_indices
  else:
    left_mask = tree.rule(data)
    repopulateTree(tree.leftTree, data[left_mask], orig_indices[left_mask])
    repopulateTree(tree.rightTree, data[~left_mask], orig_indices[~left_mask])

def makeForest(data, maxLeafSize, numTrees, distanceFunction):
  indices = np.arange(len(data))
  trees = [makeTree(data, indices, maxLeafSize, distanceFunction)
      for i in range(numTrees)]
  return NearestNeighborForest(trees, distanceFunction)

def chooseRule(data):
  ncols = data.shape[1]
  u = randomUnitVector(ncols)
  beta = np.random.uniform(0.25, 0.75)
  proj = np.dot(data, u)
  split = selectQuantile(proj, beta)
  return Rule(u, split)

class Rule(namedtuple("Rule", ["direction", "threshold"])):
  def __call__(self, data):
    '''Apply this rule to one or more rows of data'''
    return np.dot(data, self.direction) <= self.threshold

class Leaf(object):
  def __init__(self, data, orig_indices, distanceFunction):
    self.data = data
    self.orig_indices = orig_indices
    self.distanceFunction = distanceFunction

  def getLeaves(self, rows):
    assert rows.ndim == 2
    n_rows = rows.shape[0]
    return np.full((n_rows,), self, dtype='object')

  def getLeaf(self, row):
    return self

  def nearestNeighbor(self, query):
    return linearScanNearestNeighbor(
        query, self.data, self.distanceFunction)

  def kneighbors(self, query, k):
    query = np.atleast_2d(query)
    n_query = query.shape[0]
    n_obs = self.data.shape[0]

    distances = np.ma.masked_all((n_query, k,))
    indices = np.ma.masked_all((n_query, k,))
    for i in xrange(n_query):
      qd = np.zeros((n_obs,))
      for j in xrange(n_obs):
        qd[j] = self.distanceFunction(query[i, :], self.data[j, :])
      if k >= n_obs:
        closest_ind = np.arange(n_obs)
      else:
        closest_ind = np.argpartition(qd, k)[:k]
      ii = np.argsort(qd[closest_ind])
      qd = qd[closest_ind[ii]]
      qi = self.orig_indices[closest_ind[ii]]
      distances[i, :len(qd)] = qd
      indices[i, :len(qi)] = qi

    return distances, indices

class Node(object):
  def __init__(self, rule, leftTree, rightTree):
    self.rule = rule
    self.leftTree = leftTree
    self.rightTree = rightTree

  def getLeaves(self, rows):
    assert rows.ndim == 2
    n_rows = rows.shape[0]
    rule_outcomes = self.rule(rows)
    left_rows = rows[rule_outcomes, :]
    right_rows = rows[~rule_outcomes, :]
    left_leaves = self.leftTree.getLeaves(left_rows)
    right_leaves = self.rightTree.getLeaves(right_rows)

    leaves = np.full((n_rows,), None, dtype='object')
    leaves[rule_outcomes] = left_leaves
    leaves[~rule_outcomes] = right_leaves
    return leaves

  def getLeaf(self, row):
    if self.rule(row):
      return self.leftTree.getLeaf(row)
    else:
      return self.rightTree.getLeaf(row)

  def nearestNeighbor(self, query):
    leaf = self.getLeaf(query)
    return leaf.nearestNeighbor(query)

  def kneighbors(self, query, k):
    query = np.atleast_2d(query)
    n_query = query.shape[0]

    leaves = self.getLeaves(query)

    distances = np.ma.masked_all((n_query, k), dtype='float')
    indices = np.ma.masked_all((n_query, k), dtype='int')

    for q_index in range(n_query):
      d, i = leaves[q_index].kneighbors(query[q_index, :], k)
      num_found = d.size
      distances[q_index, :d.size] = d
      indices[q_index, :d.size] = i

    return distances, indices

class NearestNeighborForest(object):
  def __init__(self, trees, distanceFunction):
    self.trees = trees
    self.distanceFunction = distanceFunction

  def nearestNeighbor(self, query):
    results = [tree.nearestNeighbor(query) for tree in self.trees]
    return linearScanNearestNeighbor(query, results, self.distanceFunction)

  def kneighbors(self, query, k):
    '''
    Find closest k neighbors, using union of results across all trees in forest
    '''
    query = np.atleast_2d(query)
    n_query = query.shape[0]

    results = [tree.kneighbors(query, k) for tree in self.trees]

    distances = np.ma.masked_all((n_query, k), dtype='float')
    indices = np.ma.masked_all((n_query, k), dtype='int')

    for q_index in range(n_query):
        qd = []
        qi = []
        indices_set = set()
        for tree_index in range(len(self.trees)):
            current_distances, current_indices = results[tree_index]
            current_distances = current_distances[q_index].compressed()
            current_indices = current_indices[q_index].compressed()
            for d, i in zip(current_distances, current_indices):
                if i not in indices_set:
                    indices_set.add(i)
                    qd.append(d)
                    qi.append(i)
        qd = np.array(qd)
        qi = np.array(qi)
        if k >= len(qd):
          closest_ind = np.arange(len(qd))
        else:
          # This handles the case where trees disagree on k-nearest set
          # (that is, there are more than k unique "candidates")
          closest_ind = np.argpartition(qd, k)[:k]
        ii = np.argsort(qd[closest_ind])
        qd = qd[closest_ind[ii]]
        qi = qi[closest_ind[ii]]

        distances[q_index, :len(qd)] = qd
        indices[q_index, :len(qi)] = qi

    return distances, indices
