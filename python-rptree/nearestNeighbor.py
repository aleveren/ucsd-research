#!/usr/bin/env python

import numpy as np

def selectQuantile(values, alpha):
  rank = round(len(values) * alpha)
  return selectRank(values, rank)

def selectRank(values, rank):
  if rank <= 0:
    return min(values)
  elif rank >= len(values) - 1:
    return max(values)
  pivot = np.random.choice(values, 1)[0]
  N = len(values)
  lower = values[values < pivot]
  higher = values[values > pivot]
  if rank < len(lower):
    return selectRank(lower, rank)
  elif rank >= N - len(higher):
    numLowerOrEqual = N - len(higher)
    return selectRank(higher, rank - numLowerOrEqual)
  else:
    return pivot

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

def makeTree(data, maxLeafSize, distanceFunction):
  if len(data) <= maxLeafSize:
    return Leaf(data, distanceFunction)
  rule = chooseRule(data)
  leftSelections = np.apply_along_axis(rule, 1, data)
  leftTree = makeTree(data[leftSelections], maxLeafSize, distanceFunction)
  rightTree = makeTree(data[np.logical_not(leftSelections)], maxLeafSize,
      distanceFunction)
  return Node(rule, leftTree, rightTree)

def makeForest(data, maxLeafSize, numTrees, distanceFunction):
  trees = [makeTree(data, maxLeafSize, distanceFunction)
      for i in range(numTrees)]
  return NearestNeighborForest(trees, distanceFunction)

def chooseRule(data):
  ncols = len(data[0, :])
  u = randomUnitVector(ncols)
  beta = np.random.uniform(0.25, 0.75)
  proj = np.apply_along_axis(lambda x: np.dot(u, x), 1, data)
  split = selectQuantile(proj, beta)
  return Rule(u, split)

class Rule(object):
  def __init__(self, direction, threshold):
    self.direction = direction
    self.threshold = threshold

  def __call__(self, row):
    '''Apply this rule to a row of data'''
    return np.dot(self.direction, row) <= self.threshold

class Leaf(object):
  def __init__(self, data, distanceFunction):
    self.data = data
    self.distanceFunction = distanceFunction

  def getLeaf(self, row):
    return self

  def nearestNeighbor(self, query):
    return linearScanNearestNeighbor(
        query, self.data, self.distanceFunction)

  def kneighbors(self, query, k):
    n_obs = self.data.shape[0]
    distances = np.zeros((n_obs,))
    for i in range(n_obs):
      distances[i] = self.distanceFunction(query, self.data[i, :])
    if k >= n_obs:
      closest_indices = np.arange(n_obs)
    else:
      closest_indices = np.argpartition(distances, k)[:k]
    return distances[closest_indices], closest_indices

class Node(object):
  def __init__(self, rule, leftTree, rightTree):
    self.rule = rule
    self.leftTree = leftTree
    self.rightTree = rightTree

  def getLeaf(self, row):
    if self.rule(row):
      return self.leftTree.getLeaf(row)
    else:
      return self.rightTree.getLeaf(row)

  def nearestNeighbor(self, query):
    leaf = self.getLeaf(query)
    return leaf.nearestNeighbor(query)

  def kneighbors(self, query, k):
    leaf = self.getLeaf(query)
    return leaf.kneighbors(query, k)

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
    if query.ndim == 2:
      # Handle batches of queries
      num_queries = query.shape[0]
      out_shape = (num_queries, k)
      distances = np.zeros(out_shape, dtype = 'float')
      indices = np.zeros(out_shape, dtype = 'int')
      for j in range(num_queries):
        d, i = self.kneighbors(query[j, :], k)
        distances[j, :] = d
        indices[j, :] = i
      return distances, indices

    elif query.ndim > 2:
      raise Exception("Query shape cannot have > 2 dimensions (found {})".format(query.ndim))

    distances = []
    indices = []
    indices_set = set()
    for tree in self.trees:
        current_distances, current_indices = tree.kneighbors(query, k)
        for d, i in zip(current_distances, current_indices):
            if i not in indices_set:
                indices_set.add(i)
                distances.append(d)
                indices.append(i)
    distances = np.array(distances)
    indices = np.array(indices)
    if len(distances) <= k:
      closest = np.arange(len(distances))
    else:
      closest = np.argpartition(distances, k)[:k]
    return distances[closest], indices[closest]
