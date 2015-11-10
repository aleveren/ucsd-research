#!/usr/bin/python

import numpy as np

def main():
  np.random.seed(1)
  alpha = 1/4.0
  a = np.arange(0, 100, 10)
  np.random.shuffle(a)
  
  for x in a: print(x)
  for i, x in enumerate(sorted(a)): print("{}, {}".format(i, x))
  for i in range(-1, len(a) + 1):
    print("Select {} = {}".format(i, selectRank(a, i)))

  v = randomUnitVector(7000)
  print("v = {}, with norm {}".format(v, np.linalg.norm(v)))

  data = np.random.uniform(0, 1, [1000, 100])
  forest = makeForest(data, n0 = 100, numTrees = 10,
      distanceFunction = euclidean)
  query = np.random.uniform(0, 1, 100)
  result = forest.nearestNeighbor(query)
  print("Nearest to {}: {}".format(query, result))

  M = 100
  data = np.array([[1.0, 1.0, 1.0]])
  for i in range(3000):
    r1 = np.random.uniform(0, 1)
    r2 = np.random.uniform(0, 1)
    if i % 3 == 0:
      data = np.vstack((data, np.array([[M, r1, r2]])))
    elif i % 3 == 1:
      data = np.vstack((data, np.array([[r1, M, r2]])))
    elif i % 3 == 2:
      data = np.vstack((data, np.array([[r1, r2, M]])))
  print(data)
  print("Building trees")
  forest = makeForest(data, n0 = 100, numTrees = 10,
      distanceFunction = euclidean)
  print("Finished building trees")
  print("Running query")
  query = np.array([0.0, 0.0, 0.0])
  result = forest.nearestNeighbor(query)
  print("Nearest to {}: {}".format(query, result))

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

def makeTree(data, n0, distanceFunction):
  if len(data) < n0:
    return Leaf(data, distanceFunction)
  rule = chooseRule(data)
  leftSelections = np.apply_along_axis(rule, 1, data)
  leftTree = makeTree(data[leftSelections], n0, distanceFunction)
  rightTree = makeTree(data[np.logical_not(leftSelections)], n0,
      distanceFunction)
  return Node(rule, leftTree, rightTree)

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

def makeForest(data, n0, numTrees, distanceFunction):
  trees = [makeTree(data, n0, distanceFunction) for i in range(numTrees)]
  return NearestNeighborForest(trees, distanceFunction)

class NearestNeighborForest(object):
  def __init__(self, trees, distanceFunction):
    self.trees = trees
    self.distanceFunction = distanceFunction

  def nearestNeighbor(self, query):
    results = [tree.nearestNeighbor(query) for tree in self.trees]
    return linearScanNearestNeighbor(query, results, self.distanceFunction)

if __name__ == "__main__":
  main()
