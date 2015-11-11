#!/usr/bin/python

import numpy as np
import pandas as pd
import tempfile

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
  if not data.numRowsExceeds(maxLeafSize):
    return LazyLeaf(data, distanceFunction)
  rule = chooseRule(data)
  leftSelections = data.applyToRows(rule)
  leftTree = makeTree(data.subset(leftSelections), maxLeafSize,
      distanceFunction)
  rightTree = makeTree(data.subset(np.logical_not(leftSelections)), maxLeafSize,
      distanceFunction)
  return Node(rule, leftTree, rightTree)

def makeForest(data, maxLeafSize, numTrees, distanceFunction):
  trees = [makeTree(data, maxLeafSize, distanceFunction)
      for i in range(numTrees)]
  return NearestNeighborForest(trees, distanceFunction)

def chooseRule(data):
  ncols = data.numActiveColumns()
  u = randomUnitVector(ncols)
  beta = np.random.uniform(0.25, 0.75)
  proj = data.applyToRows(lambda x: np.dot(u, x))
  split = selectQuantile(proj, beta)
  return Rule(u, split)

class Rule(object):
  def __init__(self, direction, threshold):
    self.direction = direction
    self.threshold = threshold

  def __call__(self, row):
    '''Apply this rule to a row of data'''
    return np.dot(self.direction, row) <= self.threshold

class LazyLeaf(object):
  def __init__(self, lazyData, distanceFunction):
    self.lazyData = lazyData
    self.distanceFunction = distanceFunction

  def getLeaf(self, row):
    return self

  def nearestNeighbor(self, query):
    # At leaf nodes, we assume we can read the entire set of observations
    dataFrame = pd.read_csv(self.lazyData.filename)
    matrix = dataFrame.iloc[:, self.lazyData.columnSlice].values
    return linearScanNearestNeighbor(query, matrix, self.distanceFunction)

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

class NearestNeighborForest(object):
  def __init__(self, trees, distanceFunction):
    self.trees = trees
    self.distanceFunction = distanceFunction

  def nearestNeighbor(self, query):
    results = [tree.nearestNeighbor(query) for tree in self.trees]
    return linearScanNearestNeighbor(query, results, self.distanceFunction)

MAX_CHUNKS = 3  # TODO: remove this limit

class LazyDiskData(object):
  def __init__(
      self,
      filename,
      chunksize = 1000,
      columnSlice = slice(None)):
    self.filename = filename
    self.chunksize = chunksize
    self.columnSlice = columnSlice

  def dataRef(self):
    return pd.read_csv(self.filename, chunksize = self.chunksize)

  def matrixGenerator(self):
    d = self.dataRef()
    for chunk in d:
      matrix = chunk.iloc[:, self.columnSlice].values
      yield matrix

  def applyToRows(self, func):
    result = []
    chunk_index = 0
    for matrix in self.matrixGenerator():
      if MAX_CHUNKS is not None and chunk_index >= MAX_CHUNKS:
        break
      currentResult = np.apply_along_axis(func, 1, matrix)
      result.extend(currentResult)
      chunk_index += 1
    return np.array(result)

  def numActiveColumns(self):
    d = self.dataRef()
    chunk = d.get_chunk(1)
    matrix = chunk.iloc[:, self.columnSlice].values
    return matrix.shape[1]

  def numRowsExceeds(self, targetSize):
    d = self.dataRef()
    totalRows = 0
    for chunk in d:
      totalRows += chunk.shape[0]
      if totalRows > targetSize:
        return True
    return False

  def subset(self, mask):
    d = self.dataRef()
    temp = tempfile.NamedTemporaryFile()
    registerTempFile(temp)
    tempName = temp.name
    chunk_index = 0
    for chunk in d:
      if MAX_CHUNKS is not None and chunk_index >= MAX_CHUNKS:
        break
      if chunk_index == 0:
        chunk.iloc[0:0, :].to_csv(tempName, mode="w", header=True, index=False)
      start = chunk_index * self.chunksize
      stop = start + self.chunksize
      submask = mask[start:stop]
      subset = chunk.iloc[submask, :]
      subset.to_csv(tempName, mode="a", header=False, index=False)
      chunk_index += 1
    return LazyDiskData(tempName, chunksize = self.chunksize,
        columnSlice = self.columnSlice)

tempFiles = []
def registerTempFile(f):
  '''Register a temp file to prevent it from being garbage collected'''
  global tempFiles
  tempFiles.append(f)

if __name__ == "__main__":
  np.random.seed(1)

  exampleData = LazyDiskData("data/accumDataRDR_all.csv", 
      columnSlice = slice(3, None))
  
  u = randomUnitVector(6144)

  result = exampleData.applyToRows(lambda x: np.dot(x, u))
  print(result)
  print(result.shape)
  
  print("Finding quantile")
  quantile = selectQuantile(result, 0.25)
  print("quantile = {}".format(quantile))
  
  rule = Rule(u, quantile)
  result = exampleData.applyToRows(rule)
  print(result)
  print(result.shape)

  print("Building trees")
  forest = makeForest(exampleData, maxLeafSize = 100, numTrees = 1,
      distanceFunction = euclidean)
  print("Running query")
  result = forest.nearestNeighbor(u)
  print(result)
