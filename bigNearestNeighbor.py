#!/usr/bin/python

import numpy as np
import pandas as pd
import tempfile
from collections import namedtuple

def selectQuantile(values, alpha):
  rank = int(len(values) * alpha) - 1
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

def makeTree(data, maxLeafSize, distanceFunction, depthPerBatch):
  if not data.numRowsExceeds(maxLeafSize):
    return LazyLeaf(data, distanceFunction)

  numCols = data.numActiveColumns()

  # Compute multiple projections in a single pass through the data
  numVectors = 2 ** (depthPerBatch + 1) - 1
  vectors = [randomUnitVector(numCols) for i in range(numVectors)]
  multiDotProduct = lambda x: [np.dot(v, x) for v in vectors]
  projections = data.applyToRows(multiDotProduct)
  indices = np.arange(0, projections.shape[0], [projections.shape[0], 1])
  projections = np.hstack(indices, projections)

  quantiles = np.random.uniform(0.25, 0.75, numVectors)

  # Compute split points
  rulesTree = projectionsToRulesTree(projections, vectors, quantiles,
      maxLeafSize, columnIndex = 1)

  assert isinstance(rulesTree, Node)

  del projections
  del indices

  # TODO: partition the data
  # TODO: build tree recursively
  return None

def projectionsToRulesTree(
    projections, directions, quantiles, maxLeafSize, columnIndex):
  # Note: column 0 stores row-index; actual projections start at column 1
  numRows, numColumns = projections.shape
  assert numColumns == len(quantiles) + 1
  assert numColumns == len(directions) + 1

  if columnIndex >= numColumns or numRows <= maxLeafSize:
    # This will be a leaf node in the intermediate result, either because we
    # don't have enough precomputed projections, or because the number of rows
    # is small enough.
    # Note: The resulting tree contains a vector of indices at each leaf.
    return projections[:, 0]

  quantile = quantiles[columnIndex - 1]
  direction = directions[columnIndex - 1]
  split = selectQuantile(projections[:, columnIndex], quantile)
  rule = Rule(direction, split)

  leftMask = projections[:, columnIndex] <= split
  leftProjections = projections[leftMask, :]
  rightMask = projections[:, columnIndex] > split
  rightProjections = projections[rightMask, :]

  leftChildIndex = 2*columnIndex
  rightChildIndex = 2*columnIndex + 1

  leftTree = projectionsToRulesTree(leftProjections, directions, quantiles,
      maxLeafSize, leftChildIndex)
  rightTree = projectionsToRulesTree(rightProjections, directions, quantiles,
      maxLeafSize, rightChildIndex)

  return Node(rule, leftTree, rightTree)

def makeForest(data, maxLeafSize, numTrees, distanceFunction, depthPerBatch):
  trees = [makeTree(data, maxLeafSize, distanceFunction, depthPerBatch)
      for i in range(numTrees)]
  return NearestNeighborForest(trees, distanceFunction)

class Rule(namedtuple("Rule", ["direction", "threshold"])):
  def __call__(self, row):
    '''Apply this rule to a row of data'''
    return np.dot(self.direction, row) <= self.threshold

class LazyLeaf(namedtuple("LazyLeaf", ["lazyData", "distanceFunction"])):
  def leafIter(self):
    yield self

  def getLeaf(self, row):
    return self

  def nearestNeighbor(self, query):
    # At leaf nodes, we assume we can read the entire set of observations
    dataFrame = pd.read_csv(self.lazyData.filename)
    matrix = dataFrame.iloc[:, self.lazyData.columnSlice].values
    return linearScanNearestNeighbor(query, matrix, self.distanceFunction)

class Node(namedtuple("Node", ["rule", "leftTree", "rightTree"])):
  def leafIter(self):
    for leaf in self.leftTree.leafIter():
      yield leaf
    for leaf in self.rightTree.leafIter():
      yield leaf

  def getLeaf(self, row):
    if self.rule(row):
      return self.leftTree.getLeaf(row)
    else:
      return self.rightTree.getLeaf(row)

  def nearestNeighbor(self, query):
    leaf = self.getLeaf(query)
    return leaf.nearestNeighbor(query)

  def mapPathsToLeaves(self, pathSoFar = ""):
    pathToLeft = pathSoFar + "L"
    pathToRight = pathSoFar + "R"
    if isinstance(self.leftTree, Node):
      leftMap = self.leftTree.mapPathsToLeaves(pathToLeft)
    else:
      leftMap = {pathToLeft: self.leftTree}
    if isinstance(self.rightTree, Node):
      rightMap = self.rightTree.mapPathsToLeaves(pathToRight)
    else:
      rightMap = {pathToRight: self.rightTree}
    return dict(leftMap.items() + rightMap.items())

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
  forest = makeForest(exampleData, maxLeafSize = 1000, numTrees = 1,
      distanceFunction = euclidean, depthPerBatch = 2)
  print("Running query")
  result = forest.nearestNeighbor(u)
  print(result)
