#!/usr/bin/python

from __future__ import print_function
import heapq
import numpy as np
import pandas as pd
import tempfile
from collections import namedtuple
import copy
import timeit
import contextlib

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
  with time("projections"):
    projections = data.applyToRows(multiDotProduct)
  numRows = projections.shape[0]
  indices = np.arange(numRows).reshape([numRows, 1])
  projections = np.hstack((indices, projections))

  quantiles = np.random.uniform(0.25, 0.75, numVectors)

  # Compute split points
  with time("compute split points"):
    rulesTree = projectionsToRulesTree(projections, vectors, quantiles,
        maxLeafSize, columnIndex = 1)

  del projections
  del indices

  # Partition the data
  assert isinstance(rulesTree, Node)
  pathsToIndices = rulesTree.mapPathsToLeaves()
  with time("partition data"):
    partitionedData = data.partitionWithIndexMap(pathsToIndices)

  # Build tree recursively
  def replaceLeafRecursive(path, previousLeaf):
    leafData = partitionedData[path]
    return makeTree(leafData, maxLeafSize, distanceFunction, depthPerBatch)

  return rulesTree.replaceLeaves(replaceLeafRecursive)

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
  return NearestNeighborForest(trees, data, distanceFunction)

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
    # At leaf nodes, we assume we can search the entire set of observations
    return self.lazyData.linearScanNearestNeighbor(query, self.distanceFunction)

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

  def replaceLeaves(self, replacer, pathSoFar = ""):
    '''Creates a modified copy of this tree by replacing leaves according to
       the given function.  The provided function should take two arguments:
       a "path" representing the left/right sequence leading to a leaf node,
       and the current leaf.'''
    pathToLeft = pathSoFar + "L"
    pathToRight = pathSoFar + "R"

    if isinstance(self.leftTree, Node):
      newLeft = self.leftTree.replaceLeaves(replacer, pathToLeft)
    else:
      newLeft = replacer(pathToLeft, self.leftTree)

    if isinstance(self.rightTree, Node):
      newRight = self.rightTree.replaceLeaves(replacer, pathToRight)
    else:
      newRight = replacer(pathToRight, self.rightTree)

    return Node(copy.deepcopy(self.rule), newLeft, newRight)

class NearestNeighborForest(object):
  def __init__(self, trees, data, distanceFunction):
    self.trees = trees
    self.data = data
    self.distanceFunction = distanceFunction

  def nearestNeighbor(self, query):
    results = [tree.nearestNeighbor(query) for tree in self.trees]
    def distanceCalculator(row):
      activeColumnsOfRow = row[self.data.columnSlice]
      return self.distanceFunction(query, activeColumnsOfRow)
    nearest = min(results, key=distanceCalculator)
    return nearest

class LazyDiskData(object):
  def __init__(
      self,
      filename,
      chunksize = 10000,
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

  def partitionWithIndexMap(self, pathsToIndices):
    uniquePaths = pathsToIndices.keys()
    tempFiles = {}
    partitionedData = {}
    toMerge = []

    for path in uniquePaths:
      temp = tempfile.NamedTemporaryFile()
      registerTempFile(temp)
      tempName = temp.name
      print("path = {}, tempName = {}".format(path, tempName))
      tempFiles[path] = tempName
      partitionedData[path] = LazyDiskData(tempName,
          chunksize = self.chunksize, columnSlice = self.columnSlice)
      toMerge.append([(i, path) for i in pathsToIndices[path]])

    mergedPaths = [path for i, path in heapq.merge(*toMerge)]
  
    # Populate data files
    d = self.dataRef()
    chunk_index = 0
    for chunk in d:
      if chunk_index == 0:
        # Initialize each partition with a header
        for path, temp in tempFiles.items():
          chunk.iloc[0:0, :].to_csv(temp, mode="w", header=True, index=False)
      def grouper(rowIndex):
        return mergedPaths[rowIndex + chunk_index * self.chunksize]
      groups = chunk.groupby(grouper)
      for path, group in groups:
        currentTempFile = tempFiles[path]
        group.to_csv(currentTempFile, mode="a", header=False, index=False)
      chunk_index += 1
    
    return partitionedData

  def linearScanNearestNeighbor(self, query, distanceFunction):
    nearest = None
    minDistance = None
    chunk_index = 0
    for chunk in self.dataRef():
      for indexAndRow in chunk.itertuples():
        row = np.array(indexAndRow[1:])
        activeColumnsOfRow = row[self.columnSlice]
        currentDistance = distanceFunction(query, activeColumnsOfRow)
        if minDistance == None or currentDistance < minDistance:
          minDistance = currentDistance
          nearest = row
      chunk_index += 1
    return nearest

tempFiles = []
def registerTempFile(f):
  '''Register a temp file to prevent it from being garbage collected'''
  global tempFiles
  tempFiles.append(f)

@contextlib.contextmanager
def time(name = None, preannounce = True, printer = lambda x: print(x)):
  extraName = " [{}]".format(name) if name is not None else ""
  start = timeit.default_timer()
  if preannounce:
    printer("Starting timer{}".format(extraName))
  try:
    yield
  finally:
    elapsed = timeit.default_timer() - start
    printer("Elapsed seconds{} = {}".format(extraName, elapsed))

if __name__ == "__main__":
  np.random.seed(1)

  with time("load"):
    exampleData = LazyDiskData("data/accumDataRDR_all.csv", 
        columnSlice = slice(3, None))
  
  u = randomUnitVector(6144)

  #with time("apply dot product"):
  #  result = exampleData.applyToRows(lambda x: np.dot(x, u))
  #  print(result)
  #  print(result.shape)

  #with time("find quantile"):
  #  quantile = selectQuantile(result, 0.25)
  #  print("quantile = {}".format(quantile))

  #with time("apply rule"):
  #  rule = Rule(u, quantile)
  #  result = exampleData.applyToRows(rule)
  #  print(result)
  #  print(result.shape)

  with time("build trees"):
    forest = makeForest(exampleData, maxLeafSize = 500, numTrees = 1,
        distanceFunction = euclidean, depthPerBatch = 3)

  with time("run query"):
    result = forest.nearestNeighbor(u)
    print(result)
