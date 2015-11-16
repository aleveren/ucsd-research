#!/usr/bin/python

from __future__ import print_function
import heapq
import numpy as np
import pandas as pd
import tempfile
from collections import namedtuple, OrderedDict
import copy
import timeit
import contextlib
import shutil
import os
import pickle

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

def ensureDirectoryExists(directory):
  try:
    os.makedirs(directory)
  except OSError:
    if not os.path.isdir(directory):
      raise
  # Note: may mix new and old results when re-running with the same
  # output location. TODO: Find a way to avoid this.

def makeTree(data, maxLeafSize, distanceFunction, depthPerBatch,
    outputDir, parentPath):

  if not data.numRowsExceeds(maxLeafSize):
    print("makeTree reached leaf at {}".format(parentPath))
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
  pathsToIndices = rulesTree.mapPathsToLeaves(pathSoFar = parentPath)
  with time("partition data within '{}'".format(parentPath)):
    partitionedData = data.partitionWithIndexMap(pathsToIndices)
  unregisterTempFile(data.filename)

  # Build tree recursively
  def replaceLeafRecursive(path, previousLeaf):
    leafData = partitionedData[path]
    return makeTree(leafData, maxLeafSize, distanceFunction, depthPerBatch,
        outputDir = None, parentPath = path)

  builtTree = rulesTree.replaceLeaves(replaceLeafRecursive,
      pathSoFar = parentPath)

  if outputDir == None:
    return builtTree
  else:
    assert parentPath == ""
    ensureDirectoryExists(outputDir)

    # Move leaf data to the given directory
    def leafMover(path, previousLeaf):
      originalDataLocation = previousLeaf.lazyData.filename
      newDataLocation = outputDir + "/" + path + ".csv"
      newLazyData = previousLeaf.lazyData._replace(filename = newDataLocation)
      shutil.copy2(originalDataLocation, newDataLocation)
      return previousLeaf._replace(lazyData = newLazyData)

    persistedTree = builtTree.replaceLeaves(leafMover)

    with open(outputDir + "/tree.pkl", "w+b") as f:
      pickle.dump(persistedTree, f)

    return persistedTree

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

def makeForest(data, maxLeafSize, numTrees, distanceFunction, depthPerBatch,
    outputDir):
  if outputDir is None:
    treeOutputDirs = [None for i in range(numTrees)]
  else:
    ensureDirectoryExists(outputDir)
    treeOutputDirs = [outputDir + "/tree" + str(i) for i in range(numTrees)]
  trees = []
  for i, treeDir in enumerate(treeOutputDirs):
    with time("building tree index {}".format(i)):
      newTree = makeTree(data, maxLeafSize, distanceFunction, depthPerBatch,
          outputDir = treeDir, parentPath = "")
    trees.append(newTree)
  forest = NearestNeighborForest(trees, data.columnSlice, distanceFunction)

  if outputDir is not None:
    with open(outputDir + "/forest.pkl", "w+b") as f:
      pickle.dump(forest, f)

  return forest

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
      leftMap = OrderedDict([(pathToLeft, self.leftTree)])

    if isinstance(self.rightTree, Node):
      rightMap = self.rightTree.mapPathsToLeaves(pathToRight)
    else:
      rightMap = OrderedDict([(pathToRight, self.rightTree)])

    return OrderedDict(leftMap.items() + rightMap.items())

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

class NearestNeighborForest(namedtuple("NearestNeighborForest",
    ["trees", "columnSlice", "distanceFunction"])):

  def nearestNeighbor(self, query):
    results = [tree.nearestNeighbor(query) for tree in self.trees]
    def distanceCalculator(row):
      activeColumnsOfRow = np.array(row[self.columnSlice], dtype=float)
      return self.distanceFunction(query, activeColumnsOfRow)
    nearest = min(results, key=distanceCalculator)
    return nearest

# Provide namedtuple constructor with optional args
def LazyDiskData(
    filename,
    chunksize = 10000,
    columnSlice = slice(None)):
  return _LazyDiskData(filename, chunksize, columnSlice)

class _LazyDiskData(namedtuple("LazyDiskData",
    ["filename", "chunksize", "columnSlice"])):

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
    tempFiles = OrderedDict()
    partitionedData = OrderedDict()
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
      for rowIndex, row in chunk.iterrows():
        activeColumnsOfRow = np.array(row[self.columnSlice], dtype=float)
        currentDistance = distanceFunction(query, activeColumnsOfRow)
        if minDistance == None or currentDistance < minDistance:
          minDistance = currentDistance
          nearest = row
      chunk_index += 1
    return nearest

globalTempFiles = {}
def registerTempFile(f):
  '''Register a temp file to prevent it from being garbage collected'''
  global globalTempFiles
  globalTempFiles[f.name] = f

def unregisterTempFile(filename):
  '''Determines whether the given filename was previously registered as
     a temporary file, and if so, closes it (this ought to force deletion
     of the temp file)'''
  global globalTempFiles
  assert isinstance(filename, str)
  if filename in globalTempFiles:
    print("Removing temporary file '{}'".format(filename))
    f = globalTempFiles.pop(filename)
    f.close()

last_elapsed_time = None

@contextlib.contextmanager
def time(name = None, preannounce = True, printer = lambda x: print(x)):
  global last_elapsed_time
  extraName = " [{}]".format(name) if name is not None else ""
  start = timeit.default_timer()
  if preannounce:
    printer("Starting timer{}".format(extraName))
  try:
    yield
  finally:
    elapsed = timeit.default_timer() - start
    printer("Elapsed seconds{} = {}".format(extraName, elapsed))
    last_elapsed_time = elapsed

def getLastElapsedTime():
  return last_elapsed_time

