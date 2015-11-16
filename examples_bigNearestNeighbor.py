#!/usr/bin/python

import numpy as np
from bigNearestNeighbor import (LazyDiskData, randomUnitVector, makeForest, euclidean, time, getLastElapsedTime)
import sys
import datetime

np.random.seed(1)

if len(sys.argv) > 1:
  analysis = sys.argv[1]
else:
  analysis = "sim"

print("Analysis: {}".format(analysis))

if analysis == "subset":
  exampleData = LazyDiskData("data/accumDataRDR_subset.csv",
      columnSlice = slice(3, None))
  query = randomUnitVector(6144)
  numTrees = 1
elif analysis == "full":
  exampleData = LazyDiskData("data/accumDataRDR_all.csv",
      columnSlice = slice(3, None))
  query = randomUnitVector(6144)
  numTrees = 1
elif analysis == "sim":
  exampleData = LazyDiskData("data/testdata.csv",
      columnSlice = slice(1, None))
  query = np.zeros(10)
  numTrees = 10
else:
  raise Exception("Unrecognized analysis: {}".format(analysis))

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
outputDir = "forests/forest_" + analysis + "_" + timestamp

with time("naive linear scan query"):
  naiveResult = exampleData.linearScanNearestNeighbor(query,
      distanceFunction = euclidean)
  print(naiveResult)
naiveRuntime = getLastElapsedTime()

with time("build trees"):
  forest = makeForest(exampleData, maxLeafSize = 500, numTrees = numTrees,
      distanceFunction = euclidean, depthPerBatch = 3,
      outputDir = outputDir)

with time("run query"):
  result = forest.nearestNeighbor(query)
  print(result)

print("For comparison, naive result:\n{}\nnaive elapsed = {}".format(
    naiveResult, naiveRuntime))

print("Forest saved at: {}".format(outputDir))
