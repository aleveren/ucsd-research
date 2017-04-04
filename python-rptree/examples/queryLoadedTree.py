#!/usr/bin/env python

import pickle
import numpy as np
import sys

import context
from rptree.bigNearestNeighbor import time, loadForest

if len(sys.argv) > 1:
  filename = sys.argv[1]
else:
  print("Please provide a filename")
  sys.exit(1)

with time("load forest"):
  forest = loadForest(filename)
query = np.zeros(len(forest.trees[0].rule.direction))
with time("run query"):
  result = forest.nearestNeighbor(query)
print(result)
