#!/usr/bin/python

import pickle
import numpy as np
import sys

if len(sys.argv) > 1:
  filename = sys.argv[1]
else:
  print("Please provide a filename")
  sys.exit(1)

with open(filename) as f:
  forest = pickle.load(f)
query = np.zeros(len(forest.trees[0].rule.direction))
result = forest.nearestNeighbor(query)
print(result)
