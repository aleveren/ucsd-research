#!/usr/bin/python

import pandas as pd
from nearestNeighbor import makeForest, euclidean

limit = 2000
d = pd.read_csv("data/accumDataRDR_all.csv", chunksize=limit+1).get_chunk()
data = d.iloc[:-1, 3:].values
query = d.iloc[-1, 3:].values

print("Data size: {}".format(data.shape))
print("Building trees")
forest = makeForest(data, maxLeafSize = 100, numTrees = 10,
    distanceFunction = euclidean)
print("Running query")
result = forest.nearestNeighbor(query)
print("Nearest to {}:\n{}".format(query, result))

# A few results:
# N = 1000 ==> 12 seconds (total), <1GB memory usage
# N = 2000 ==> 25 seconds (total), ~1.5GB memory usage
# N = 5000 ==> 75 seconds (total), ~3GB memory usage
