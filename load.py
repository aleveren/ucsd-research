#!/usr/bin/python

import pandas as pd

maxNum = 5666
batchSize = 50
fileNumbers = []
n = batchSize
while n <= maxNum:
  fileNumbers.append(n)
  n += batchSize
fileNumbers.append(maxNum)

files = ["data/accumDataRDR_batch{}.csv.bz2".format(n) for n in fileNumbers]

d = pd.read_csv(files[0], compression = "bz2")
columns = d.columns

print(columns)

nrows = 0
for filename in files:
  print("Reading {}".format(filename))
  d = pd.read_csv(filename, compression = "bz2")
  nrows += len(d)
  print("  cumulative number of rows: {}".format(nrows))
