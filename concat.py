#!/usr/bin/python

import os

maxNum = 5666
batchSize = 50
fileNumbers = range(batchSize, maxNum+1, batchSize)
if maxNum % batchSize != 0:
  fileNumbers.append(maxNum)

files = ["data/accumDataRDR_batch{}.csv.bz2".format(n) for n in fileNumbers]

outfile = "data/accumDataRDR_all.csv"
print("Getting header")
os.system("bzip2 -c -d {} | head -n 1 > {}".format(files[0], outfile))
for f in files:
  print("Concatenating {}".format(f))
  os.system("bzip2 -c -d {} | tail -n +2 >> {}".format(f, outfile))
