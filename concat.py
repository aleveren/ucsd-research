#!/usr/bin/python

import os

maxNum = 5666
batchSize = 50
fileNumbers = []
n = batchSize
while n <= maxNum:
  fileNumbers.append(n)
  n += batchSize
fileNumbers.append(maxNum)

files = ["data/accumDataRDR_batch{}.csv.bz2".format(n) for n in fileNumbers]

outfile = "data/accumDataRDR_all.csv"
print("Getting header")
os.system("bzip2 -c -d {} | head -n 1 > {}".format(files[0], outfile))
for f in files:
  print("Concatenating {}".format(f))
  os.system("bzip2 -c -d {} | tail -n +2 >> {}".format(f, outfile))
