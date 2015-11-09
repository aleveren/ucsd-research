#!/usr/bin/python

import pandas as pd

chunks = pd.read_csv("data/accumDataRDR_all.csv", chunksize = 20000)

nrows = 0
for chunk in chunks:
  nrows += len(chunk)
  print("Cumulative number of rows: {}".format(nrows))
