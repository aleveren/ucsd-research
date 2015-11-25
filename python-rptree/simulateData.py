#!/usr/bin/python

import numpy as np
import pandas as pd

def main(filename, seed = 1):
  if seed is not None:
    np.random.seed(seed)

  numRows = 5000
  dims = 10
  d = np.random.uniform(0, 1, [numRows, dims])
  d = np.hstack((np.zeros([numRows, 1]), d))

  colnames = ["index"] + ["X"+str(i) for i in range(dims)]

  M = np.sqrt(dims) + 1

  for i in range(numRows):
    j = np.random.randint(dims)
    d[i, 0] = i
    d[i, j+1] = M

  d[np.random.randint(numRows), 1:] = np.ones(dims)

  df = pd.DataFrame(d, columns = colnames)
  df.to_csv(filename, header=True, index=False)

if __name__ == "__main__":
  filename = "../data/testdata.csv"
  main(filename)
