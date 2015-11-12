#!/usr/bin/python

import numpy as np
import pandas as pd

def main(filename, seed = 1):
  if seed is not None:
    np.random.seed(seed)

  numRows = 5000
  dims = 10
  d = np.random.uniform(0, 1, [numRows, dims])
  
  M = np.sqrt(dims) + 1
  
  for i in range(numRows):
    j = np.random.randint(dims)
    d[i, j] = M
  
  d[np.random.randint(numRows), :] = np.ones(dims)
  
  df = pd.DataFrame(d, columns = ["X"+str(i) for i in range(dims)])
  df.to_csv(filename, header=True, index=False)

if __name__ == "__main__":
  filename = "data/testdata.csv"
  main(filename)
