#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import argparse
from collections import OrderedDict

from dim_reduction import normalizeRows, extractWavelength

def main():
  df = pd.read_csv("../data/CCS/subsetShots_5pct.csv", nrows=1000)
  df = normalizeRows(df, useWidths=False)
  nrow = df.shape[0]
  rowsum = df.iloc[:,3:].sum(axis=0)
  cumulative = rowsum.cumsum()

  wavelengths = map(extractWavelength, df.columns[3:])

  k = 50  # desired number of features
  alphas = [nrow * (2*j - 1) / (2.0 * k) for j in range(1, k+1)]
  new_alphas = []
  new_xs = []
  alpha_index = 0
  w_index = 0
 
  while True:
    if alpha_index >= len(alphas) or w_index >= len(wavelengths):
      break
    if cumulative[w_index] >= alphas[alpha_index]:
      if len(new_xs) == 0 or wavelengths[w_index] != new_xs[-1]:
        new_xs.append(wavelengths[w_index])
        new_alphas.append(alphas[alpha_index])
      alpha_index += 1
    else:
      w_index += 1

  print new_xs
  print len(new_xs)
  print cumulative[-1] / float(nrow)

  plt.figure()
  ax = plt.gca()
  for a in new_alphas:
    ax.axhline(y=a / float(nrow), linestyle='-', color='#cccccc')
  for x in new_xs:
    ax.axvline(x=x, linestyle='-', color='#cccccc')
  #ax.twinx().plot(wavelengths, rowsum / float(nrow), color=(0.5,0.7,0.5,0.3))
  ax.plot(wavelengths, cumulative / float(nrow))

  plt.show()

if __name__ == "__main__":
  main()
