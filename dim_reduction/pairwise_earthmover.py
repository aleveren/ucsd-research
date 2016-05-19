#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import argparse
import sklearn.metrics.pairwise as smp
from collections import OrderedDict

from dim_reduction import (extractWavelength, replaceEnforced)
from earthmover import earthmover1d

inputs = [
    "../data/CCS/subsetShots_5pct_reduced_by_density.csv",
    "../data/CCS/subsetShots_5pct_reduced.csv",
    "../data/CCS/subsetShots_5pct_reduced_uniform.csv",
    "../data/CCS/subsetShots_5pct_normalized.csv",
    ]

def main():
  sampleSize = 500

  distMatrices = []

  for filename in inputs:
    outfilename = replaceEnforced(filename, "../data/CCS/", "pairwise_earthmover_")
    np.random.seed(1)

    print "reading data: {}".format(filename)
    df = pd.read_csv(filename)
    indices = np.random.choice([0,1], sampleSize).astype(bool)
    df = df.iloc[indices, :]

    print "computing pairwise distances"
    wavelengths = map(extractWavelength, df.columns[3:])
    metric = lambda y1, y2: earthmover1d(wavelengths, y1, y2)
    D = smp.pairwise_distances(df.iloc[:, 3:], metric=metric)

    distMatrices.append(D)

    print "saving to {}".format(outfilename)
    pd.DataFrame(D).to_csv(outfilename, index=None, header=None)

  for i in range(len(inputs)-1):
    diff = distMatrices[i] - distMatrices[-1]
    frobenius = np.linalg.norm(diff, ord='fro')

    print "diff between {} and {}: {}".format(
        inputs[i], inputs[-1], frobenius)

if __name__ == "__main__":
  main()
