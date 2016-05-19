#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import argparse
from collections import OrderedDict

from dim_reduction import (
    normalizeRows,
    extractWavelength,
    reduceDimRow,
    nearestColumnMapping,
)

def main():
  print "reading data"
  df = pd.read_csv("../data/CCS/subsetShots_5pct.csv")
  print "preprocessing data"
  df = normalizeRows(df, useWidths=False)

  k = 50  # desired number of features
  wavelengths = map(extractWavelength, df.columns[3:])
  new_xs = np.linspace(min(wavelengths), max(wavelengths), k)

  print new_xs

  print "reducing data"
  mapping = nearestColumnMapping(new_xs, df.columns[3:])
  newData = df.apply(reduceDimRow, axis = 1, centers = mapping)
  print "saving reduced data"
  newData.to_csv("../data/CCS/subsetShots_5pct_reduced_uniform.csv", index=None)

if __name__ == "__main__":
  main()
