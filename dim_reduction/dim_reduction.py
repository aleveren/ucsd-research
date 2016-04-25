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

def replaceEnforced(s, toRemove, toAdd):
  newString = s.replace(toRemove, toAdd)
  assert newString != s
  return newString

def extractWavelength(col):
  return float(re.search(r"wavelength_(\d+\.?\d*)", col).group(1))

def normalizeRows(d):
  unchanged = d.iloc[:, :3]
  toChange = d.iloc[:, 3:]
  # set negative entries to 0
  no_neg = toChange.where(toChange >= 0, other=0)
  # normalize rows to sum to 1
  sums = no_neg.sum(axis=1)
  normalized = no_neg.div(sums, axis=0)
  return pd.concat([unchanged, normalized], axis=1)

def nearestNeighborMapping(centers, allPoints):
  centers = np.reshape(centers, [len(centers), 1])
  allPoints = np.reshape(allPoints, [len(allPoints), 1])
  nbrs = NearestNeighbors(n_neighbors=1).fit(centers)
  nearestIndices = list(nbrs.kneighbors(allPoints)[1].flatten())
  nearest = [centers[i,0] for i in nearestIndices]
  mapping = OrderedDict(zip(allPoints.flatten(), nearest))
  return mapping

def nearestColumnMapping(centers, cols):
  allPoints = map(extractWavelength, cols)
  floatMapping = nearestNeighborMapping(centers, allPoints)
  newCols = ["wavelength_{:.3f}".format(floatMapping[p]) for p in allPoints]
  mapping = OrderedDict(zip(cols, newCols))
  return mapping

def reduceDimRow(row, centers):
  newRow = OrderedDict()
  newRow.update(row.iloc[:3].to_dict())
  for newCol in centers.values():
    newRow[newCol] = 0
  for oldCol, intensity in row.iloc[3:].to_dict().items():
    newRow[centers[oldCol]] += intensity
  return pd.Series(newRow)

parser = argparse.ArgumentParser()
parser.add_argument('--force-rerun', action='store_true', default=False)
args = parser.parse_args()

filenames   = [
  "../data/CCS/ALL.CSV",
  "../data/RDR/ALL.CSV",
]

sampledFiles = []

for filename in filenames:
  np.random.seed(1)

  sampleFrac = 0.05
  
  outFilename = replaceEnforced(filename, "ALL.CSV", "subsetShots_5pct.csv")

  sampledFiles.append(outFilename)

  print "sampling from {}".format(filename)
  if os.path.exists(outFilename) and not args.force_rerun:
    print "  Already exists; skipping"
    continue

  sampledFrames = []
  rowCount = 0
  chunkIter = pd.read_csv(filename, chunksize=10000)
  for d in chunkIter:
    sampledFrames.append(d.sample(frac=sampleFrac))
  
    rowCount += d.shape[0]
    print rowCount
  
  print("Concatenating...")
  sampled = pd.concat(sampledFrames)

  print("Saving to {} ...".format(outFilename))
  sampled.to_csv(outFilename, index=False)

for filename in sampledFiles:
  print "quantiles for {}".format(filename)

  quantileFilename = replaceEnforced(filename, ".csv", "_quantiles.csv")
  if os.path.exists(quantileFilename) and not args.force_rerun:
    print "  Already exists; skipping"
    continue

  d = pd.read_csv(filename)
  quantiles = d.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1])

  numNegative = pd.Series(
      (len(d.loc[:, c].pipe(lambda x: x[x < 0])) for c in d.columns),
      index = d.columns).rename("num_negative")

  quantiles = quantiles.append(numNegative).transpose()

  quantiles.to_csv(quantileFilename)

centerFilenames = []
for filename in sampledFiles:
  centerFilename = replaceEnforced(filename, ".csv", "_centers.csv")
  centerFilenames.append(centerFilename)

  print "threshold intensities / cluster for {}".format(filename)

  if os.path.exists(centerFilename) and not args.force_rerun:
    print "  Already exists; skipping"
    continue

  minIntensity = 0.002  # units: fraction (0 < f < 1)

  d = normalizeRows(pd.read_csv(filename))

  wavelengthCols = [x for x in d.columns if "wavelength" in x]

  allWavelengths = map(extractWavelength, wavelengthCols)

  colToWavelength = { col: extractWavelength(col) for col in wavelengthCols }

  wavelengths = []
  for i, row in d.iterrows():
    if i % 100 == 0: print i
    for col, val in row.loc[wavelengthCols].iteritems():
      if val >= minIntensity:
        wavelengths.append(colToWavelength[col])
  wavelengths = np.reshape(wavelengths, [len(wavelengths), 1])
  wavelengths = pd.DataFrame(wavelengths).sample(frac = 0.01)

  print "wavelengths shape: ", wavelengths.shape

  kmodel = KMeans(n_clusters = 50)
  kmodel.fit(wavelengths)
  centers = np.sort(kmodel.cluster_centers_, axis = 0)
  mapping = nearestColumnMapping(centers, wavelengthCols)
  pd.DataFrame.from_dict(
      {
        "from": mapping.keys(),
        "to": mapping.values(),
        "from_numeric": map(extractWavelength, mapping.keys()),
        "to_numeric": map(extractWavelength, mapping.values()),
      }).to_csv(centerFilename, index=None)

for centerFilename, sampledFilename in zip(centerFilenames, sampledFiles):
  dimReducedFilename = replaceEnforced(sampledFilename, ".csv", "_reduced.csv")

  print("reducing dimensions for {}".format(sampledFilename))

  if os.path.exists(dimReducedFilename) and not args.force_rerun:
    print "  Already exists; skipping"
    continue

  centers = pd.read_csv(centerFilename)
  centers = OrderedDict(zip(centers["from"], centers["to"]))

  d = normalizeRows(pd.read_csv(sampledFilename))
  d = d.apply(reduceDimRow, axis = 1, centers = centers)
  d.to_csv(dimReducedFilename, index=None)

for centerFilename, bigFilename in zip(centerFilenames, filenames):
  dimReducedFilename = replaceEnforced(bigFilename, ".CSV", "_reduced.csv")

  print("reducing dimensions for {}".format(bigFilename))

  if os.path.exists(dimReducedFilename) and not args.force_rerun:
    print "  Already exists; skipping"
    continue

  centers = pd.read_csv(centerFilename)
  centers = OrderedDict(zip(centers["from"], centers["to"]))

  # clear output file
  with open(dimReducedFilename, 'w') as f:
    f.write('')

  rowCount = 0
  for chunk in pd.read_csv(bigFilename, chunksize=1000):
    needsHeader = (rowCount == 0)
    print rowCount
    rowCount += chunk.shape[0]

    processed = normalizeRows(chunk) \
        .apply(reduceDimRow, axis = 1, centers = centers)

    with open(dimReducedFilename, 'a') as f:
      processed.to_csv(f, index=None, header = needsHeader)

