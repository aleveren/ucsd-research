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

def normalizeRows(d, useWidths):
  unchanged = d.iloc[:, :3]
  toChange = d.iloc[:, 3:]
  if useWidths:
    # multiply by "width" of wavelength ranges
    wavelengths = map(extractWavelength, toChange.columns)
    deltas = getDeltas(wavelengths).as_matrix()
    toChange = toChange.mul(deltas, axis=1)
  # set negative entries to 0
  no_neg = toChange.where(toChange >= 0, other=0)
  # normalize rows to sum to 1
  sums = no_neg.sum(axis=1)
  normalized = no_neg.div(sums, axis=0)
  return pd.concat([unchanged, normalized], axis=1)

def testNormalize():
  global dfc, dfr, dfc0, dfc1, dfr0, dfr1  # TODO: remove
  dfr = pd.read_csv("../data/RDR/subsetShots_5pct.csv", nrows=5)
  dfc = pd.read_csv("../data/CCS/subsetShots_5pct.csv", nrows=5)
  dfr0 = normalizeRows(dfr, useWidths=False)
  dfr1 = normalizeRows(dfr, useWidths=True)
  dfc0 = normalizeRows(dfc, useWidths=False)
  dfc1 = normalizeRows(dfc, useWidths=True)

  print dfr0.iloc[0,:3]
  print dfr1.iloc[0,:3]
  print dfc0.iloc[0,:3]
  print dfc1.iloc[0,:3]

  def plotToAxes(ax):
    ax.plot(dfr0.iloc[0,3:].tolist(),  'g-', label='RDR data')
    ax.plot(dfr1.iloc[0,3:].tolist(), 'g--', label='RDR data, width-adjusted')
    ax.plot(dfc0.iloc[0,3:].tolist(),  'b-', label='CCS data')
    ax.plot(dfc1.iloc[0,3:].tolist(), 'b--', label='CCS data, width-adjusted')

  fig = plt.figure()
  ax = fig.add_subplot(1, 2, 1)
  plotToAxes(ax)
  ax.legend(loc='upper left')
  ax.set_title("example spectra")

  otherAxes = fig.add_subplot(1, 2, 2)
  plotToAxes(otherAxes)
  otherAxes.set_xlim(100,200)
  otherAxes.set_ylim(0,0.005)
  otherAxes.set_title("zoomed view")

  mappingData = pd.read_csv("../data/CCS/subsetShots_5pct_centers.csv")
  origWavelength = mappingData["from_numeric"].tolist()
  newWavelength = mappingData["to_numeric"].tolist()

  mapper = OrderedDict(zip(mappingData["from"], mappingData["to"]))

  spectrum = dfc0.iloc[0, 3:]
  # insert NaN's to avoid connecting "gaps" (hacky)
  spectrum[2048] = np.nan
  spectrum[4096] = np.nan

  spectrum_after = reduceDimRow(dfc0.iloc[0,:], mapper).iloc[3:]
  wavelength_after = map(extractWavelength, spectrum_after.index)

  fig = plt.figure(figsize=(7,12))

  ax = fig.add_subplot(3,1,1)
  ax.plot(origWavelength, newWavelength, 'bx')
  ax.set_xlabel("wavelength")
  ax.set_ylabel("nearest cluster center")
  ax.set_title("Wavelength clustering results, based on CCS data")
  xlim = ax.get_xlim()

  ax = fig.add_subplot(3,1,2)
  ax.plot(origWavelength, spectrum, 'b.-')
  ax.set_xlabel("wavelength")
  ax.set_ylabel("normalized intensity")
  ax.set_xlim(xlim)

  ax = fig.add_subplot(3,1,3)
  ax.stem(wavelength_after, spectrum_after, 'b.-')
  ax.set_xlabel("wavelength")
  ax.set_ylabel("normalized intensity after dimensionality reduction")
  ax.set_xlim(xlim)

  plt.show()

def getDeltas(w):
  # Compute width of the range of wavelengths covered by each data point
  deltas = pd.Series(w).diff()
  # Adjust for large "holes" in the spectrum
  assert np.isnan(deltas[0])
  deltas[0] = deltas[1]
  largeDeltaIndices = deltas[deltas > 1].index
  deltas[largeDeltaIndices] = deltas[largeDeltaIndices - 1]
  return deltas

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

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--force-rerun', action='store_true', default=False)
  args = parser.parse_args()

  filenames   = [
    "../data/CCS/ALL.CSV",
    "../data/RDR/ALL.CSV",
  ]

  sampledFiles = []

  # generate random samples
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

  # compute quantiles on samples
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

  # perform "wavelength clustering" to identify cluster centers
  centerFilenames = []
  for filename in sampledFiles:
    np.random.seed(1)

    centerFilename = replaceEnforced(filename, ".csv", "_centers.csv")
    centerFilenames.append(centerFilename)

    print "threshold intensities / cluster for {}".format(filename)

    if os.path.exists(centerFilename) and not args.force_rerun:
      print "  Already exists; skipping"
      continue

    minIntensity = 0.002  # units: fraction (0 < f < 1)

    d = normalizeRows(pd.read_csv(filename), useWidths=False)

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

  # reduce dimensions of small samples
  for centerFilename, sampledFilename in zip(centerFilenames, sampledFiles):
    dimReducedFilename = replaceEnforced(
        sampledFilename, ".csv", "_reduced.csv")

    print("reducing dimensions for {}".format(sampledFilename))

    if os.path.exists(dimReducedFilename) and not args.force_rerun:
      print "  Already exists; skipping"
      continue

    centers = pd.read_csv(centerFilename)
    centers = OrderedDict(zip(centers["from"], centers["to"]))

    d = normalizeRows(pd.read_csv(sampledFilename), useWidths=False)
    d = d.apply(reduceDimRow, axis = 1, centers = centers)
    d.to_csv(dimReducedFilename, index=None)

  # reduce dimensions of full datasets (using batches)
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

      processed = normalizeRows(chunk, useWidths=False) \
          .apply(reduceDimRow, axis = 1, centers = centers)

      with open(dimReducedFilename, 'a') as f:
        processed.to_csv(f, index=None, header = needsHeader)

if __name__ == "__main__":
  main()
