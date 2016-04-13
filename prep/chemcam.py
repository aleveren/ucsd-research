#!/usr/bin/env python

from __future__ import print_function

import os, os.path
import urllib2
import numpy as np
import pandas as pd
import re
import argparse
import sys

def download(url, dest, verbose=True):
  if verbose:
    if dest is None:
      print("Downloading '{}' ...".format(url))
    else:
      print("Downloading '{}' --> '{}' ...".format(url, dest))

  try:
    response = urllib2.urlopen(url)
  except (urllib2.HTTPError, urllib2.URLError) as e:
    print("  Encountered error: {}".format(e))
    return None
  else:
    body = response.read()
    if dest is not None:
      with open(dest, "w") as f:
        f.write(body)
    print("  Done.")
    return body

parser = argparse.ArgumentParser(description = "Download ChemCam data.")
parser.add_argument("--force_download", action="store_true", default=False)
parser.add_argument("--small_test", action="store_true", default=False)
parser.add_argument("--dest", action="store", default="../data")
args = parser.parse_args()

dest = args.dest
forceDownload = args.force_download
smallTest = args.small_test

# Set up data directories
dirsToCreate = [
  dest,
  dest + "/MOC",
  dest + "/RDR",
  dest + "/CCS",
]
for d in dirsToCreate:
  if not os.path.exists(d):
    print("Creating directory '{}'".format(d))
    os.makedirs(d)

baseUrl = "http://pds-geosciences.wustl.edu/" + \
    "msl/msl-m-chemcam-libs-4_5-rdr-v1/mslccm_1xxx"

# Download summary data
filename = "msl_ccam_obs.csv"
localFile = dest + "/" + filename
if forceDownload or not os.path.exists(localFile):
  url = baseUrl + "/document/" + filename
  download(url, localFile)
summaryData = pd.read_csv(localFile, dtype=object)
print("Summary data before filtering: {}".format(summaryData.shape))

# Filter summary data
indices = (summaryData["EDR Type"] == "CL5") & (summaryData["PDS?"] != "No")
summaryData = summaryData[indices]
print("Summary data after filtering: {}".format(summaryData.shape))

# Sanity check: timestamps should be unique
timestamps = []
for rowIndex, row in summaryData.iterrows():
  match = re.search(r'CL5_(\d+)EDR', row["EDR Filename"])
  timestamps.append(match.group(1))
assert len(timestamps) == len(set(timestamps))

# Download MOC files
print("Downloading MOC directory info")
response = download(baseUrl + "/data/moc/", dest = None)
pattern = r'moc_\d+_\d+\.csv'
mocFiles = sorted(list(set(re.findall(pattern, response))))

for f in mocFiles:
  localFile = dest + "/MOC/" + f
  if forceDownload or not os.path.exists(localFile):
    toDownload = baseUrl + "/data/moc/" + f
    download(toDownload, localFile)
  # Note: use header = 7 when loading this data

# Download both RDR and CCS data files
wavelengths = None
for detailType in ["RDR", "CCS"]:
  progressFile = dest + "/" + detailType + "/progress.txt"

  if forceDownload or not os.path.exists(progressFile):
    with open(progressFile, "w") as f:
      f.write("")

  with open(progressFile, "r") as f:
    alreadyAppended = [line.strip() for line in f.readlines()]
  print("alreadyAppended = {}".format(alreadyAppended))

  accumFile = dest + "/" + detailType + "/ALL.CSV"

  if len(alreadyAppended) == 0:
    with open(accumFile, "w") as f:
      f.write("")
    accumNeedsHeader = True
  else:
    accumNeedsHeader = False

  for rowWithIndex in summaryData.iterrows():
    rowIndex = rowWithIndex[0]
    row = rowWithIndex[1]

    if smallTest and rowIndex > 100:
      continue

    spacecraftClock = \
        int(re.search(r'CL5_(\d+)EDR', row["EDR Filename"]).group(1))
    detailFilename = re.sub(r"M\d\.DAT", "P3.CSV",
        row["EDR Filename"].replace("EDR", detailType))
    paddedSol = row["Sol"].rjust(5, "0")
    toDownload = baseUrl + "/data/sol" + paddedSol + "/" + detailFilename
    localFile = dest + "/" + detailType + "/" + detailFilename

    if detailFilename in alreadyAppended:
      print("Skipping already-processed file: {}".format(detailFilename))
      continue

    if forceDownload or not os.path.exists(localFile):
      result = download(toDownload, localFile)
      if result is None:
        print("Failure downloading {}".format(toDownload), file = sys.stderr)
        continue

    print("Processing '{}'".format(detailFilename))

    headerIndex = 0
    with open(localFile, "r") as f:
      for lineIndex, line in enumerate(f.readlines()):
        if line.startswith("#"):
          headerIndex = lineIndex
    if headerIndex != 16:
      print("Unexpected header index at {}: {}".format(
          detailFilename, headerIndex), file = sys.stderr)

    data = pd.read_csv(localFile, header = headerIndex)
    cols = [x.strip() for x in data.columns]
    shotColIndices = []
    shots = []
    for colIndex, colName in enumerate(cols):
      match = re.search(r'shot(\d+)', colName)
      if match:
        shots.append(int(match.group(1)))
        shotColIndices.append(colIndex)

    assert 'wave' in data.columns[0], \
        "First column '{}' does not contain 'wave'".format(data.columns[0])
    currentWavelengths = data[data.columns[0]]
    if wavelengths is None:
      wavelengths = currentWavelengths
    assert np.array_equal(wavelengths, currentWavelengths)

    toDrop = [data.columns[i] for i in range(len(data.columns))
        if i not in shotColIndices]
    data.drop(toDrop, axis = 1, inplace = True)
    data = data.T
    data.insert(0, "spacecraftClock", spacecraftClock)
    data.insert(1, "sol", int(row["Sol"]))
    data.insert(2, "shot", shots)
    newColNames = {}
    for i, w in enumerate(wavelengths):
      newColNames[i] = "wavelength_{}".format(w)
    data.rename(columns = newColNames, inplace = True)

    exportFile = dest + "/" + detailType + "/" + \
        detailFilename.replace(".CSV", ".transpose.CSV")
    data.to_csv(exportFile, index = False)

    with open(accumFile, "a") as f:
      data.to_csv(f, header = accumNeedsHeader, index = False)
    if accumNeedsHeader:
      accumNeedsHeader = False

    os.remove(localFile)
    os.remove(exportFile)

    with open(progressFile, "a") as f:
      f.write(detailFilename + "\n")
    alreadyAppended.append(detailFilename)
