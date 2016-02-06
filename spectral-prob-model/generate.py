#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

class PeakDistrib(namedtuple("PeakDistrib", [
    "mean_location", "sigma_location",
    "mean_height", "sigma_height",
    "mean_width", "sigma_width"])):
  pass

class GaussianParams(namedtuple("GaussianParams", [
    "mean", "sigma", "height"])):
  def density(self, x):
    var = self.sigma ** 2
    unscaled = np.exp(-0.5 * (x - self.mean) ** 2 / float(var))
    return self.height * unscaled

mwidth = np.log(0.1)
sigloc = 1e-1
sigheight = 1e-6
sigwidth = 1e-6

compoundPeaks = [
  [
    PeakDistrib(np.log(1), sigloc, np.log(10), sigheight, mwidth, sigwidth),
    PeakDistrib(np.log(2), sigloc, np.log(15), sigheight, mwidth, sigwidth),
  ],
  [
    PeakDistrib(np.log(3), sigloc, np.log(20), sigheight, mwidth, sigwidth),
    PeakDistrib(np.log(4), sigloc, np.log(25), sigheight, mwidth, sigwidth),
  ],
  [
    PeakDistrib(np.log(5), sigloc, np.log(30), sigheight, mwidth, sigwidth),
    PeakDistrib(np.log(6), sigloc, np.log(35), sigheight, mwidth, sigwidth),
  ],
]

np.random.seed(1)
numSamples = 10
xsample = np.arange(0, 10, 0.01)

plt.figure()

for sampleIndex in range(numSamples):
  gaussians = []

  for ps in compoundPeaks:
    #a_mean = 1.0  # For now, use the same prior distrib for each abundance
    #a_sigma = 0.1
    #abundance = np.random.lognormal(a_mean, a_sigma)
    abundance = 1.0

    for peak in ps:
      loc = np.random.lognormal(peak.mean_location, peak.sigma_location)
      width = np.random.lognormal(peak.mean_width, peak.sigma_width)
      height = np.random.lognormal(peak.mean_height, peak.sigma_height)
      gaussians.append(GaussianParams(loc, width, abundance * height))

  print("SAMPLE")
  for g in gaussians:
    print(g)

  sample = np.zeros(len(xsample))
  for g in gaussians:
    sample += [g.density(x) for x in xsample]

  ax = plt.gca()
  ax.plot(xsample, sample)

plt.show()
