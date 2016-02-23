#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

class Compound(namedtuple("Compound", ["abundance", "peaks"])):
  pass

class Constant(namedtuple("Constant", ["value"])):
  def sample(self):
    return self.value

class LogNormalDistrib(namedtuple("LogNormalDistrib", ["mean", "sigma"])):
  def sample(self):
    return np.random.lognormal(self.mean, self.sigma)

class PeakDistrib(namedtuple("PeakDistrib", [
    "mean_location", "sigma_location",
    "mean_height", "sigma_height",
    "mean_width", "sigma_width"])):
  def sample(self, abundance):
    loc = np.random.lognormal(self.mean_location, self.sigma_location)
    width = np.random.lognormal(self.mean_width, self.sigma_width)
    height = np.random.lognormal(self.mean_height, self.sigma_height)
    return Peak(loc, width, abundance * height)

class Peak(namedtuple("Peak", ["mean", "sigma", "height"])):
  def density(self, x):
    var = self.sigma ** 2
    unscaled = np.exp(-0.5 * (x - self.mean) ** 2 / float(var))
    return self.height * unscaled

mwidth = np.log(0.1)
sigloc = 5e-2
sigheight = 5e-2
sigwidth = 1e-6

compoundPeaks = [
  Compound(abundance = Constant(1.0), peaks = [
    PeakDistrib(np.log(1), sigloc, np.log(10), sigheight, mwidth, sigwidth),
    PeakDistrib(np.log(2), sigloc, np.log(15), sigheight, mwidth, sigwidth),
  ]),
  Compound(abundance = Constant(1.0), peaks = [
    PeakDistrib(np.log(3), sigloc, np.log(20), sigheight, mwidth, sigwidth),
    PeakDistrib(np.log(4), sigloc, np.log(25), sigheight, mwidth, sigwidth),
  ]),
  Compound(abundance = Constant(1.0), peaks = [
    PeakDistrib(np.log(5), sigloc, np.log(30), sigheight, mwidth, sigwidth),
    PeakDistrib(np.log(6), sigloc, np.log(35), sigheight, mwidth, sigwidth),
  ]),
]

np.random.seed(1)
numSamples = 2
xsample = np.arange(0, 7, 0.01)

plt.figure()

ax = plt.gca()

for compound in compoundPeaks:
  for peak in compound.peaks:
    ax.axvline(x=np.exp(peak.mean_location), linestyle=':', color='k')

for sampleIndex in range(numSamples):
  gaussians = []

  for compound in compoundPeaks:
    abundance = compound.abundance.sample()
    for peak in compound.peaks:
      gaussians.append(peak.sample(abundance))

  print("SAMPLE")
  for g in gaussians:
    print(g)

  sample = np.zeros(len(xsample))
  for g in gaussians:
    sample += [g.density(x) for x in xsample]

  ax.plot(xsample, sample)

plt.show()
