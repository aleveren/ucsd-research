#!/usr/bin/env python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import OrderedDict, defaultdict

from earthmover import earthmover1d

def uniform_randomEndpoints(xs, xmin = None, xmax = None, numIntervals = 1):
  if xmin is None: xmin = min(xs)
  if xmax is None: xmax = max(xs)
  endpoints = [np.random.uniform(xmin, xmax)
      for i in range(2*numIntervals)]
  #endpoints = sorted([np.random.uniform(xmin, xmax)
  #    for i in range(2*numIntervals)])
  result = np.zeros_like(xs)
  for i in range(numIntervals):
    a = min([endpoints[2*i], endpoints[2*i + 1]])
    b = max([endpoints[2*i], endpoints[2*i + 1]])
    result += uniform(xs, a, b)
  result /= np.sum(result)
  return result

def uniform(xs, a, b):
  ys = np.zeros_like(xs)
  ys[(xs > a) & (xs < b)] = (max(xs)-min(xs)) / float((b - a) * len(xs))
  return ys

def demo():
  np.random.seed(1)

  n = 1000.0
  xmin = -2.0
  xmax = 3.0
  xs = np.linspace(xmin, xmax, n)
  #ys = np.cos(4 * np.pi * xs) ** 2
  #ys = np.random.lognormal(0, 1, len(xs)) * (xs > 0) * (xs < 1); ys /= np.sum(ys)
  ys = uniform(xs, 0.3, 0.4)
  #ys = uniform_randomEndpoints(xs)

  centers = np.array([-1,0,1,2])

  m1 = nearestMapping(xs, centers)
  xs1, ys1 = discretize(xs, ys, m1)
  showDiscretization(xs, ys, xs1, ys1, m1)

  m2 = proportionalMapping(xs, centers)
  xs2, ys2 = discretize(xs, ys, m2)
  showDiscretization(xs, ys, xs2, ys2, m2)

  for i in range(8):
    testName = "nearest" if i % 2 == 0 else "proportional"
    testMapping = m1 if i % 2 == 0 else m2

    p1 = uniform_randomEndpoints(xs, 0, 1)
    p2 = uniform_randomEndpoints(xs, 0, 1)
    if i >= 4:
      p1 *= np.random.lognormal(0, 1, len(xs))
      p1 /= np.sum(p1)
      if i >= 6:
        p2 *= np.random.lognormal(0, 1, len(xs))
        p2 /= np.sum(p2)
    distance_before = earthmover1d(xs, p1, p2)
    xs_new, p1_new = discretize(xs, p1, testMapping)
    xs_new2, p2_new = discretize(xs, p2, testMapping)
    assert np.array_equal(xs_new, xs_new2)
    distance_after = earthmover1d(xs_new, p1_new, p2_new)

    print distance_before
    print distance_after
    title = "{}: original distance = {:.2f}, discretized distance = {:.2f}".format(testName, distance_before, distance_after)

    epsilon = 1e-6
    if distance_after - distance_before > epsilon:
      title += "; counterexample"

    fig, axes = plt.subplots(3, 2, sharex='col')
    a1 = [axes[i][0] for i in range(3)]
    a2 = [axes[i][1] for i in range(3)]
    showDiscretization(xs, p1, xs_new, p1_new, testMapping, axes = a1)
    showDiscretization(xs, p2, xs_new, p2_new, testMapping, axes = a2)

    fig.suptitle(title)

    d11_before = earthmover1d(xs, p1, p1)
    d11_after = earthmover1d(xs_new, p1_new, p1_new)
    assert d11_before == 0
    assert d11_after == 0

  plt.show()

def nearestMapping(xs, centersList):
  result = OrderedDict()
  for x in xs:
    index, ctr = min(enumerate(centersList), key=lambda c: np.abs(c[1] - x))
    result[x] = [(1.0, ctr)]
  return result

def proportionalMapping(xs, centersList):
  result = OrderedDict()
  for x in xs:
    index, ctr = min(enumerate(centersList), key=lambda c: np.abs(c[1] - x))
    if (index == 0 and x <= ctr) or \
        (index == len(centersList) - 1 and x >= ctr):
      result[x] = [(1.0, ctr)]
    else:
      lowIndex = index if x >= ctr else index - 1
      hiIndex = index if x <= ctr else index + 1
      lowCenter = centersList[lowIndex]
      hiCenter = centersList[hiIndex]
      fracLow = (hiCenter - x) / float(hiCenter - lowCenter)
      fracHi = (x - lowCenter) / float(hiCenter - lowCenter)
      result[x] = [(fracLow, lowCenter), (fracHi, hiCenter)]
  return result

def discretize(xs, ys, mapping):
  assert len(xs) == len(ys), "Length mismatch"
  new_pairs = defaultdict(float)
  for x, y in zip(xs, ys):
    destinations = mapping[x]
    for proportion, x_new in destinations:
      new_pairs[x_new] += y * proportion
  xs_new = sorted(new_pairs.keys())
  ys_new = [new_pairs[x] for x in xs_new]
  return np.array(xs_new), np.array(ys_new)

def showDiscretization(xs, ys, xs_new, ys_new, mapping, axes = None):
  if axes is None:
    fig, axes = plt.subplots(3, 1, sharex=True)
  for i in range(3):
    axes[i].margins(0.1)

  axes[0].plot(xs, ys, 'b-')
  axes[1].stem(xs_new, ys_new, 'b')
  for c_index, x_new in enumerate(xs_new):
    proportion_mapped = np.zeros_like(xs)
    for i, x in enumerate(xs):
      destinations = mapping[x]
      for proportion, xprime in destinations:
        if xprime == x_new:
          proportion_mapped[i] += proportion
    axes[2].plot(xs, c_index + 0.6 * proportion_mapped)

if __name__ == "__main__":
  demo()

