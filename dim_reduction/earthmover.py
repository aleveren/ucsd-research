#!/usr/bin/env python

import numpy as np

def earthmover1d(xs, p1, p2):
  widths = np.ediff1d(xs, to_end=[0.0])
  return np.sum(widths * np.abs(np.cumsum(p1) - np.cumsum(p2)))

def assertClose(a, b, epsilon=1e-8):
  assert abs(a - b) < epsilon

def test():
  xs = np.array([1,2,3])
  ys1 = np.array([1,2,3]) / 6.0

  assertClose(earthmover1d(xs, ys1, np.array([1,1,4])/6.), 1/6.)
  assertClose(earthmover1d(xs, ys1, np.array([2,2,2])/6.), 1/3.)
  assertClose(earthmover1d(xs, ys1, np.array([0.5,2,3.5])/6.), 1/6.)

  assertClose(earthmover1d(np.array([1,2,4]), ys1, np.array([1,1,4])/6.), 1/3.)
  assertClose(earthmover1d(np.array([1,2,2.5]), ys1, np.array([1,1,4])/6.),
      1/12.)
  assertClose(earthmover1d(np.array([0,2,3]), ys1, np.array([2,1,3])/6.), 1/3.)
  assertClose(earthmover1d(np.array([1.5,2,3]), ys1, np.array([2,1,3])/6.),
      1/12.)

if __name__ == "__main__":
  test()
