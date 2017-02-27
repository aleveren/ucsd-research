#!/usr/bin/python

import urllib

filenames = [
  "train-images-idx3-ubyte.gz",
  "t10k-images-idx3-ubyte.gz",
  "train-labels-idx1-ubyte.gz",
  "t10k-labels-idx1-ubyte.gz"]

for filename in filenames:
  urllib.urlretrieve("http://yann.lecun.com/exdb/mnist/" + filename, filename)
