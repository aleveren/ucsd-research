#!/usr/bin/python

from __future__ import print_function, division

import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt

LABEL_MAGIC = 2049
IMAGE_MAGIC = 2051

def parse_labels(filename):
  print("Parsing {} as labels".format(filename))
  with gzip.GzipFile(filename) as f:
    magicNumber, numItems = struct.unpack('>II', f.read(8))
    assert magicNumber == LABEL_MAGIC
    print("  numItems = {}".format(numItems))

    result = np.array(struct.unpack('B' * numItems, f.read(numItems)))

    print("  Done")
    return result

def parse_images(filename, add_constant = True):
  print("Parsing {} as images".format(filename))
  with gzip.GzipFile(filename) as f:
    magicNumber, numImages, numRows, numCols = \
        struct.unpack('>IIII', f.read(16))
    assert magicNumber == IMAGE_MAGIC
    print("  numImages = {}".format(numImages))
    print("  numRows = {}".format(numRows))
    print("  numCols = {}".format(numCols))

    dataBytes = numImages * numRows * numCols
    data = np.array(struct.unpack('B' * dataBytes, f.read(dataBytes)))
    result = np.reshape(data, (numImages, numRows * numCols))

    if add_constant:
      # Insert column of 1's
      constant_column = np.full((numImages, 1), 1.0)
      result = np.hstack([constant_column, result])

    print("  Done")
    return result

def show_image(im, label = None, ax = None, diff = False):
  if im.size == 28 * 28:
    im = im.reshape(28, 28)
  elif im.size == 28 * 28 + 1:
    im = im[1:].reshape(28, 28)
  else:
    raise Exception("Unexpected image shape: {}".format(im.shape))
  if ax is None:
    fig, ax = plt.subplots()

  if diff:
    max_abs = np.max(np.abs(im))
    imshow_kwargs = {'cmap': 'seismic', 'vmax': max_abs, 'vmin': -max_abs}
  else:
    imshow_kwargs = {'cmap': 'gray_r'}
  ax.imshow(im, interpolation = 'nearest', **imshow_kwargs)

  if label is not None:
    ax.set_title("Label: {}".format(label))

if __name__ == "__main__":
  testLabels = parse_labels('t10k-labels-idx1-ubyte.gz')
  testImages = parse_images('t10k-images-idx3-ubyte.gz')
  trainLabels = parse_labels('train-labels-idx1-ubyte.gz')
  trainImages = parse_images('train-images-idx3-ubyte.gz')
  
  print(testLabels.shape)
  print(testImages.shape)
  print(trainLabels.shape)
  print(trainImages.shape)
  
  # Display a few images for debugging purposes
  for i in range(10):
    show_image(testImages[i], testLabels[i])
  plt.show()
