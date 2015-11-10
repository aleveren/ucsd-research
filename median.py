#!/usr/bin/python

import numpy as np

def main():
  np.random.seed(1)
  alpha = 1/4.0
  a = np.arange(0, 100, 10)
  np.random.shuffle(a)
  
  for x in a: print(x)
  for i, x in enumerate(sorted(a)): print("{}, {}".format(i, x))
  for i in range(-1, len(a) + 1):
    print("Select {} = {}".format(i, selectRank(a, i)))

def selectQuantile(values, alpha):
  rank = round(len(values) * alpha)
  return selectRank(values, rank)

def selectRank(values, rank):
  if rank <= 0:
    return min(values)
  elif rank >= len(values) - 1:
    return max(values)
  pivot = np.random.choice(values, 1)[0]
  N = len(values)
  lower = values[values < pivot]
  higher = values[values > pivot]
  if rank < len(lower):
    return selectRank(lower, rank)
  elif rank >= N - len(higher):
    numLowerOrEqual = N - len(higher)
    return selectRank(higher, rank - numLowerOrEqual)
  else:
    return pivot

if __name__ == "__main__":
  main()
