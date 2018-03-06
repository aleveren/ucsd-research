from __future__ import print_function

from sim_data import SimData

#import os
#import sys
#
#sys.path.append(os.path.abspath(".."))
#import text_utils

sim = SimData(
    branching_factors = [3],
    num_docs = 1000,
    doc_length = 200,
    topic_sharpness = 20,
    alpha_leaves = 0.1,
    alpha_depths = 1,
    heavy_words_per_topic = 2,
    vocab_size = 8,
)

print("Generating simulated data...")
docs = sim.generate()

filename = "simulated_data.txt"

print("Exporting to {}...".format(filename))
with open(filename, 'w') as f:
    for d in docs:
        print(d, file=f)
print("Done.")
