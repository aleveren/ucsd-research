from __future__ import print_function

import numpy as np
from sim_data import SimData
from collections import Counter

#import os
#import sys
#
#sys.path.append(os.path.abspath(".."))
#import text_utils

np.random.seed(1)

sim = SimData(
    branching_factors = [3, 3],
    num_docs = 100000,
    doc_length = 2,
    topic_sharpness = 20,
    alpha_leaves = 0.01,
    alpha_depths = 1,
    heavy_words_per_topic = 2,
    #vocab_size = 8,
)

print("Generating simulated data...")
docs = sim.generate()

print("Topic matrix")
print(np.array2string(sim.topics_by_index, max_line_width=1000, precision=4, suppress_small=True))

print("Topic co-occurrence (empirical)")
num_topics = sim.topics_by_index.shape[0]
count_cooccurrence = np.zeros((num_topics, num_topics))
node_indices = np.array([sim.docs_aux[i]['node_indices_by_word_slot'] for i in range(len(sim.docs_aux))])
for doc_index in range(node_indices.shape[0]):
    i = node_indices[doc_index, 0]
    j = node_indices[doc_index, 1]
    count_cooccurrence[i, j] += 1
coocurrence = count_cooccurrence / np.sum(count_cooccurrence, axis=(0,1))
print(np.array2string(coocurrence, max_line_width=1000, precision=4, suppress_small=True))

print("Topic co-occurrence (empirical, without diagonal)")
coocurrence_alt = coocurrence - np.diag(np.diag(coocurrence))
print(np.array2string(coocurrence_alt, max_line_width=1000, precision=4, suppress_small=True))

print("Minimum spanning tree")
import networkx as nx
g = nx.Graph()
for i in range(num_topics):
    for j in range(num_topics):
        if i < j:
            # Note: use -coocurrence so that minimum_spanning_tree yields maximum spanning tree
            c = -0.5 * (coocurrence_alt[i, j] + coocurrence_alt[j, i])
            g.add_edge(i, j, weight = c)
tree = nx.algorithms.minimum_spanning_tree(g)
print(tree.edges())

output = False

if output:
    filename = "simulated_data.txt"

    print("Exporting to {}...".format(filename))
    with open(filename, 'w') as f:
        for d in docs:
            print(d, file=f)
    print("Done.")

    filename = "simulated_data_concise.txt"

    print("Exporting to {}...".format(filename))
    with open(filename, 'w') as f:
        for d in docs:
            c = Counter(d.split(' '))
            for i, v in enumerate(sim.vocab):
                if i > 0:
                    print(" ", end="", file=f)
                print("{}:{}".format(v, c[v]), end="", file=f)
            print("", file=f)
    print("Done.")
