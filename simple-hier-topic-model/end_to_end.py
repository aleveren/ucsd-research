import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csc_matrix, dok_matrix

from collections import Counter
import sys
import datetime
import pickle

from utils import nice_tree_plot, niceprint, permute_square, invert_permutation, without_diag
from compute_pam import compute_combo_tensor_pam, IndividualNodeAlphaCalc
from example_graphs import make_tree
from sim_data import PAMSampler, topics_griffiths_steyvers
from lda_collapsed_gibbs import CollapsedGibbs
from tree_extraction import Aho
from tree_extraction.Aho import get_ratio_matrix
from alpha_extract import AlphaExtract
from param_stats import topic_difference, find_flat_permutation, find_structural_permutation

sys.path.insert(0, '../anchor-word-recovery/')
from learn_topics import Analysis as AnchorAnalysis

class Analysis(object):
    def __init__(
            self,
            # Data generation params:
            true_tree,
            true_alphas,
            num_docs,
            words_per_doc,
            vocab_size,
            true_topics,
            leaf_to_index,
            # Extraction params:
            num_topics_to_train,
            delta_min,
            alpha_max,
            topic_extraction_strategy,
            topic_extraction_params):

        self.true_tree = true_tree
        self.true_alphas = true_alphas
        self.num_docs = num_docs
        self.words_per_doc = words_per_doc
        self.vocab_size = vocab_size
        self.true_topics = true_topics
        self.leaf_to_index = leaf_to_index
        self.num_topics_to_train = num_topics_to_train
        self.delta_min = delta_min
        self.alpha_max = alpha_max
        self.topic_extraction_strategy = topic_extraction_strategy
        self.topic_extraction_params = topic_extraction_params

        self.true_num_topics = true_topics.shape[0]
        self.threshold = self.delta_min / ((1 + self.alpha_max) * (1 + self.alpha_max + self.delta_min))
    
    def run(self):
        self.alpha_calc = IndividualNodeAlphaCalc(values = self.true_alphas)
        self.true_cooccur = compute_combo_tensor_pam(g = self.true_tree, alpha = self.alpha_calc)
        
        # Generate data
        self.sampler = PAMSampler(
            g = self.true_tree,
            num_docs = self.num_docs,
            words_per_doc = self.words_per_doc,
            vocab_size = self.vocab_size,
            alpha_calc = self.true_alphas,
            topic_dict = make_topic_dict(self.true_topics, self.leaf_to_index),
        )
        self.sampler.sample()
        self.corpus = make_short_corpus(self.sampler.docs)

        # Compute empirical statistics from topic-labeled data
        self.emp_cooccur = compute_empirical_cooccur(sampler = self.sampler, leaf_to_index = self.leaf_to_index)
        self.emp_topics = compute_empirical_topics(sampler = self.sampler, leaf_to_index = self.leaf_to_index)
        
        # Extract topics and co-occurrence from data
        if self.topic_extraction_strategy == "CollapsedGibbs":
            kwargs = dict()
            kwargs.update(self.topic_extraction_params)
            kwargs.update(dict(
                corpus = self.corpus,
                num_topics = self.num_topics_to_train,
                vocab_size = self.vocab_size,
            ))
            self.collapsed_gibbs_kwargs = kwargs

            self.collapsed_gibbs = CollapsedGibbs()
            self.collapsed_gibbs.fit(**kwargs)

            self.est_topics = self.collapsed_gibbs.topics_by_sample()[-1]  # TODO: revisit this calculation?
            self.est_cooccur = self.collapsed_gibbs.cooccurrence_by_sample()[-1]  # TODO: revisit this calculation?

        elif self.topic_extraction_strategy == "AnchorWords":
            sparse_docs = dok_matrix((self.vocab_size, self.num_docs))
            for doc_index, doc in enumerate(self.sampler.docs):
                counter = Counter(doc)
                for vocab_index, count in counter.items():
                    sparse_docs[vocab_index, doc_index] += count
            sparse_docs = csc_matrix(sparse_docs)

            num_digits = len(str(self.vocab_size - 1))
            vocab_strings = ["w{:0{}d}".format(i, num_digits) for i in range(self.vocab_size)]

            kwargs = dict()
            kwargs.update(dict(
                # default params
                loss = "L2",
                seed = 100,
                eps = 1e-6,
                new_dim = 1000,
                max_threads = 8,
                anchor_thresh = 100,
                top_words = 10,
            ))
            kwargs.update(self.topic_extraction_params) # user-supplied params
            kwargs.update(dict(
                # params that our code is responsible for
                infile = sparse_docs,
                vocab_file = vocab_strings,
                outfile = None,
                K = self.num_topics_to_train,
            ))

            self.anchor_analysis = AnchorAnalysis(**kwargs)
            self.anchor_analysis.run()

            self.est_topics = self.anchor_analysis.A.transpose()
            self.est_cooccur = self.anchor_analysis.R

        else:
            raise ValueError("Unrecognized strategy: '{}'".format(topic_extraction_strategy))
        
        # Extract tree & other parameters from data
        self.est_results = self.cooccur_to_tree_and_alphas(self.est_cooccur, self.threshold)
        self.emp_results = self.cooccur_to_tree_and_alphas(self.emp_cooccur, self.threshold)
        self.true_c_results = self.cooccur_to_tree_and_alphas(self.true_cooccur, self.threshold)
        
        self.est_alphas, self.est_tree = [self.est_results[x] for x in ["alphas", "tree"]]
        self.emp_alphas, self.emp_tree = [self.emp_results[x] for x in ["alphas", "tree"]]
        self.true_c_alphas, self.true_c_tree = [self.true_c_results[x] for x in ["alphas", "tree"]]
        
        return self.est_topics, self.est_tree, self.est_alphas
        
    def cooccur_to_tree_and_alphas(self, cooccur, threshold):
        result = dict()

        result["ratio_matrix"] = get_ratio_matrix(cooccur)
        result["constraints"] = Aho.get_constraints(result["ratio_matrix"], threshold = threshold)
        result["tree"] = Aho.extract(m = result["ratio_matrix"], apply_ratio = False, threshold = threshold)
        result["alpha_extract"] = AlphaExtract(g = result["tree"], R = cooccur)
        result["alphas"] = result["alpha_extract"].extract()
        
        return result

    def save(self, filename, to_prune = "DEFAULT", add_timestamp = True):
        # Prune some un-pickle-able objects
        if to_prune == "DEFAULT":
            to_prune = [
                ["collapsed_gibbs"],
                ["anchor_analysis"],
            ]
        pruned = dict()
        pruned_string = "PRUNED FOR SERIALIZATION"
        for path in to_prune:
            prev = None
            curr = self
            missing = False
            for p in path:
                prev = curr
                if not hasattr(curr, p) or getattr(curr, p) == pruned_string:
                    missing = True
                    break
                curr = getattr(curr, p)
            if not missing:
                setattr(prev, path[-1], pruned_string)
                pruned[tuple(path)] = curr

        if add_timestamp:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_at_%H-%M-%S')
            filename = filename.replace('.pkl', '_' + timestamp + '.pkl')
        print("Saving to {}".format(filename))

        # Do the actual saving
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

        # Reattach the un-pickle-able objects
        for path, value in pruned.items():
            curr = self
            for p in path[:-1]:
                curr = getattr(curr, p)
            setattr(curr, path[-1], value)


def make_short_corpus(docs):
    corpus_short = []
    for doc in docs:
        ctr = Counter(doc)
        doc_short = [(k, v) for k, v in ctr.items()]
        corpus_short.append(doc_short)
    return corpus_short

def compute_empirical_cooccur(sampler, leaf_to_index):
    num_topics = len(sampler.topics)
    p = np.zeros((num_topics, num_topics), dtype='float')
    for node_selections in sampler.doc_nodes:
        theta = np.zeros(num_topics, dtype='float')
        counter = Counter(node_selections)
        for node, count in counter.items():
            theta[leaf_to_index[node]] += count
        theta /= theta.sum()
        p += np.outer(theta, theta)
    p /= p.sum()
    return p

def compute_empirical_topics(sampler, leaf_to_index):
    K = len(sampler.topics)
    V = sampler.vocab_size
    counter = Counter()
    for doc_index in range(sampler.num_docs):
        doc = sampler.docs[doc_index]
        node_selections = sampler.doc_nodes[doc_index]
        for vocab_word_index, node in zip(doc, node_selections):
            counter[leaf_to_index[node], vocab_word_index] += 1

    result = np.zeros((K, V), dtype='float')
    for coords, count in counter.items():
        result[coords] += count
    result /= result.sum(axis = 1, keepdims = True)

    return result

def make_topic_dict(topics, leaf_to_index):
    if isinstance(topics, dict):
        return topics
    result = dict()
    for leaf, index in leaf_to_index.items():
        result[leaf] = topics[index, :]
    return result
