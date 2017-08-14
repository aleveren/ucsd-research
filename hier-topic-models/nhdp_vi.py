import numpy as np
import matplotlib.pyplot as plt
import string
from scipy.sparse import dok_matrix
from collections import Counter
import copy

class NHDP(object):
    def __init__(self):
        self.num_iter = 1000
        self.subsample_size = 20
        self.step_size = 0.001
        self.state = None
        self.history = None

    def inference(self, data):
        '''
        Perform stochastic variational inference using the nHDP model

        Parameters:
        * data: a matrix representing a corpus in the bag-of-words model;
            each row is a document, and each column is a vocab word.
            data[i,j] = n ==> document i contains vocab word j a total of n times.
            See the prep_data function below.
        '''

        self.history = History()
        self.init_state(data)
        self.history.append(self.state)
        num_docs, vocab_size = data.shape

        for i in range(self.num_iter):
            indices = self.subset_indices(self.subsample_size, num_docs)
            subset = data[indices, :]
            for doc in subset:
                tree = self.subtree(doc)
                self.update_local(tree, doc)
            self.update_global(subset, num_docs)
            self.history.append(self.state)

        return copy.deepcopy(self.state)

    def init_state(self, data):
        

        # Initialize the variational distributions
        self.state = dict()
        self.state["lam"] = 0
        self.state["tau1"] = 0
        self.state["tau2"] = 0
        self.state["u"] = 0
        self.state["v"] = 0
        self.state["z"] = 0
        self.state["a"] = 0
        self.state["b"] = 0
        self.state["nu"] = 0

    def subset_indices(self, k, n):
        assert k >= 0
        if k > n:
            k = n
        mask = np.array([True for i in range(k)] + [False for i in range(n-k)])
        np.random.shuffle(mask)
        return np.nonzero(mask)[0]

    def subtree(self, doc):
        pass

    def update_local(self, tree, doc):
        pass

    def update_global(self, subset, num_docs):
        subset_size = subset.shape[0]
        rho = self.step_size

        for path in tree:
            for vocab_word_index in range(len(vocab)):
                lam_prime = 0
                for row in subset:
                    for word_slot in row:
                        if w[d,n] == vocab_word_index:
                            lam_prime += self.state["nu"][doc_index, n, path]
                lam_prime *= num_docs / float(subset_size)
                self.state["lam"][path, word] = lam0 + (1-rho) * self.state["lam"][path, word] + rho * lam_prime

        for path in tree:
            tp = TODO
            tpp = TODO
            self.state["tau1"][path] = 1 + (1-rho) * self.state["tau1"][path] + rho * tp
            self.state["tau2"][path] = alpha + (1-rho) * self.state["tau2"][path] + rho * tpp

def prep_data(corpus):
    '''
    Parameters:
    * corpus: a list of strings (each string is a document)

    Returns:
    * vocab (list of words), data (sparse matrix of counts)
    '''
    # Build up vocabulary
    num_docs = 0
    docs = []
    vocab_to_indices = dict()
    for doc_string in corpus:
        num_docs += 1
        doc_counter = Counter()
        for word in doc_string.split():
            word = word.translate(None, string.punctuation).lower()
            if word not in vocab_to_indices:
                vocab_to_indices[word] = len(vocab_to_indices)
            word_index = vocab_to_indices[word]
            doc_counter[word_index] += 1
        docs.append(doc_counter)
    vocab = sorted(vocab_to_indices.keys(), key=lambda x: vocab_to_indices[x])

    # Build sparse matrix
    data = dok_matrix((num_docs, len(vocab)), dtype='int')
    for i, doc_counter in enumerate(docs):
        for j in doc_counter.keys():
            data[i,j] = doc_counter[j]
    data = data.tocsr()
    return vocab, data

class History(object):
    def __init__(self):
        self._reset()

    def _reset(self):
        self._contents = dict()
        self._history_len = 0

    def keys(self):
        return self._contents.keys()

    def __len__(self):
        return self._history_len

    def __getitem__(self, x):
        if isinstance(x, tuple):
            result = self._contents
            while len(x) > 0:
                if not isinstance(result, dict):
                    result = np.asarray(result)
                result = result[x[0]]
                x = x[1:]
            return result
        else:
            return self._contents[x]

    def append(self, state):
        if self._history_len == 0:
            for stat_name in state.keys():
                self._contents[stat_name] = []
        not_updated = set(self._contents.keys())
        for stat_name, value in state.items():
            self._contents[stat_name].append(copy.deepcopy(value))
            not_updated.remove(stat_name)
        assert len(not_updated) == 0, "Didn't update: {}".format(not_updated)
        self._history_len += 1
