import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class CachingDict(defaultdict):
    def __missing__(self, key):
        if not self.default_factory:
            defaultdict.__missing__(self, key)
        else:
            value = self.default_factory(key)
            dict.__setitem__(self, key, value)
            return value

def indep_rand():
    '''
    Generate an "independent" random state,
    for generating approximately independent streams of random numbers.
    '''
    seed = np.random.randint(2 ** 31 - 1)
    return np.random.RandomState(seed = seed)

def beta_collection(a, b):
    rnd = indep_rand()
    def gen():
        return rnd.beta(a, b)
    return defaultdict(gen)

def dirichlet_tree(alpha_vector):
    rnd = indep_rand()
    def gen():
        return rnd.dirichlet(alpha_vector)
    return defaultdict(gen)

class BetaStickBreakingDraw(object):
    '''Represents a lazily-evaluated instance of a single draw from a
    stick-breaking contruction based on beta distributions.
    Note: lazy evaluation is required, because a single evaluation represents
    an infinite distribution over all non-negative ingeters.'''
    def __init__(self, a, b, prepopulate = 10):
        self.a = a
        self.b = b
        self.rnd_thetas = indep_rand()
        self.rnd_draws = indep_rand()
        self.cached_vs = []
        self.cached_thetas = []
        self.sum_thetas = 0.0
        for i in range(prepopulate):
            self.extend_thetas()

    def extend_thetas(self):
        remaining = 1.0 - self.sum_thetas
        next_v = self.rnd_thetas.beta(self.a, self.b)
        next_theta = remaining * next_v
        self.cached_vs.append(next_v)
        self.cached_thetas.append(next_theta)
        self.sum_thetas += next_theta

    def draw(self):
        '''Draw a random non-negative integer based on the distribution
        defined by this instance.'''
        cumulative = 0.0
        p = self.rnd_draws.uniform(0.0, 1.0)
        i = 0
        while True:
            if i == len(self.cached_thetas):
                self.extend_thetas()
            cumulative += self.cached_thetas[i]
            if cumulative > p:
                return i
            i += 1    

class NCRPDraw(object):
    def __init__(self, alpha):
        self.alpha = alpha
        self.stick_breaking_distribs = defaultdict(lambda: BetaStickBreakingDraw(a = 1.0, b = self.alpha))

    def all_cached_thetas(self):
        return {k: v.cached_thetas for k, v in self.stick_breaking_distribs.items()}

    def all_cached_vs(self):
        return {k: v.cached_vs for k, v in self.stick_breaking_distribs.items()}

    def draw(self):
        return NCRPPathDraw(self)

class NCRPPathDraw(object):
    def __init__(self, ncrp):
        self.ncrp = ncrp
        self.path = []

    def __getitem__(self, key):
        assert isinstance(key, int)
        assert key >= 0
        while key >= len(self.path):
            distrib = self.ncrp.stick_breaking_distribs[self.path]
            next_item = distrib.draw()
            self.path.append(next_item)
        return self.path[key]

class NHDP(object):
    def __init__(self, alpha, beta, gamma1, gamma2, topic_alpha_vector, vocab):
        self.alpha = alpha
        self.beta = beta
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.topic_alpha_vector = topic_alpha_vector
        self.vocab = vocab
        assert len(topic_alpha_vector) == len(vocab)
        self.topics_by_path = dirichlet_tree(topic_alpha_vector)
        self.global_ncrp = NCRPDraw(alpha = alpha)
        self.atoms_by_path = CachingDict(lambda path:
            defaultdict(lambda: self.global_ncrp.stick_breaking_distribs[path].draw()))
        self.rnd = indep_rand()

    def draw_corpus(self, num_documents, document_length):
        if not np.iterable(document_length):
            document_length = [document_length for i in range(num_documents)]
        corpus = []
        for i in range(num_documents):
            doc, ph = self.draw_document(document_length = document_length[i])
            corpus.append((doc, ph))
        return corpus

    def draw_u_tree(self):
        return beta_collection(self.gamma1, self.gamma2)

    def draw_document(self, document_length):
        doc = []
        path_history = []
        u_tree = self.draw_u_tree()
        local_ncrp = NCRPDraw(alpha = self.beta)
        for word_slot_index in range(document_length):
            word, path = self.draw_word(u_tree, local_ncrp)
            doc.append(word)
            path_history.append(path)
        return doc, path_history

    def draw_word(self, u_tree, local_ncrp):
        path = ()
        while True:
            prob_stop = u_tree[path]
            stop = self.rnd.choice([True, False],
                p = [prob_stop, 1 - prob_stop])
            if stop:
                break
            atoms = self.atoms_by_path[path]
            atom_index = local_ncrp.stick_breaking_distribs[path].draw()
            next_path_element = atoms[atom_index]
            path = tuple(path) + (next_path_element,)
        topic = self.topics_by_path[path]
        vocab_word_index = self.rnd.choice(
            np.arange(len(self.vocab)), p = topic)
        vocab_word = self.vocab[vocab_word_index]
        return vocab_word, path
