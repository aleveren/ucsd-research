import numpy as np
from collections import defaultdict
import copy

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

class CollapsedGibbs(object):
    def fit(self,
            corpus,
            num_topics,
            vocab_size,
            burn_in = 10,
            lag = 20,
            num_samples = 100,
            alpha = 1.0,
            beta = 1.0,
            update_alpha_every = 0):
        alpha = np.broadcast_to(alpha, (num_topics,))
        beta = np.broadcast_to(beta, (vocab_size,))
        beta_sum = beta.sum()
        lag = max(1, lag)
        burn_in = max(0, burn_in)
        num_samples = max(0, num_samples)
        update_alpha_every = max(0, update_alpha_every)

        self.alpha_original = alpha.copy()

        doc_lengths = [sum(x[1] for x in doc) for doc in corpus]
        blank_counter = lambda: np.zeros((num_topics,), dtype='int')
        topic_counts = {
            "overall": blank_counter(),
            "by_doc": defaultdict(blank_counter),
            "by_vocab": defaultdict(blank_counter),
            "by_doc_vocab": defaultdict(blank_counter),
        }

        def sample_once(doc_index, vocab_index, topic_old):
            A = topic_counts["by_doc"][doc_index]
            B = topic_counts["by_vocab"][vocab_index]
            C = topic_counts["overall"]
            D = topic_counts["by_doc_vocab"][doc_index, vocab_index]
            if topic_old is not None:
                for x in [A, B, C, D]:
                    x[topic_old] -= 1
            probs = (alpha + A) * (beta[vocab_index] + B) / (beta_sum + C)
            probs /= probs.sum()
            topic_new = np.random.choice(num_topics, p = probs)
            for x in [A, B, C, D]:
                x[topic_new] += 1                
        
        for di, doc in enumerate(tqdm(corpus, desc = "Initializing")):
            for v, c in doc:
                for pi in range(c):
                    sample_once(doc_index = di, vocab_index = v, topic_old = None)
        del di, doc, v, c, pi
        
        num_iters = burn_in + lag * num_samples
        doc_increment = 1.0 / len(corpus)
        if tqdm is not None:
            pbar_train = tqdm(
                total = float(num_iters),
                desc = "Training",
                bar_format='{l_bar}{bar}| {n:.5g}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            )
        else:
            pbar_train = None
        self.samples = []
        for i in range(num_iters):
            for di, doc in enumerate(corpus):
                for v, vc in doc:
                    current_counts = topic_counts["by_doc_vocab"][di, v].copy()
                    for t, tc in enumerate(current_counts):
                        for j in range(tc):
                            sample_once(doc_index = di, vocab_index = v, topic_old = t)
                if pbar_train is not None:
                    pbar_train.update(n = doc_increment)
            update_alpha = (update_alpha_every > 0) and (i % update_alpha_every == 0)
            if update_alpha:
                # Compute moments of observed probabilities, and estimate new alpha
                # Source: Thomas P. Minka, "Estimating a Dirichlet distribution" (2000)
                num_docs = len(corpus)
                m1 = np.zeros(num_topics)
                m2 = np.zeros(num_topics)
                for di, t in topic_counts["by_doc"].items():
                    p = t.astype('float') / t.sum()
                    m1 += p
                    m2 += p ** 2
                m1 /= num_docs
                m2 /= num_docs
                if m2[0] != m1[0] ** 2:
                    s = (m1[0] - m2[0]) / (m2[0] - m1[0] ** 2)  # Minka, eqn (21)
                    alpha = m1 * s

            if i >= burn_in and (i - burn_in) % lag == 0:
                sample = copy.deepcopy(topic_counts)
                sample["alpha"] = alpha.copy()
                self.samples.append(sample)

        if pbar_train is not None:
            pbar_train.close()

        self.corpus = copy.deepcopy(corpus)
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.alpha = alpha.copy()
        self.beta = beta.copy()

    def topics_by_sample(self):
        counts = np.zeros((len(self.samples), self.num_topics, self.vocab_size))
        for si, sample in enumerate(self.samples):
            for (di, v), topic_count in sample["by_doc_vocab"].items():
                counts[si, :, v] += topic_count
        numer = counts + self.beta[np.newaxis, np.newaxis, :]
        denom = counts.sum(axis = 2, keepdims = True) + self.beta.sum()
        return numer / denom
        #return counts / counts.sum(axis = 2, keepdims = True)
        
    def cooccurrence_by_sample(self):
        num_samples = len(self.samples)
        result = np.zeros((num_samples, self.num_topics, self.num_topics))
        for si, sample in enumerate(self.samples):
            for di, topic_count in sample["by_doc"].items():
                probs = topic_count / topic_count.sum()
                result[si, :, :] += np.outer(probs, probs)
        result /= result.sum(axis = (1, 2), keepdims = True)
        return result
