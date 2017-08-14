'''
Generative model for Pachinko Allocation Modeling (PAM)
'''

from __future__ import division, print_function

import numpy as np
from collections import Counter

class TreeLike(object):
    def pretty_string(self, indent=0):
        indent_str = " " * indent
        result = self.__class__.__name__
        result += "('" + str(self.id) + "'\n"
        for i, c in enumerate(self.children):
            result += (" " * (indent + 2))
            result += str(self.alpha[i]) + " => "
            if isinstance(c, TreeLike):
                result += c.pretty_string(indent = indent+2)
            else:
                result += "'" + c + "'"
            result += "\n"
        result += indent_str + ")"
        return result

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.id)

class TopicInstance(TreeLike):
    def __init__(self, id, alpha, children):
        assert len(alpha) == len(children)
        self.id = id
        self.alpha = alpha
        self.children = children

    def sample_child(self):
        return np.random.choice(self.children, p=self.alpha)

    def sample_path(self, prefix=None):
        if prefix is None:
            prefix = []
        c = self.sample_child()
        if not isinstance(c, TreeLike):
            return prefix + [c]
        else:
            return c.sample_path(prefix = prefix + [c])

class Topic(TreeLike):
    def __init__(self, id, alpha, children):
        assert len(alpha) == len(children)
        self.id = id
        self.alpha = alpha
        self.children = children

    def sample_distribs_for_document(self):
        new_alpha = np.random.dirichlet(self.alpha)
        new_children = []
        for c in self.children:
            if isinstance(c, Topic):
                new_children.append(c.sample_distribs_for_document())
            else:
                new_children.append(c)
        return TopicInstance(id = self.id, alpha = new_alpha, children = new_children)
    
    def sample_doc(self, doc_length):
        distrib = self.sample_distribs_for_document()
        result = []
        for i in range(doc_length):
            path = distrib.sample_path()
            result.append(path[-1])
        return result
    
    def sample_corpus(self, doc_length, num_docs):
        if not np.iterable(doc_length):
            doc_length = doc_length * np.ones((num_docs,), dtype='int')
        assert len(doc_length) == num_docs
        corpus = []
        for i in range(num_docs):
            doc = self.sample_doc(doc_length = doc_length[i])
            corpus.append(doc)
        return corpus


def example():
    np.random.seed(1)

    pam = Topic(
        id = "root",
        alpha = [7, 7],
        children = [
            Topic(
                id = "L",
                alpha = [1, 4],
                children = [
                    Topic(id = "LL", alpha = [0.5, 0.5], children = ["a", "b"]),
                    Topic(id = "LR", alpha = [0.05, 0.05], children = ["c", "d"]),
                ],
            ),
            Topic(
                id = "R",
                alpha = [5, 3, 2],
                children = [
                    Topic(id = "RL", alpha = [0.2, 0.6, 0.2], children = ["e", "f", "g"]),
                    Topic(id = "RM", alpha = [4, 1, 5], children = ["h", "i", "j"]),
                    Topic(id = "RR", alpha = [10, 70, 20], children = ["k", "l", "m"]),
                ],
            )
        ])

    print(pam.pretty_string())

    distrib_tree = distrib_tree = pam.sample_distribs_for_document()
    print(distrib_tree.pretty_string())

    doc = pam.sample_doc(doc_length = 1000)
    count = Counter(doc)
    for k in sorted(count.keys()):
        print("{}: {}".format(k, count[k]))

    corpus = pam.sample_corpus(doc_length = 7, num_docs = 10)
    print(corpus)


if __name__ == "__main__":
    example()
