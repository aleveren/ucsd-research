from collections import namedtuple

class Tree(namedtuple("Tree", ["data", "children"])):
    @classmethod
    def leaf(cls, data = None):
        return cls(data = data, children = [])
