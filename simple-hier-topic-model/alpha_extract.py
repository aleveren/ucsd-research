import numpy as np
import networkx as nx

def alphas_from_R_lists(R_star_sames, R_star_nexts):
    Rs = R_star_sames
    Rn = R_star_nexts
    num_alphas = len(Rs)
    
    denom_prod_1 = 1
    denom_prod_2 = 1
    for i in range(num_alphas):
        denom_prod_1 *= Rs[i]
        denom_prod_2 *= Rn[i]
    denom = denom_prod_1 - denom_prod_2
    
    alphas = []
    for i in range(num_alphas):
        numer = 0
        for n in range(num_alphas):
            curr_term = 1
            for m in range(num_alphas):
                offset = (i + m) % num_alphas
                if m < n:
                    curr_term *= Rs[offset]
                else:
                    curr_term *= Rn[offset]
            numer += curr_term
        alphas.append(numer / denom)

    return alphas

class AlphaExtract(object):
    def __init__(self, g, R, leaf_to_index = None):
        self.g = g
        self.R = R
        self.init_leaf_to_index(leaf_to_index)

    def init_leaf_to_index(self, leaf_to_index):
        g = None
        del g
        
        if leaf_to_index is not None:
            self.leaf_to_index = leaf_to_index
        else:
            def helper(n, leaves):
                if self.g.out_degree(n) == 0:
                    leaves.append(n)
                for nbr in self.g.neighbors(n):
                    helper(nbr, leaves)
            leaves = []
            helper(self.g.graph['root'], leaves)
            self.leaf_to_index = {n: i for i, n in enumerate(leaves)}

    def extract(self):
        tree = g = R = nbr = nbrs = None
        del tree, g, R, nbr, nbrs

        data = {x: dict() for x in self.g.nodes()}
        
        def pass1(node, parent, sibling_offset):
            data[node]["parent"] = parent
            if parent is not None:
                siblings = list(self.g.neighbors(parent))
            else:
                siblings = []
            data[node]["siblings"] = siblings
            data[node]["sibling_offset"] = sibling_offset

            children = list(self.g.neighbors(node))
            for i, child in enumerate(children):
                pass1(node = child, parent = node, sibling_offset = i)

            if len(children) == 0:
                data[node]["representative"] = node
                data[node]["is_leaf"] = True
            else:
                data[node]["representative"] = data[children[0]]["representative"]
                data[node]["is_leaf"] = False

        def pass2(node):
            for child in self.g.neighbors(node):
                pass2(child)
            
            if data[node]["parent"] is not None:
                rep = data[node]["representative"]
                i = self.leaf_to_index[rep]

                siblings = data[node]["siblings"]
                sibling_offset = data[node]["sibling_offset"]
                next_sib = siblings[(sibling_offset + 1) % len(siblings)]
                next_rep = data[next_sib]["representative"]
                j = self.leaf_to_index[next_rep]

                data[node]["next_sibling"] = next_sib
                data[node]["rep_leaf_index"] = i
                data[node]["rep_leaf_index_next_sib"] = j

        def pass3(node):
            children = list(self.g.neighbors(node))
            
            for child in children:
                pass3(child)

            if len(children) == 0:
                # Base case: node is a leaf
                data[node]["f0"] = 1
                data[node]["f1"] = 1
                return

            R_star_sames = []
            R_star_nexts = []

            for child in children:
                if "f0" not in data[child]:
                    grandchildren = list(self.g.neighbors(child))
                    gc_alpha = [data[x]["alpha"] for x in grandchildren]
                    gc_f0 = [data[x]["f0"] for x in grandchildren]
                    gc_f1 = [data[x]["f1"] for x in grandchildren]
                    data[child]["f0"] = gc_f0[0] * (0 + gc_alpha[0]) / (0 + sum(gc_alpha))
                    data[child]["f1"] = gc_f1[0] * (1 + gc_alpha[0]) / (1 + sum(gc_alpha))

            for child in children:
                next_sib = data[child]["next_sibling"]
                i = data[child]["rep_leaf_index"]
                j = data[child]["rep_leaf_index_next_sib"]
                ia, ja = min(i, j), max(i, j)
                R_star_sames.append(self.R[i, i])
                R_star_nexts.append(self.R[ia, ja] * data[child]["f1"] / data[next_sib]["f0"])

            alphas = alphas_from_R_lists(R_star_sames, R_star_nexts)

            for i, child in enumerate(children):
                data[child]["alpha"] = alphas[i]
            
        root = self.g.graph["root"]
        pass1(root, parent=None, sibling_offset=None)
        pass2(root)
        pass3(root)

        self.data = data
    
        alphas = {node: data[node]["alpha"] for node in self.g.nodes() if data[node]["parent"] is not None}
        # return data
        return alphas
