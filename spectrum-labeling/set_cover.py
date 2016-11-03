from itertools import chain, combinations

def set_cover_exact_slow(sets):
    '''
    Exhaustive search algorithm to find the smallest set of subsets
    that together cover all available items.
    '''
    universe = set()
    for s in sets:
        universe |= set(s)

    set_indices = xrange(len(sets))
    combos = (combinations(set_indices, r) for r in xrange(len(sets) + 1))
    powerset = chain.from_iterable(combos)
    best = None
    for indices in powerset:
        union = set()
        for i in indices:
            union |= set(sets[i])
        if len(union) == len(universe):
            if best is None or len(indices) < len(best):
                best = list(indices)
    return best

def set_cover_approx_fast(sets, weights = None):
    '''
    Greedy algorithm to find an approximation to the smallest set of
    subsets that together cover all available items.
    
    For details, see:
    http://pages.cs.wisc.edu/~shuchi/courses/880-S07/scribe-notes/lecture03.pdf
    '''
    universe = set()
    for s in sets:
        universe |= set(s)
    assert len(universe) > 0

    if weights is None:
        weights = [1 for s in sets]
    assert len(weights) == len(sets)
    
    remaining_sets = {i: s for i, s in enumerate(sets)}
    uncovered = set() | universe
    indices = []
    union = set()
    while len(uncovered) > 0:
        to_add = None
        best_score = None
        for set_index, s in remaining_sets.items():
            intersect = set(s) & uncovered
            score = len(intersect) / float(weights[set_index])
            if to_add is None or score > best_score:
                best_score = score
                to_add = set_index
        indices.append(to_add)
        union |= set(remaining_sets[to_add])
        uncovered -= set(remaining_sets[to_add])
        del remaining_sets[to_add]

    return indices
