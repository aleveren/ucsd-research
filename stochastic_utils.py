import numpy as np

class CRP(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def simulate(self, n_rounds):
        '''Simulate a finite number of rounds of CRP using the
        given concentration parameter.'''

        seating = []
        for i in range(n_rounds):
            next_seat = self.simulate_round(seating)
            seating.append(next_seat)

        return np.array(seating)

    def simulate_round(self, seating):
        if not seating:
            return 0  # First customer sits at table 0
        n_occupied = max(seating) + 1
        distrib = np.zeros(n_occupied + 1)
        for i in range(len(seating)):
            distrib[seating[i]] += 1.0
        distrib[n_occupied] = self.alpha
        distrib /= float(self.alpha + len(seating))
        next_seat = np.random.choice(np.arange(n_occupied + 1, dtype='int'), p = distrib)
        return next_seat

class NCRP(object):
    def __init__(self, alpha):
        self.alpha = alpha
        self.crp = CRP(alpha)

    def simulate_round(self, seating, truncate_level):
        path = []
        for level_index in range(truncate_level):
            seating_at_level = [t[level_index] for t in seating if t[:len(path)] == tuple(path)]
            next_seat = self.crp.simulate_round(seating_at_level)
            path.append(next_seat)
        return tuple(path)

    def simulate(self, n_rounds, truncate_level):
        seating = []
        for i in range(n_rounds):
            next_seat = self.simulate_round(seating, truncate_level)
            seating.append(next_seat)
        return seating
