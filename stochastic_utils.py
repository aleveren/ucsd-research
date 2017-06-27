import numpy as np

class CRP(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def simulate(self, n_rounds):
        '''Simulate a finite number of rounds of CRP using the
        given concentration parameter.'''

        seating = [0]  # First customer sits at table 0
        for i in range(n_rounds - 1):
            next_seat = self.simulate_round(seating)
            seating.append(next_seat)

        return np.array(seating)

    def simulate_round(self, seating):
        n_occupied = max(seating) + 1
        distrib = np.zeros(n_occupied + 1)
        for i in range(len(seating)):
            distrib[seating[i]] += 1.0
        distrib[n_occupied] = self.alpha
        distrib /= float(self.alpha + len(seating))
        next_seat = np.random.choice(np.arange(n_occupied + 1, dtype='int'), p = distrib)
        return next_seat
