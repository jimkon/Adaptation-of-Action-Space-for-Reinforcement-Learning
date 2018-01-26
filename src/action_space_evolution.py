import numpy as np
from numpy.random import uniform
import math


def temp_plot(d1, d2):
    import matplotlib.pyplot as plt
    plt.hist(d1, bins=len(d1))
    plt.plot(np.linspace(0, 1, len(d2)), d2)
    plt.show()


class Action_space_evolution:

    def update_population(self, population, scores):
        self._population = population
        self._scores = scores
        # can check if the shape of population and scores are the same

    def get_next_generation(self):
        return self._population


class ParticleFilter(Action_space_evolution):

    ADAPTION_FACTOR = .1
    ZERO_OFFSET = .005
    MIN_POPULATION = .2

    def __init__(self, max_actions):
        self._upper_limit = int(max_actions)
        self._lower_limit = int(max_actions * self.MIN_POPULATION)
        self._sigma = 1 / max_actions  # ????

    def _utility_factor(self):
        n_prev = len(self._population)
        # does not count how many times each action used
        n_used = len(np.where(self._scores > 0)[0])
        return n_used / n_prev

    def _calculate_next_gens_size(self):
        f_util = self._utility_factor()
        # increase f_util because we want always excess in actions
        f_util += self.ADAPTION_FACTOR
        valid_factors = [1 - self.ADAPTION_FACTOR, 1 + self.ADAPTION_FACTOR]
        factor = valid_factors[np.argmin(
            np.abs(
                np.array(
                    valid_factors)
                - np.array(
                    [f_util] * len(valid_factors))))]

        result = int(math.ceil(factor * len(self._population)))
        # return max(min(result, self._upper_limit), self._lower_limit)
        return result

    def _get_distribution(self):
        n = len(self._scores)
        total = np.sum(self._scores)
        noise = np.random.standard_exponential(self._scores.shape) * self.ZERO_OFFSET
        temp = noise + self._scores / total
        total = np.sum(temp)
        return temp / total

    def _sample_new_particles(self, n):
        distribution = self._get_distribution()
        samples = np.random.choice(len(self._population),
                                   size=n,
                                   p=distribution)
        new_particles = self._population[samples]
        noise = np.random.standard_exponential(new_particles.shape) * self._sigma
        return new_particles + noise

    def get_next_generation(self):
        temp = np.copy(self._scores)
        next_gens_size = self._calculate_next_gens_size()
        next_gen = self._sample_new_particles(next_gens_size)
        next_gen = np.sort(next_gen, axis=0)
        return next_gen

        # not fully implemented


class Genetic_Algorithm(Action_space_evolution):

    """
    alive/total percentage must be adjustable.
    offsprings/alive percentage must be adjustable.
    next_gen/prev_gen percentage must be adjustable.
    """

    def selection(self, population):
        normalized_scores = self.normalize_scores(self._scores)
        probs = uniform(size=normalized_scores.shape)
        index_of_alive = np.where(normalized_scores >= probs)[0]
        alive = population[index_of_alive]
        return alive

    def crossover(self, population):
        offsprings = 1

    def mutation(self, individual):
        pass

    def get_next_generation(self):
        next_gen = super().get_next_generation()
        alive = self.selection(next_gen)
        return alive
        # return self.crossover(alive)

    # def normalize_scores(self, scores):
        # return self._scores[:, 0] / np.max(self._scores)

    def normalize_scores(self, scores):
        return 1 - 1 / (2 + self._scores[:, 0])
