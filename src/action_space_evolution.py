import numpy as np
from numpy.random import uniform


class Action_space_evolution:

    def update_population(self, population, scores):
        self._population = population
        self._scores = scores
        # can check if the shape of population and scores are the same

    def get_next_generation(self):
        return self._population


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
