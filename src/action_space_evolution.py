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

    def selection(self, population):
        # normalized_scores = np.copy(self._scores) / np.sum(self._scores)
        normalized_scores = self.normalize_scores(self._scores)
        # print('normalized_scores\n', normalized_scores)
        probs = uniform(size=normalized_scores.shape)
        # print('probs\n', probs)
        index_of_alive = np.where(normalized_scores >= probs)[0]
        # print('index_of_alive\n', index_of_alive)
        alive = population[index_of_alive]
        # print('alive\n', alive)
        # exit()
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
