#!/usr/bin/python3
import numpy as np
from my_plotlib import *
import matplotlib.pyplot as plt
from data import *
import math
import data_graph
import gym
from gym.spaces import Box, Discrete

"""data wrapper for action space fields, and some graphs"""


class Action_space_data(Data):

    def __init__(self, low=[], high=[], points=0, load_path=None):
        if load_path is None:
            self.max_points = points
            self.dims = len(low)
            name = 'action_space_{}dim_{}'.format(self.dims, self.max_points)
            super().__init__(name)

            self.add_arrays(['actors_action', 'nearest_discrete_neighbor',
                             'search_point', 'lenght', 'dims', 'max_points'])
            self.add_to_array('dims', self.dims)
            self.add_to_array('max_points', self.max_points)
        else:
            name = load_path
            super().__init__(name)

    def store_ndn(self, point):
        self.add_to_array('nearest_discrete_neighbor', point)

    def store_search_point(self, point):
        self.add_to_array('search_point', point)

    def store_continuous_action(self, point):
        self.add_to_array('actors_action', point)

    def store_lenght(self, l):
        self.add_to_array('lenght', l)

    def get_mean_action_error(self):
        # self.get_data('')
        pass


if __name__ == '__main__':
    monitor = Action_space_data(load_path='action_space_1dim_1000')
    monitor.load()
    monitor.print_data()

    # plot_action_density(monitor)
    # plot_space_adaption_history(monitor)
    # plot_lenghts(monitor)
    # plot_space_and_actions_across_episodes(monitor)

    # print(monitor.break_into_episodes('space'))
