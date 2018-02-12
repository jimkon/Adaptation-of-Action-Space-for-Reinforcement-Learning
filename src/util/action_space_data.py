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


def plot_actions(mon):

    proto_actions = mon.get_data('search_point')
    plt.plot(proto_actions, label='proto action')

    nn_actions = mon.get_data('nearest_discrete_neighbor')
    plt.plot(nn_actions, label='nn')

    actions = mon.get_data('action')
    plt.plot(actions, label='final action')

    mean_error = mon.get_mean_action_error()
    print(mean_error[len(mean_error) - 1])
    plt.plot(mean_error, label='mean error')

    # s_action = mon.get_data('search_point')
    # c_actions = mon.get_data('actors_action')
    plt.legend()
    plt.grid(True)
    plt.show()


class Action_space_data(Data):

    # action: the action that is the final result
    # search_point: the point that wolp asks for search
    # nearest_discrete_neighbor: the nearest discrete action to the search point

    def __init__(self, low=[], high=[], points=0, load_path=None):
        if load_path is None:
            self.max_points = points
            self.dims = len(low)
            name = 'action_space_{}dim_{}'.format(self.dims, self.max_points)
            super().__init__(name)

            self.add_arrays(['action', 'nearest_discrete_neighbor',
                             'search_point', 'lenght', 'dims', 'max_points'])
            self.add_to_array('dims', self.dims)
            self.add_to_array('max_points', self.max_points)
        else:
            name = load_path
            super().__init__(name)

    def store_action(self, point):
        self.add_to_array('action', point)

    def store_ndn(self, point):
        self.add_to_array('nearest_discrete_neighbor', point)

    def store_search_point(self, point):
        self.add_to_array('search_point', point)

    def store_lenght(self, l):
        self.add_to_array('lenght', l)

    def get_mean_action_error(self):
        proto_actions = self.get_data('search_point')
        nn_actions = self.get_data('nearest_discrete_neighbor')

        abs_sum = np.absolute(proto_actions - nn_actions)
        print('avg', np.average(abs_sum))

        res = []
        count = 0
        total = 0
        for i in abs_sum:
            total += i
            count += 1
            res.append(total / count)
        return res


if __name__ == '__main__':
    monitor = Action_space_data(load_path='action_space_1dim_10000')
    monitor.load()
    monitor.print_data()

    # print('mean error', monitor.get_mean_action_error())

    plot_actions(monitor)
