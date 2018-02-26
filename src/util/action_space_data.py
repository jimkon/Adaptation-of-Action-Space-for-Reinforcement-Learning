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
    plt.plot(mean_error, label='mean error')

    # s_action = mon.get_data('search_point')
    # c_actions = mon.get_data('actors_action')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_actions_distribution(mon):
    bins = 30
    bw = 0.9

    proto_actions = mon.get_data('search_point')
    nn_actions = mon.get_data('nearest_discrete_neighbor')
    actions = mon.get_data('action')

    plt.hist([proto_actions, nn_actions, actions], bins=bins,
             rwidth=bw, align='mid', label=['proto action', 'nn', 'final action'])

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_lenghts(mon):
    lenghts = mon.get_data('lenght')
    plt.plot(lenghts, label='lenghts')

    avg = average_timeline(lenghts)
    plt.plot(avg, label='avg = {}'.format(avg[len(avg) - 1]))

    min_l = np.min(lenghts[1:])
    plt.plot([0, len(lenghts)], [min_l, min_l], label='min = {}'.format(min_l))

    max_l = np.max(lenghts)
    plt.plot([0, len(lenghts)], [max_l, max_l], label='max = {}'.format(max_l))

    plt.legend()
    plt.grid(True)
    plt.show()


def plot_lenght_distribution(mon):
    lenghts = mon.get_data('lenght')
    plt.hist(lenghts, bins='auto', histtype='bar', rwidth=0.6)
    plt.show()


def plot_error(mon):
    mean_error = mon.get_mean_action_error()
    plt.plot(mean_error, label='mean error = {}'.format(mean_error[len(mean_error) - 1]))

    lenghts = mon.get_data('lenght')
    avg = np.average(lenghts)
    expected_error = 1.0 / (4 * avg)  # discretization step /4
    plt.plot([0, len(mean_error)], [expected_error, expected_error],
             label='expected error = {}'.format(expected_error))

    proto_actions = mon.get_data('search_point')
    nn_actions = mon.get_data('nearest_discrete_neighbor')

    abs_sum = np.absolute(proto_actions - nn_actions)
    wind_avg = apply_func_to_window(abs_sum, int(0.01 * len(abs_sum)), np.average)
    plt.plot(wind_avg, label='windowed avg')

    max_points = mon.get_data('max_points')
    min_expected_error = 1.0 / (4 * max_points)  # discretization step /4
    plt.plot([0, len(mean_error)], [min_expected_error, min_expected_error],
             label='min expected error = {}'.format(min_expected_error))

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
        # print('avg', np.average(abs_sum))

        return average_timeline(abs_sum)


def average_timeline(x):
    res = []
    count = 0
    total = 0
    for i in x:
        total += i
        count += 1
        res.append(total / count)
    return res


def apply_func_to_window(data, window_size, func):
    data_lenght = len(data)
    res = []
    for i in range(data_lenght):
        start = int(max(i - window_size / 2, 0))
        end = int(min(i + window_size / 2, data_lenght - 1))
        res.append(func(data[start:end]))

    return res


if __name__ == '__main__':
    monitor = Action_space_data(load_path='action_space_1dim_1002')
    monitor.load()
    monitor.print_data()

    # print('mean error', monitor.get_mean_action_error())

    # plot_actions(monitor)
    # plot_actions_distribution(monitor)
    # plot_lenghts(monitor)
    # plot_lenght_distribution(monitor)
    plot_error(monitor)
