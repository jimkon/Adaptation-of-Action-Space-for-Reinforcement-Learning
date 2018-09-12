#!/usr/bin/python3
import numpy as np
import itertools
import pyflann

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/jim/Desktop/dip/Adaptive-Discretization/src/')

import bin_exploration


"""
    This class represents a n-dimensional cube with a specific number of points embeded.
    Points are distributed uniformly in the initialization. A search can be made using the
    search_point function that returns the k (given) nearest neighbors of the input point.
"""


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class Space:

    def __init__(self, low, high, points):
        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)

        self._action_space_module = bin_exploration.Tree(self._dimensions, points)
        # self._action_space_module.plot(save=True)
        ################################################
        # pdf = np.array(list(gaussian(i, mu=0.5, sig=0.15) for i in np.linspace(0, 1, 10000)))
        # pdf = pdf / np.sum(pdf)
        #
        # samples = np.random.choice(np.linspace(0, 1, 10000),
        #                            10000,
        #                            p=pdf)
        # samples = np.reshape(samples, (len(samples), 1))
        #
        # for i in range(10):
        #     self._action_space_module.feed(samples)
        #     if not self._action_space_module.update():
        #         break
        # self._action_space_module.plot()
        # ################################################
        self.__space = self._action_space_module.get_points()
        # plt.hist(self.__space)
        # plt.show()
        # exit()

        self._flann = pyflann.FLANN()
        self.rebuild_flann()

    def update(self):
        changed = self._action_space_module.update()

        if changed:
            # self._action_space_module.plot(save=True)
            self._flann.delete_index()

            self.__space = self._action_space_module.get_points()

            self.rebuild_flann()

    def search_point(self, point, k):
        p_in = self._import_point(point)

        self._action_space_module.search_nearest_node(p_in)

        if self.get_current_size() < k:
            k = self.get_current_size()
        indexes, _ = self._flann.nn_index(p_in, k)

        knns = self.__space[indexes]

        p_out = []
        for p in knns:
            p_out.append(self._export_point(p))

        if k == 1:
            p_out = [p_out]

        if len(indexes.shape) == 2:
            index_out = indexes[0]
        else:
            index_out = indexes

        return np.array(p_out), index_out

    def action_selected(self, actions_index):
        # action selected for actors action and got reward
        # self._action_space_module.expand_towards(self._import_point(actors_action))
        # node = self._action_space_module.get_node(actions_index)
        # self._action_space_module.search_nearest_node(node.get_location())
        pass

    def rebuild_flann(self):
        self._index = self._flann.build_index(np.copy(self.__space), algorithm='kdtree')

    def _import_point(self, point):
        return (point - self._low) / self._range

    def _export_point(self, point):
        return self._low + point * self._range

    def get_space(self):
        return self.__space

    def get_current_size(self):
        return self._action_space_module.get_current_size()

    def get_size(self):
        return self._action_space_module.get_size()

    def shape(self):
        return self.__space.shape

    def get_number_of_actions(self):
        return self.shape()[0]

    def plot_space(self, filename=''):
        self._action_space_module.plot(save=True, filename=filename)
        return
        dims = self._dimensions

        if dims > 3:
            print(
                'Cannot plot a {}-dimensional space. Max 3 dimensions'.format(dims))
            return

        space = self.get_space()

        if additional_points is not None:
            for i in additional_points:
                space = np.append(space, additional_points, axis=0)

        if dims == 1:
            for x in space:
                plt.plot([x], [0], 'o')

            plt.show()
        elif dims == 2:
            for x, y in space:
                plt.plot([x], [y], 'o')

            plt.show()
        else:
            plot_3d_points(space)


def init_uniform_space(low, high, points):
    dims = len(low)
    points_in_each_axis = round(points**(1 / dims))

    axis = []
    for i in range(dims):
        axis.append(list(np.linspace(low[i], high[i], points_in_each_axis)))

    space = []
    for _ in itertools.product(*axis):
        space.append(list(_))

    return np.array(space)
