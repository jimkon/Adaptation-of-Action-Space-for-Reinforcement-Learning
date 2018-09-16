import numpy as np
import itertools
import pyflann

import matplotlib.pyplot as plt


import adiscr


"""
    This class represents a n-dimensional cube with a specific number of points embeded.
    Points are distributed uniformly in the initialization.
"""


# def gaussian(x, mu, sig):
#     return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


class Space:

    def __init__(self, low, high, points, adaptation="auto", arg1="direct", arg2=10000, arg3=10):
        """
        low, high: arrays that define the lowest and highest edge of a n-dimension box, low[i]<high[i] for each i<=n
        points: the number of points inside this box
        adaptation: can be either 'auto', 'custom' or 'off'
            'auto': if 'auto', search points are stored into a buffer with size arg2 (default=10000)
                    and when the buffer is full, action space makes an adaptation update. arg1 (default='direct')
                    specifies the error_function of the adaptive tree. arg3 is ingored.
            'custom': if 'custom', arg1 (default='direct') specifies the error_function of the adaptive tree, arg2 specifies an array
                    made by drawn samples of a custom Probability Density Function (PDF) on which action space adapts fully on initialization.
                    arg3 specifies the maximum number of iterations (default=10) of the adaptation process. Each iteration might take a while
                    depending on the resolution of the PDF.
            'off': if 'off', action space is initialized to a uniform PDF and stays the stays as it is. arg1,
                    arg2 and arg3 are ignored.
        """
        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)

        self._adaptation_flag = False

        assert adaptation in ['auto', 'custom',
                              'off'], "adaptation parameter can be either 'auto', 'custom' or 'off'"

        if adaptation is 'auto':
            try:
                self._action_space_module = adiscr.Tree(
                    self._dimensions, points, error_function=arg1)
            except AssertionError as e:
                raise Exception(
                    "When adaptation='auto' or 'custom', arg1 has to specify an error_function for the adiscr.Tree")

            assert arg2 > 0, "When adaptation='custom', arg2 has to specify a buffer size"
            self._adaptation_flag = True
            self._sample_buffer = []
            self._max_buffer_size = int(arg2)
        elif adaptation is 'custom':
            try:
                self._action_space_module = adiscr.Tree(
                    self._dimensions, points, error_function=arg1)
            except AssertionError as e:
                raise Exception(
                    "When adaptation='auto' or 'custom', arg1 has to specify an error_function for the adiscr.Tree")

            assert type(
                arg2) is np.ndarray, "When adaptation='custom', arg2 has to specify an array of samples drawn by a custom PDF"
            assert arg3 > 0, "When adaptation='custom', arg3 has to specify the maximum number of iterations of the adaptation process"

            self._action_space_module.adapt_to_samples(arg2, max_iterations=int(arg3))
        else:
            self._action_space_module = adiscr.Tree(self._dimensions, points)

        self.__space = self._action_space_module.get_points()
        self._flann = pyflann.FLANN()
        self._rebuild_flann()

    def update(self):
        if not self._adaptation_flag:
            return False

        if len(self._sample_buffer) > self._max_buffer_size:
            changed = self._action_space_module.feed_and_update(self._sample_buffer)
            self._sample_buffer = []
            if changed:
                # self._action_space_module.plot(save=True)
                self._flann.delete_index()

                self.__space = self._action_space_module.get_points()

                self._rebuild_flann()
            return changed
        return False

    def search_point(self, point, k):

        k = min(self.get_current_size(), k)
        # if self.get_current_size() < k:
        #     k = self.get_current_size()

        p_in = self._import_point(point)

        if self._adaptation_flag:
            self._sample_buffer.append(p_in)
            # self._action_space_module.search_nearest_node(p_in)

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

    def _rebuild_flann(self):
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
