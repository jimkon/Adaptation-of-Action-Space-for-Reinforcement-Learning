import numpy as np
import itertools
import pyflann

import util.my_plotlib as mplt
from util.data_graph import plot_3d_points
import action_space_evolution as ev


"""
    This class represents a n-dimensional cube with a specific number of points embeded.
    Points are distributed uniformly in the initialization. A search can be made using the
    search_point function that returns the k (given) nearest neighbors of the input point.
"""


class Space:

    def __init__(self, low, high, points, monitor):
        self._low = np.array(low)
        self._high = np.array(high)
        self._range = self._high - self._low
        self._dimensions = len(low)

        self.__space = init_uniform_space(np.zeros(self._dimensions),
                                          np.ones(self._dimensions),
                                          points)
        self.__actions_score = np.zeros(self.__space.shape[0])
        # self.__actions_evolution = ev.Action_space_evolution()
        # self._actions_evolution = ev.Genetic_Algorithm()
        self._actions_evolution = ev.ParticleFilter(self.__space, points * 2)

        self._flann = pyflann.FLANN()
        self.rebuild_flann()

        self.monitor = monitor
        self.monitor.add_arrays(['space', 'usage', 'lenght', 'actors_action'])

    def rebuild_flann(self):
        self._index = self._flann.build_index(np.copy(self.__space), algorithm='kdtree')

    def update(self):
        self._flann.delete_index()
        self.monitor.add_to_array('usage', self.__actions_score)
        self.monitor.add_to_array('space', self.__space)
        self.monitor.add_to_array('lenght', len(self.__space))

        new_actions = self._actions_evolution.get_next_generation(self.__actions_score)
        self.__space = np.reshape(new_actions, (len(new_actions), self._dimensions))

        self.__actions_score = np.zeros(self.__space.shape[0])
        self.rebuild_flann()

    def new_actions(self):
        return self.__space[np.where(self.__actions_score > 0)]

    def search_point(self, point, k):
        p_in = self._import_point(point)
        indexes, _ = self._flann.nn_index(p_in, k)
        knns = self.__space[indexes]
        p_out = []
        for p in knns:
            p_out.append(self._export_point(p))

        return np.array(p_out)[0], indexes[0]

    # remove proto action
    def action_selected(self, actions_index, actors_action):
        # action selected for actors action and got reward
        self.__actions_score[actions_index] += 1
        self.monitor.add_to_array('actors_action', self._import_point(actors_action))

    def _import_point(self, point):
        return (point - self._low) / self._range

    def _export_point(self, point):
        return self._low + point * self._range

    def get_space(self):
        return self.__space

    def shape(self):
        return self.__space.shape

    def get_number_of_actions(self):
        return self.shape()[0]

    def plot_usage(self):
        lines = []
        count = self.__actions_score
        x = self.get_space()

        # lines.append(mplt.Line(x, count, line_color="b"))
        for i in range(len(x)):
            lines.append(mplt.Line([x[i], x[i]], [-.1, count[i]], line_color='#000000'))

        mplt.plot_lines(lines, labels=False)

    def plot_space(self, additional_points=None):

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
            lines = []
            for x in space:
                lines.append(mplt.Line([x], [0], line_color='o'))

            mplt.plot_lines(lines, labels=False)
        elif dims == 2:
            lines = []
            for x, y in space:
                lines.append(mplt.Line([x], [y], line_color='o'))

            mplt.plot_lines(lines, labels=False)
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
