#!/usr/bin/python3
import numpy as np
import math
import action_space as space


import matplotlib.pyplot as plt


class Node:

    BRANCH_MATRIX = None

    def __init__(self, location, radius, parent):
        self._value = 0
        self._location = location
        self._radius = radius
        self._low_limit = location - radius
        self._high_limit = location + radius
        self._branches = [None] * len(self.BRANCH_MATRIX)
        self._parent = parent

        self.__achieved_precision_limit = False

        if parent is not None:
            self._level = parent._level + 1
            self._value = parent._value
        else:
            self._level = 0

        if np.array_equal(self._low_limit, self._high_limit):
            # print('expansion of', self, 'stopped')
            raise ArithmeticError('Node: Low == High :{}=={}'.format(
                self._low_limit, self._high_limit))

    def expand(self, towards_point=None):
        if not self.is_expandable():
            return []
        new_radius = self._radius / 2

        new_nodes = []
        for i in range(len(self.BRANCH_MATRIX)):
            if self._branches[i] is not None:
                continue

            new_location = self._location + self.BRANCH_MATRIX[i] * new_radius

            try:
                new_node = Node(new_location, new_radius, self)
            except Exception as e:
                self.__achieved_precision_limit = True
                return new_nodes

            if (towards_point is not None) and (not new_node._covers_point(towards_point)):
                continue
            self._branches[i] = new_node
            new_nodes.append(new_node)

        return new_nodes

    def search(self, point, increase=1):
        if not self._covers_point(point):
            return None
        self._value += increase
        if self.is_leaf() or np.array_equal(self.get_location(), point):
            return self
        for branch in self.get_branches():
            res = branch.search(point, increase)
            if res is not None:
                return res

        return self

    def delete(self):
        if self.is_root():
            return

        self._parent._branches.remove(self)

    def recursive_collection(self, result_array, func, traverse_cond_func, collect_cond_func):
        if not traverse_cond_func(self):
            return
        if collect_cond_func(self):
            result_array.append(func(self))
        for branch in self.get_branches():
            branch.recursive_collection(result_array, func, traverse_cond_func, collect_cond_func)

    def get_value(self):
        return self._value

    def reset_value(self):
        self._value = 0

    def get_level(self):
        return self._level

    def get_location(self):
        return self._location

    def get_connection_with_parent(self):
        if self._parent is None:
            return self._location, self._location

        return self._parent._location, self._location

    def get_branches(self):
        res = []
        for branch in self._branches:
            if branch is not None:
                res.append(branch)
        return res

    def number_of_childs(self):
        return len(self.get_branches())

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return len(self.get_branches()) == 0

    def is_expandable(self):
        return self.number_of_childs() < len(self.BRANCH_MATRIX) or self.__achieved_precision_limit

    def _covers_point(self, point):
        check1 = self.point_less_or_equal_than_point(self._low_limit, point)
        check2 = self.point_less_or_equal_than_point(point, self._high_limit)
        return check1 and check2

    def _equals(self, node):
        return np.array_equal(self.get_location(), node.get_location())

    def __str__(self):
        return 'loc={} level={} r={} br={} v={} parent_loc={}'.format(self._location,
                                                                      self._level,
                                                                      self._radius,
                                                                      len(self.get_branches()),
                                                                      self._value,
                                                                      self._parent.get_location() if self._parent is not None else None)

    __repr__ = __str__

    @staticmethod
    def point_less_or_equal_than_point(point1, point2):
        assert len(point1) == len(point2), 'points must have same lenght'
        for i in range(len(point1)):
            if point1[i] > point2[i]:
                return False
        return True

    @staticmethod
    def _init_branch_matrix(dims):
        low = -1 * np.ones(dims)
        high = 1 * np.ones(dims)
        n = 2**dims
        Node.BRANCH_MATRIX = space.init_uniform_space(low, high, n)


"""











"""


class Exploration_tree:

    EXPANSION_VALUE_THRESHOLD = 1

    def __init__(self, dims, avg_nodes, autoprune=True):

        self._size = avg_nodes
        self._autoprune = autoprune
        self._dimensions = dims
        Node._init_branch_matrix(self._dimensions)

        self._branch_factor = self._dimensions * 2
        root = Node(np.ones(dims) * 0.5, 0.5, None)
        self._nodes = [root]
        self._root = root

        self._min_level = 0

        init_level = Exploration_tree.compute_level(avg_nodes, self._branch_factor)
        for i in range(init_level):
            self.expand_nodes(0)

        self._total_distance = 0
        self._total_distance_count = 0

    def search_nearest_node(self, point, increase=True):

        point = self.correct_point(point)

        node = self._root.search(point, increase)

        self._total_distance += np.linalg.norm(node.get_location() - point)
        print('dist ', point, node.get_location(), 'is',
              np.linalg.norm(node.get_location() - point))
        self._total_distance_count += 1

        return node

    def update(self):
        print('------------------UPDATE---------------')

        # choose a value for cut and expand
        # valid values only where selected_cut_values<selected_exp_values!!!!
        # selected_exp_index = 1
        # selected_cut_index = 0
        ########################

        # selected_exp_value = exp_unique_values[selected_exp_index]
        # selected_cut_value = cut_unique_values[selected_cut_index]
        # print('selected values exp index, value', selected_exp_index, selected_exp_value)
        # print('selected values cut index, value', selected_cut_index, selected_cut_value)

        # expand
        selected_exp_value = self._expand_threshold_value(1)

        print('selected values exp index, value', selected_exp_value)
        # exit()
        self.expand_nodes(selected_exp_value)
        # self.plot()
        # cut
        selected_cut_value = min(self._prune_threshold_value(), selected_exp_value - 1)
        print('selected values cut index, value', selected_cut_value)
        to_cut = self.get_prunable_nodes()
        for node in to_cut:
            if node.get_value() <= selected_cut_value:
                node.delete()
        self.plot()
        # print('expected new size=', delta_size_table[selected_cut_index][selected_exp_index],
        #       'final size=', self.get_current_size())

        self._refresh_nodes()
        self._reset_values()

    def get_node(self, index):
        node = self.get_nodes()[index]
        return node

    def recursive_traversal(self, func=(lambda node: node),
                            traverse_cond_func=(lambda node: True),
                            collect_cond_func=(lambda node: True)):
        res = []
        self._root.recursive_collection(res, func, traverse_cond_func, collect_cond_func)
        return res

    def _prune_threshold_value(self):

        excess = self.get_current_size() - self.get_size()
        print('size', self.get_size(), excess)
        if excess <= 0:
            return -1
            # return -1, self.get_current_size()

        values = list(node.get_value() for node in self.get_prunable_nodes())

        unique, counts = np.unique(values, return_counts=True)
        print(unique, counts)

        unique = np.insert(unique, 0, -1)
        counts = np.insert(counts, 0, 0)

        for i in range(1, len(counts)):
            counts[i] += counts[i - 1]

        print(unique, counts)

        delta_size = np.abs(counts - excess)
        print('delta', delta_size)

        result_value = unique[np.argmin(delta_size)]
        print('result', result_value)

        return result_value

    def _expand_threshold_value(self, reward_factor):
        mean_distance = self._get_mean_distance()
        max_mean_distance = self._get_max_mean_distance()
        distance_factor = mean_distance / max_mean_distance

        factor = min(distance_factor * reward_factor, 1)
        print('factor', factor, mean_distance, max_mean_distance)

        v_exp = list(node.get_value() for node in self.get_expendable_nodes())
        exp_unique_values, exp_counts = np.unique(v_exp, return_counts=True)

        # adding max+1 and removing 0
        exp_unique_values = np.append(exp_unique_values, np.max(exp_unique_values) + 1)
        exp_unique_values = exp_unique_values if exp_unique_values[0] != 0 else exp_unique_values[1:]

        print(exp_unique_values)

        cont_value = np.interp(factor, [0, 1],
                               [np.max(exp_unique_values), np.min(exp_unique_values)])
        print('cont value', cont_value)

        # find closest value to cont_value
        result_value = exp_unique_values[np.argmin(np.abs(exp_unique_values - cont_value))]

        return result_value

    def compute_delta_size(self):
        to_expand = self.get_expendable_nodes()

        to_cut = self.get_prunable_nodes()

        # computing table containing all the possible new tree sizes
        v_exp = list(node.get_value() for node in to_expand)
        v_cut = list(node.get_value() for node in to_cut)

        expand_ratio = self._branch_factor - \
            np.average(list(node.number_of_childs() for node in to_expand))
        cut_ratio = 1
        print('expand ration', expand_ratio, 'cut ratio', cut_ratio)

        exp_unique_values, exp_counts = np.unique(v_exp, return_counts=True)

        exp_unique_values = np.append(exp_unique_values,
                                      exp_unique_values[len(exp_unique_values) - 1] + 1)
        exp_counts = np.append(exp_counts, 0)

        exp_unique_values = np.flip(exp_unique_values, 0)
        exp_counts = np.flip(exp_counts, 0)

        print('expand_values\n', exp_unique_values, '\n', exp_counts)
        for i in range(1, len(exp_counts)):
            exp_counts[i] += exp_counts[i - 1]

        exp_counts = exp_counts * expand_ratio
        print('exp_counts_sum\n', exp_counts)

        cut_unique_values, cut_counts = np.unique(v_cut, return_counts=True)

        cut_unique_values = np.insert(cut_unique_values, 0, -1)
        cut_counts = np.insert(cut_counts, 0, 0)

        print('cut_values\n', cut_unique_values, '\n', cut_counts)

        for i in range(1, len(cut_counts)):
            cut_counts[i] += cut_counts[i - 1]

        cut_counts = cut_counts * cut_ratio
        print('cut_counts_sum\n', cut_counts)

        delta_size_table = []
        for i in range(len(cut_counts)):
            max_valid_index = np.where(cut_unique_values[i] == exp_unique_values)[0]
            if len(max_valid_index) == 0:
                max_valid_index = len(exp_unique_values)
            else:
                max_valid_index = max_valid_index[0]

            delta_size_table.append(exp_counts[:max_valid_index] - cut_counts[i])

        delta_size_table = np.array(delta_size_table) + self.get_current_size() - self.get_size()

        print('\nexp', exp_unique_values)
        count = 0
        for _ in delta_size_table:
            print(cut_unique_values[count], ' ', _)
            count += 1

        return delta_size_table

    def get_prunable_nodes(self):
        return self.recursive_traversal(collect_cond_func=(lambda node: node.is_leaf()))

    def get_expendable_nodes(self):
        return self.recursive_traversal(collect_cond_func=(lambda node: node.is_expandable()))

    def _get_mean_distance(self):
        if self._total_distance_count == 0:
            return 0
        result = self._total_distance / self._total_distance_count
        self._total_distance = 0
        self._total_distance_count = 0
        return result

    def _get_max_mean_distance(self):
        return 1 / (4 * self.get_current_size())

    def _reset_values(self):
        self.recursive_traversal(func=lambda node: node.reset_value())

    def expand_nodes(self, value_threshold):
        to_expand = self.get_expendable_nodes()
        for node in to_expand:
            if node.get_value() >= value_threshold:
                new_nodes = node.expand()
                self._nodes.extend(new_nodes)

    def get_nodes(self):
        return self._nodes

    def _refresh_nodes(self):
        self._nodes = self.recursive_traversal()

    def get_points(self):
        return np.array(list(node.get_location() for node in self.get_nodes()))

    def get_current_size(self):
        return len(self._nodes)

    def get_size(self):
        return self._size

    def print_all_nodes(self):
        nodes = self._nodes
        print('tree contains', len(nodes), 'nodes, min level=', self._min_level)
        for node in nodes:
            print(node)

    SAVE_ID = 0

    def plot(self, save=False, path='/home/jim/Desktop/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/results/pics'):
        nodes = self.get_nodes()
        # plt.figure()
        plt.grid(True)
        print('nodes to plot:', len(nodes))
        for node in nodes:
            parent, child = node.get_connection_with_parent()
            r = 0
            b = 0
            # if node.get_value() == 0 and node._parent is not None:

            if node.is_expandable():
                b = 255
            if node.is_leaf():
                r = 255

            if self._dimensions == 1:
                x = [child[0], parent[0]]
                y = [node._level, node._level - 1]

                plt.plot([x, x], [0.1, 0], '#000000', linewidth=0.5)

            else:
                x = [child[0], parent[0]]
                y = [child[1], parent[1]]

                plt.plot([child[0], child[0]], [-0.1, -0.12], '#000000', linewidth=.5)
                plt.plot([-0.1, -0.12], [child[1], child[1]], '#000000', linewidth=.5)

            plt.plot(x, y,
                     '#{:02x}00{:02x}'.format(r, b), linewidth=0.2)
            plt.plot(x[0], y[0],
                     '#{:02x}00{:02x}'.format(r, b), marker='.')

        if self._dimensions == 1:
            f = 0.1
            hist, _ = np.histogram(self.get_points().flatten(), bins=int(len(nodes) * f))

            hist = hist * f / len(nodes)
            max_h = np.max(hist)

            plt.plot(np.linspace(0, stop=1, num=len(hist)), hist,
                     linewidth=0.5, label='density (max {})'.format(max_h))

            v = tree.recursive_traversal(func=(lambda node: node.get_value()),
                                         collect_cond_func=lambda node: node.is_expandable())
            max_v = np.max(v)
            if max_v != 0:
                plt.plot(np.linspace(0, stop=1, num=len(v)), v /
                         np.max(v) - 1, label='values (max {})'.format(max_v))

        if save:
            plt.savefig("{}/a{}.png".format(path, self.SAVE_ID))
            self.SAVE_ID += 1
        else:
            plt.legend()
            plt.show()

    @staticmethod
    def compute_level(n, branches_of_each_node):
        total = 0
        power = -1
        prev = 0
        while total <= n:
            power += 1
            prev = total
            total += branches_of_each_node**power

        if total - n > n - prev:
            return max(power - 1, 0)
        return max(power, 0)

    @staticmethod
    def correct_point(point):
        new_point = []
        for c in point:
            if c > 1:
                new_point.append(1)
            elif c < 0:
                new_point.append(0)
            else:
                new_point.append(c)

        return new_point


if __name__ == '__main__':

    dims = 1
    tree_size = 127
    iterations = 1000
    max_size = 100

    tree = Exploration_tree(dims, tree_size)
    # tree.plot()
    samples_size_buffer = np.random.random(iterations) * max_size + 1
    samples_size_buffer = samples_size_buffer.astype(int)

    count = 0
    for i in samples_size_buffer:
        print(count, '----new iteration, searches', i)
        count += 1
        samples = 0.1 + np.abs(np.random.standard_normal((i, dims))) * 0.1
        starting_size = tree.get_current_size()
        for p in samples:
            p = list(p)
            tree.search_nearest_node(p)

        # ending_size = tree.get_size()
        # # print('added', i, 'points(', samples, '): size before-after', starting_size,
        # #       '-', ending_size, '({})'.format(ending_size - starting_size))
        # if starting_size + i != ending_size:
        #     print('ERROR')
        #     tree.plot()
        #     exit()
        # tree.plot()

        tree.update()
        tree.plot()

        exit()
    tree.plot()
