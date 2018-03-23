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

    def search(self, point, increase=True):
        if not self._covers_point(point):
            return None
        if increase:
            self.increase_value()
        if self.is_leaf() or np.array_equal(self.get_location(), point):
            return self
        for branch in self.get_branches():
            res = branch.search(point, increase)
            if res is not None:
                return res

        return self

    def delete(self):
        print('delete')
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

    def is_root(self):
        return self._parent is None

    def is_leaf(self):
        return len(self.get_branches()) == 0

    def is_expandable(self):
        return len(self.get_branches()) < len(self.BRANCH_MATRIX) or self.__achieved_precision_limit

    def increase_value(self):
        self._value += 1

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

        self._limit_size = avg_nodes
        self._autoprune = autoprune
        self._dimensions = dims
        Node._init_branch_matrix(self._dimensions)

        self._branch_factor = self._dimensions * 2
        root = Node(np.ones(dims) * 0.5, 0.5, None)
        self._nodes = [root]
        self._root = root

        self._min_level = 0

        init_level = Exploration_tree.compute_level(avg_nodes, self._branch_factor)
        self._add_layers(init_level)
        self._refresh_nodes()

    def search_nearest_node(self, point, increase=True):
        node = self._root.search(point, increase)
        return node

    def get_node(self, index):
        node = self.get_nodes()[index]
        return node

    def recursive_traversal(self, func=(lambda node: node),
                            traverse_cond_func=(lambda node: True),
                            collect_cond_func=(lambda node: True)):
        res = []
        self._root.recursive_collection(res, func, traverse_cond_func, collect_cond_func)
        return res

    # def _get_cutoff_value(self):
    #     excess = self.get_size() - self._limit_size
    #     if excess < 0:
    #         return -1, self.get_size()
    #     values = self.recursive_traversal(
    #         lambda node: node.get_value() if node.get_level() > self._min_level else -1)
    #
    #     unique, counts = np.unique(values, return_counts=True)
    #     counts[0] = 0  # drop the -1
    #     cumulative = []
    #     total = 0
    #     for i in counts:
    #         total += i
    #         cumulative.append(total)
    #
    #     diff = np.array(cumulative)
    #     new_size = self.get_size() - diff
    #     diff_from_pref_size = np.abs(new_size - self._limit_size)
    #
    #     argmin = np.argmin(diff_from_pref_size)
    #     res_value = unique[argmin]
    #
    #     return res_value, new_size[argmin]

    def get_leaves(self):
        return self.recursive_traversal(collect_cond_func=(lambda node: node.is_leaf()))

    def _reset_values(self):
        self.recursive_traversal(func=lambda node: node.reset_value())

    def _add_layers(self, n):
        for i in range(n):
            self._add_layer()

    def _add_layer(self):
        leaves = self.get_leaves()
        for node in leaves:
            node.expand()

    def get_nodes(self):
        return self._nodes

    def _refresh_nodes(self):
        self._nodes = self.recursive_traversal()

    def get_points(self):
        return np.array(list(node.get_location() for node in self.get_nodes()))

    def get_size(self):
        return len(self._nodes)

    def get_limit_size(self):
        return self._limit_size

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
            if node.get_value() == 0 and node._parent is not None:

                if node._level > self._min_level and node.get_value() == 0:
                    b = 255
                else:
                    r = 255
            if self._dimensions == 1:
                x = [child[0], parent[0]]
                y = [node._level, node._level - 1]

                plt.plot([x, x], [-0.1, -0.2], '#000000', linewidth=0.5)
            else:
                x = [child[0], parent[0]]
                y = [child[1], parent[1]]

                plt.plot([child[0], child[0]], [-0.1, -0.12], '#000000', linewidth=.5)
                plt.plot([-0.1, -0.12], [child[1], child[1]], '#000000', linewidth=.5)

            plt.plot(x, y,
                     '#{:02x}00{:02x}'.format(r, b), linewidth=0.2)
            plt.plot(x[0], y[0],
                     '#{:02x}00{:02x}'.format(r, b), marker='.')

        if save:
            plt.savefig("{}/a{}.png".format(path, self.SAVE_ID))
            self.SAVE_ID += 1
        else:
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
    tree_size = 15
    iterations = 1000
    max_size = 10

    tree = Exploration_tree(dims, tree_size)

    tree.plot()
    print(len(tree.get_leaves()))
    # samples_size_buffer = np.random.random(iterations) * max_size + 1
    # samples_size_buffer = samples_size_buffer.astype(int)
    #
    # count = 0
    # for i in samples_size_buffer:
    #     print(count, '----new iteration, searches', i)
    #     count += 1
    #     samples = np.abs(np.random.standard_normal((i, dims))) * 0.4
    #     starting_size = tree.get_size()
    #     for p in samples:
    #         p = list(p)
    #         tree.expand_towards(p)
    #
    #     ending_size = tree.get_size()
    #     # print('added', i, 'points(', samples, '): size before-after', starting_size,
    #     #       '-', ending_size, '({})'.format(ending_size - starting_size))
    #     if starting_size + i != ending_size:
    #         print('ERROR')
    #         tree.plot()
    #         exit()
    #
    #     tree.prune()
    # tree.plot()
