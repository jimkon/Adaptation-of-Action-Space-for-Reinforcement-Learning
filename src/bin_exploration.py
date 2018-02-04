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
        self._low_limit = (-radius) * np.ones(len(location)) + location
        self._high_limit = (radius) * np.ones(len(location)) + location
        self._branches = []
        self._parent = parent
        if parent is not None:
            self._level = parent._level + 1
        else:
            self._level = 0

    def expand(self):
        if not self.is_leaf():
            return None
        new_radius = self._radius / 2
        for mat in self.BRANCH_MATRIX:
            new_location = self._location + mat * new_radius
            self._branches.append(Node(new_location, new_radius, self))

        return self._branches

    def search(self, point, increase=True):
        if not self._covers_point(point):
            return None
        if increase:
            self.increase_value()
        if self.is_leaf():
            return self
        for branch in self._branches:
            res = branch.search(point, increase)
            if res is not None:
                return res

        return None

    def recursive_pruning(self, min_level, value_threshold):
        if self.is_leaf():
            return
        if self._level >= min_level and self._value <= value_threshold:
            print('prune', self)
            self._branches = []

        else:
            for branch in self._branches:
                branch.recursive_pruning(min_level, value_threshold)

    def collect_sub_branches(self):
        if self.is_leaf():
            return [self]
        else:
            result = [self]
            for branch in self._branches:
                result.extend(branch.collect_sub_branches())
            return result

    def get_value(self):
        return self._value

    def reset_value(self):
        self._value = 0

    def get_location(self):
        return self._location

    def get_connection_with_parent(self):
        if self._parent is None:
            return self._location, self._location

        return self._parent._location, self._location

    def is_leaf(self):
        return len(self._branches) == 0

    def increase_value(self):
        self._value += 1

    def _covers_point(self, point):
        check1 = self.point_less_or_equal_than_point(self._low_limit, point)
        check2 = self.point_less_or_equal_than_point(point, self._high_limit)
        return check1 and check2

    def __str__(self):
        return 'loc={} level={} r={} br={} v={}'.format(self._location,
                                                        self._level,
                                                        self._radius,
                                                        len(self._branches),
                                                        self._value)

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


class Exploration_tree:

    def __init__(self, dims, n):
        self._dimensions = dims
        self._branch_factor = self._dimensions * 2
        root = Node(np.ones(dims) * 0.5, 0.5, None)
        self._nodes = [root]
        self._lenght = 1
        self._root = root
        Node._init_branch_matrix(self._dimensions)

        self._min_level = self.compute_level(n, self._branch_factor)
        self._add_layers(self._min_level)

    def expand_towards(self, point):
        assert len(point) == self._dimensions, 'input point must have same number of dimensions: {}'.format(
            self._dimensions)
        node = self.search_nearest_node(point, increase=True)
        if node is not None:
            self._expand_node(node)

    def prune(self, value_threshold=0):
        self._root.recursive_pruning(self._min_level, value_threshold)
        self._nodes = self._root.collect_sub_branches()
        self._lenght = len(self._nodes)
        self._reset_values()

    def search_nearest_node(self, point, increase=True):
        node = self._root.search(point, increase)
        return node

    def get_node(self, index):
        node = self.get_nodes()[index]
        node.increase_value()
        return node

    def _expand_node(self, node):
        new_nodes = node.expand()
        if new_nodes is None:
            return
        self._nodes.extend(new_nodes)
        self._lenght += len(new_nodes)

    def _reset_values(self):
        nodes = self.get_nodes()
        for node in nodes:
            node.reset_value()

    def _add_layers(self, n):
        for i in range(n):
            self._add_layer()

    def _add_layer(self):
        current_nodes = np.copy(self.get_nodes())
        for node in current_nodes:
            self._expand_node(node)

    def get_nodes(self):
        return self._nodes

    def get_location(self):
        nodes = self.get_nodes()
        res = []
        for node in nodes:
            res.append(node.get_location())

        return res

    def plot(self):
        nodes = self.get_nodes()
        plt.figure()
        plt.grid(True)
        print('nodes to plot:', len(nodes))
        for node in nodes:
            print(node)
            parent, child = node.get_connection_with_parent()
            color = (50 * node._level) % 255
            # plt.plot([parent[0], child[0]], [parent[1], child[1]],
            #          '#{:02x}0000'.format(color), marker='*')
            if self._dimensions == 1:
                x = child[0]
                y = node._level
                plt.plot([x, x], [-0.9, -1], '#000000', linewidth=1)
            else:
                x = child[0]
                y = child[1]
            plt.plot([x], [y],
                     '#0000{:02x}'.format(255 if node.get_value() == 0 else 0), marker='1')

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


if __name__ == '__main__':

    tree = Exploration_tree(1, 20)

    samples = np.random.standard_normal(20)
    for p in samples:
        tree.expand_towards([p])

    for i in range(10):
        p = [np.random.normal(loc=0.5, scale=.1)]
        # print(p)
        # tree.search_nearest_node(p)
    tree.plot()
    print('pruning above', tree._min_level, 'starting size = ', tree._lenght)
    tree.prune()
    print('size after pruning = ', tree._lenght)
    tree.plot()
