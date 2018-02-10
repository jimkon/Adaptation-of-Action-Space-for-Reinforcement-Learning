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

    def expand(self, full_expansion=True, towards=None):
        if not self.is_expandable():
            return None
        new_radius = self._radius / 2

        temp_branches = []
        for mat in self.BRANCH_MATRIX:
            new_location = self._location + mat * new_radius
            new_node = Node(new_location, new_radius, self)
            for node in self._branches:
                if node._equals(new_node):
                    continue
            temp_branches.append(new_node)

        if (not full_expansion) and (towards is not None):
            for node in temp_branches:
                if node._covers_point(towards):
                    self._branches.append(node)
                    break
        else:
            self._branches = temp_branches

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

        return self

    def recursive_pruning(self, min_level, value_threshold):
        if self.is_leaf():
            return

        to_remove = []
        for node in self._branches:
            if node.get_value() <= value_threshold and node.get_level() > min_level:
                to_remove.append(node)

        for node in to_remove:
            self._branches.remove(node)

        for node in self._branches:
            node.recursive_pruning(min_level, value_threshold)

        # needs_pruning = self._level >= min_level and self._value <= value_threshold
        # if needs_pruning:
        #     self._branches = []
        # else:
        #     for branch in self._branches:
        #         branch.recursive_pruning(min_level, value_threshold)

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

    def get_level(self):
        return self._level

    def get_location(self):
        return self._location

    def get_connection_with_parent(self):
        if self._parent is None:
            return self._location, self._location

        return self._parent._location, self._location

    def is_leaf(self):
        return len(self._branches) == 0

    def is_expandable(self):
        return len(self._branches) < len(self.BRANCH_MATRIX)

    def increase_value(self):
        self._value += 1

    def _covers_point(self, point):
        check1 = self.point_less_or_equal_than_point(self._low_limit, point)
        check2 = self.point_less_or_equal_than_point(point, self._high_limit)
        return check1 and check2

    def _equals(self, node):

        return np.array_equal(self.get_location(), node.get_location())

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

    FULL_EXPANSION = True
    EXPANSION_VALUE_THRESHOLD = 1
    AUTOPRUNE_PERCENTAGE = .8
    INIT_TO_MAX_ACTIONS_RATIO = .01

    def __init__(self, dims, max_nodes, autoprune=False):
        self._max_size = max_nodes
        self._autoprune = autoprune
        self._dimensions = dims
        self._branch_factor = self._dimensions * 2
        root = Node(np.ones(dims) * 0.5, 0.5, None)
        self._nodes = [root]
        self._lenght = 1
        self._root = root
        Node._init_branch_matrix(self._dimensions)

        init_actions = int(max(5, self._max_size * self.INIT_TO_MAX_ACTIONS_RATIO))

        self._min_level = self.compute_level(init_actions, self._branch_factor)
        self._add_layers(self._min_level)

        self.value_threshold = 0

    def expand_towards(self, point):
        assert len(point) == self._dimensions, 'input point must have same number of dimensions: {} given point: {}'.format(
            self._dimensions, point)

        node = self.search_nearest_node(point, increase=True)
        if (node is not None) and self.get_lenght() < self._max_size:
            self._expand_node(node,
                              towards=point)

    def prune(self, value_threshold=0):
        if self._autoprune and (self.get_lenght() / self._max_size < self.AUTOPRUNE_PERCENTAGE):
            return
        self._root.recursive_pruning(self._min_level, value_threshold)
        self._nodes = self._root.collect_sub_branches()
        self._lenght = len(self._nodes)
        self._reset_values()

    def search_nearest_node(self, point, increase=True):
        node = self._root.search(point, increase)
        return node

    def get_node(self, index):
        node = self.get_nodes()[index]
        # self.expand_towards(node.get_location())
        return node
        # return self.search_nearest_node(node.get_location())

    def _expand_node(self, node, towards=None):
        if node.get_level() > self._min_level and node.get_value() < self.EXPANSION_VALUE_THRESHOLD:
            return

        new_nodes = node.expand(self.FULL_EXPANSION, towards)
        if new_nodes is None:
            return
        else:
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

    def get_points(self):
        result = []
        nodes = self.get_nodes()
        for node in nodes:
            result.append(node.get_location())
        return np.array(result)

    def get_lenght(self):
        return self._lenght

    def get_max_lenght(self):
        return self._max_size

    SAVE_ID = 0

    def plot(self, save=False, path='/home/jim/Desktop/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/results/pics'):
        nodes = self.get_nodes()
        plt.figure()
        plt.grid(True)
        print('nodes to plot:', len(nodes))
        for node in nodes:
            # print(node)
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


if __name__ == '__main__':

    dims = 1

    tree = Exploration_tree(dims, 10)

    # for i in [2, 20, 30]:
    #     samples = np.abs(0.3 * np.random.standard_normal((i, dims))) % 1
    #     print('samples added', len(samples), samples)
    #     for p in samples:
    #         p = list(p)
    #         tree.expand_towards(p)
    #
    #     tree.plot()
    #     print('pruning above', tree._min_level, 'starting size = ', tree._lenght)
    #
    #     tree.prune()
    #
    #     print('size after pruning = ', tree._lenght)
    #     tree.plot()

    # print(tree.get_node(np.random.randint(0, high=tree.get_lenght())))
    # tree.plot()
    # tree.prune()
    # tree.plot()
