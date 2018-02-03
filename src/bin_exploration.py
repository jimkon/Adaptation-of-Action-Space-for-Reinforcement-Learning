#!/usr/bin/python3
import numpy as np
import action_space as space


import matplotlib.pyplot as plt


class Node:

    BRANCH_MATRIX = None

    def __init__(self, location, radius, parent):
        self._value = 0
        self._location = location
        self._radius = radius
        self._branches = []
        self._parent = parent
        if parent is not None:
            self._level = parent._level + 1
        else:
            self._level = 0

    def expand(self):
        if len(self._branches) > 0:
            return None
        new_radius = self._radius / 2
        for mat in self.BRANCH_MATRIX:
            new_location = self._location + mat * new_radius
            self._branches.append(Node(new_location, new_radius, self))

        return self._branches

    def get_connection_with_parent(self):
        if self._parent is None:
            return self._location, self._location

        return self._parent._location, self._location

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
        self.add_layer()
        self.add_layer()
        self.add_layer()
        # self.add_layer()

    def add_layer(self):
        current_nodes = np.copy(self.get_nodes())
        for node in current_nodes:
            self._expand_node(node)

    def _expand_node(self, node):
        new_nodes = node.expand()
        if new_nodes is None:
            return
        self._nodes.extend(new_nodes)
        self._lenght += len(new_nodes)

    def get_nodes(self):
        return self._nodes

    def plot(self):
        nodes = self.get_nodes()
        plt.figure()
        plt.grid(True)
        print(len(nodes))
        for node in nodes:
            parent, child = node.get_connection_with_parent()
            color = 50 * node._level
            plt.plot([parent[0], child[0]], [parent[1], child[1]],
                     '#{:02x}0000'.format(color), marker='o')

        plt.show()


if __name__ == '__main__':
    tree = Exploration_tree(2, 1)

    tree.plot()
