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

        self.reset_value()

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

    def search(self, point, min_dist_till_now=1):
        if not self._covers_point(point):
            return None, 0

        dist_to_self = np.linalg.norm(self._location - point)

        if min_dist_till_now > dist_to_self:
            min_dist_till_now = dist_to_self

        branches = self.get_branches()
        for branch_i in range(len(branches)):
            branch = branches[branch_i]
            res, branch_dist = branch.search(point, min_dist_till_now)
            if res is not None:
                # print('dist to child', branch, '=', branch_dist)

                if branch_dist > dist_to_self:
                    self._value += dist_to_self
                    return res, dist_to_self
                else:
                    self._value_without_branch[branch_i] += dist_to_self
                    return res, branch_dist

        # print(self, point, dist_to_self if min_dist_till_now == dist_to_self else 0)
        self._value += dist_to_self if min_dist_till_now == dist_to_self else 0
        return self, dist_to_self

    def delete(self):
        if self.is_root():
            return
        # uncaught exception !!!!
        self._parent._branches[self._parent._branches.index(self)] = None

    def recursive_collection(self, result_array, func, traverse_cond_func, collect_cond_func):
        if not traverse_cond_func(self):
            return
        if collect_cond_func(self):
            result_array.append(func(self))
        for branch in self.get_branches():
            branch.recursive_collection(result_array, func, traverse_cond_func, collect_cond_func)

    def get_value(self):
        return self._value

    # def

    def reset_value(self):
        self._value = 0
        self._value_without_branch = np.zeros(len(self.BRANCH_MATRIX))

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


class Tree:

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

        init_level = Tree.compute_level(avg_nodes, self._branch_factor)
        for i in range(init_level):
            self.expand_nodes(0)

        self._total_distance = 0
        self._total_distance_count = 0

    def search_nearest_node(self, point, increase=True):

        point = Tree.correct_point(point)

        node, dist = self._root.search(point, increase)

        self._total_distance += dist

        self._total_distance_count += 1

        return node

    def update(self, reward_factor=1):
        # print('------------------UPDATE---------------')
        debug_flag = False

        if debug_flag:
            nodes = list(({'loc': node.get_location()[0], 'v': node.get_value()}
                          for node in self.get_expendable_nodes()))
            nodes = sorted(nodes, key=lambda node: node['loc'])
            points = list(item['loc'] for item in nodes)
            values = list(item['v'] for item in nodes)
            max_v = np.max(values)
            plt.plot(points, values, 'b^--', label='values (max={})'.format(max_v))

        # find the expand value
        selected_exp_value = self._expand_threshold_value(reward_factor)
        size_before = self.get_current_size()
        # expand nodes with values greater or equal to the choosen one
        self.expand_nodes(selected_exp_value)
        size_after = self.get_current_size()
        # print('selected values exp index, value', selected_exp_value,
        #       '# new nodes', size_after - size_before)

        # find the cut value
        selected_cut_value = self._prune_threshold_value(max_threshold=selected_exp_value)
        assert selected_cut_value < selected_exp_value, 'cut value > expand value'

        size_before = self.get_current_size()
        to_cut = self.get_prunable_nodes()
        # cut the nodes with values below (not equal) to the choosen one
        for node in to_cut:
            if node.get_value() <= selected_cut_value:
                node.delete()

        # self.plot()

        self._refresh_nodes()

        ######
        if debug_flag:
            nodes = list(({'loc': node.get_location()[0], 'v': node.get_value()}
                          for node in self.get_expendable_nodes()))
            nodes = sorted(nodes, key=lambda node: node['loc'])
            points = list(item['loc'] for item in nodes)
            values = list(item['v'] for item in nodes)
            max_v = np.max(values)
            # values = values / max_v
            # values = apply_func_to_window(values, int(.1 * len(values)), np.average)

            plt.plot([0, 1], [selected_exp_value, selected_exp_value],
                     'g', label='exp = {}'.format(selected_exp_value))

            plt.plot([0, 1], [selected_cut_value, selected_cut_value],
                     'r', label='cut = {}'.format(selected_cut_value))
            plt.plot(points, values, 'mv--', label='values (max={})'.format(max_v))
            plt.grid(True)
            plt.legend()
            plt.show()
        ####

        self._reset_values()

        size_after = self.get_current_size()
        # print('selected values cut index, value', selected_cut_value,
        #       '# deleted nodes', size_after - size_before)

    def _prune_threshold_value(self, max_threshold=np.inf):

        excess = self.get_current_size() - self.get_size()
        if excess <= 0:
            return -1

        values = list(node.get_value() for node in self.get_prunable_nodes())

        unique, counts = np.unique(values, return_counts=True)

        valid_values_indexex = np.where(unique < max_threshold)[0]

        unique = unique[valid_values_indexex]
        counts = counts[valid_values_indexex]

        unique = np.insert(unique, 0, -1)
        counts = np.insert(counts, 0, 0)

        for i in range(1, len(counts)):
            counts[i] += counts[i - 1]

        delta_size = np.abs(counts - excess)

        result_value = unique[np.argmin(delta_size)]

        return result_value

    def _expand_threshold_value(self, factor=1):

        factor = max(factor, .01)

        v_exp = list(node.get_value() for node in self.get_expendable_nodes())

        avg_v = np.average(v_exp)

        result_value = avg_v / factor

        return result_value

    def _evaluate(self):
        mean_distance = self._get_mean_distance()
        max_mean_distance = self._get_max_mean_distance()
        distance_factor = mean_distance / max_mean_distance
        return distance_factor

    def get_node(self, index):
        node = self.get_nodes()[index]
        return node

    def recursive_traversal(self, func=(lambda node: node),
                            traverse_cond_func=(lambda node: True),
                            collect_cond_func=(lambda node: True)):
        res = []
        self._root.recursive_collection(res, func, traverse_cond_func, collect_cond_func)
        return res

    def get_prunable_nodes(self):
        return self.recursive_traversal(collect_cond_func=(lambda node: node.is_leaf()))

    def get_expendable_nodes(self):
        return self.recursive_traversal(collect_cond_func=(lambda node: node.is_expandable()))

    def _get_mean_distance(self):
        if self._total_distance_count == 0:
            return 0
        result = self._total_distance / self._total_distance_count

        return result

    def _get_max_mean_distance(self):
        return 1 / (4 * self.get_current_size())

    def _reset_values(self):
        self.recursive_traversal(func=lambda node: node.reset_value())
        self._total_distance = 0
        self._total_distance_count = 0

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
        # print('nodes to plot:', len(nodes))
        plt.title('tree size={}'.format(len(nodes)))
        plt.plot([0, 1], [0, 0], '|', linewidth=2)
        for node in nodes:
            parent, child = node.get_connection_with_parent()
            r = 0
            b = 0
            # if node.get_value() == 0 and node._parent is not None:

            # if node.is_expandable():
            #     b = 255
            # if node.is_leaf():
            #     r = 255

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

        # if self._dimensions == 1:
        #     # f = 0.1
        #     # hist, x_h = np.histogram(self.get_points().flatten(), bins=int(len(nodes) * f))
        #     # x_h = list((x_h[i] + x_h[i - 1]) / 2 for i in range(1, len(x_h)))
        #     # hist = hist * f / len(nodes)
        #     # max_h = np.max(hist)
        #     # hist = hist / max_h
        #     # plt.plot(x_h, hist,
        #     #          linewidth=1, label='density (max {})'.format(max_h))
        #
        #     v = self.recursive_traversal(func=(lambda node: node.get_value()),
        #                                  collect_cond_func=lambda node: node.is_expandable())
        #     x = sorted(self.recursive_traversal(func=(lambda node: node.get_location()),
        #                                         collect_cond_func=lambda node: node.is_expandable()))
        #     ev = self._expand_threshold_value(1) - .5
        #     max_v = np.max(v)
        #     if max_v != 0:
        #         plt.plot(x, v /
        #                  np.max(v) - 1, label='values (max {})'.format(max_v))
        #         plt.plot([x[0], x[len(x) - 1]], [ev / np.max(v) - 1] * 2,
        #                  label='expansion threshold = {}(f={})'.format(ev + 0.5, self._get_mean_distance() / self._get_max_mean_distance()), linewidth=0.8)

        # plt.legend()
        plt.grid(True)
        plt.xlim(-.1, 1.1)
        if save:
            plt.savefig("{}/a{}.png".format(path, self.SAVE_ID))
            self.SAVE_ID += 1
        else:
            plt.show()
        plt.gcf().clear()

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
    from bin_exp_test import test, test2
    test2()
    # dims = 1
    # tree_size = 127
    # iterations = 100
    # max_size = 20
    #
    # tree = Exploration_tree(dims, tree_size)
    # # tree.plot()
    # samples_size_buffer = np.random.random(iterations) * max_size + 10
    # samples_size_buffer = samples_size_buffer.astype(int)
    #
    # samples = None
    # count = 0
    # for i in samples_size_buffer:
    #     print(count, '----new iteration, searches', i)
    #     count += 1
    #
    #     center = 0.1  # if count < 40 else .7
    #     samples = center + np.abs(np.random.standard_normal((i, dims))) * 0.05
    #     starting_size = tree.get_current_size()
    #     for p in samples:
    #         p = list(p)
    #         tree.search_nearest_node(p)
    #
    #     # ending_size = tree.get_size()
    #     # # print('added', i, 'points(', samples, '): size before-after', starting_size,
    #     # #       '-', ending_size, '({})'.format(ending_size - starting_size))
    #     # if starting_size + i != ending_size:
    #     #     print('ERROR')
    #     #     tree.plot()
    #     #     exit()
    #     tree.plot()
    #
    #     tree.update()
    #     # tree.plot()
    #
    #     # exit()
    # # tree.plot()
