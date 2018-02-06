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


def plot_lenghts(mon):
    lines = []

    lenghts = mon.get_data('lenght')
    x = np.arange(len(lenghts))

    scores = mon.get_total_scores()

    lines.append(Line(x, lenghts, line_color='r', text='Population'))
    lines.append(Line(x, scores, line_color='g', text='total score/episode'))
    lines.append(Line(x, mon.get_number_of_unique_actions_used(),
                      line_color='b', text='unique actions/episode'))
#
    # plot_space_adaption(monitor, 1)
    plot_lines(lines)


def plot_space_adaption_history(mon):
    lenghts = mon.get_data('lenght')

    for i in range(1, len(lenghts)):
        plot_space_adaption(mon, i)


def plot_space_adaption(mon, ep):
    ep = max(ep, 1)
    space = mon.break_into_episodes('space')[ep - 1:ep + 1]
    actions = mon.break_into_episodes('actors_action')[ep - 1:ep + 1]
    score = mon.break_into_episodes('usage')[ep - 1:ep + 1]
    total_actions = [len(actions[0]), len(actions[1])]

    score = score[0] / total_actions[0]

    max_height = np.max(score)

    for i in range(len(space[0])):
        plt.plot([space[0][i], space[0][i]], [0, score[i]], 'b')

    plt.plot(space[0], -.05 * max_height * np.ones(len(space[0])), 'b1',
             label='{} actions ep={} ({} steps)'.format(
             len(space[0]), ep - 1, int(total_actions[0])))
    plt.plot(actions[0], -.1 * max_height * np.ones(len(actions[0])), 'g2',
             label='{} actions ep={} '.format(
             int(len(actions[0])), ep - 1))
    plt.plot(space[1], -.15 * max_height * np.ones(len(space[1])), 'r2',
             label='{} actions ep={} ({} steps)'.format(
             len(space[1]), ep, int(total_actions[1])))

    plt.grid()
    plt.legend()
    plt.show()


def plot_space_and_actions_across_episodes(mon):
    lines = []
    print('processing data')
    spaces = mon.break_into_episodes('space')
    scores = np.array(mon.break_into_episodes('usage'))
    actions = mon.break_into_episodes('actors_action')
    max_scores = []
    for s in scores:
        if len(s) > 0:
            max_scores.append(np.max(s))
        else:
            max_scores.append(-1)

    episodes = len(spaces)
    print('start plotting data')
    x = np.arange(episodes)
    # plotting actions
    for i in x:
        ep = actions[i]
        for j in range(len(ep)):
            # f = scores[i][j] / max_scores[i]
            # color = '#{:02x}0000'.format(int(255 * f))
            lines.append(Point(i, ep[j],
                               line_color='#00ff00', marker='1'))
    # plotting spaces
    for i in x:
        space = spaces[i]
        pts = []
        dtype = [('height', float), ('score', float)]
        for j in range(len(space)):
            f = scores[i][j] / max_scores[i]
            pts.append((space[j], f))

        pts = np.array(pts, dtype=dtype)
        pts = np.sort(pts, order='score')

        for pt in pts:
            f = pt['score']
            color = '#{:02x}0000'.format(int(255 * (f ** 2)))
            f += .7
            lines.append(Point(i, pt['height'],
                               line_color=color, line_width=f, marker='1'))

    plot_lines(lines, labels=False)


def plot_action_density(mon):
    if 'densities' not in mon.get_keys():
        print('computing action space density')
        mon.compute_densities()
        mon.save()

    data = mon.get_data('densities')
    n = int(data[0])
    offset = 1 / n
    count = 1
    data = data[count:]
    # print(data)
    lines = []
    x_axis = range(int(len(data) / n))
    for j in range(n):
        lines.append(Line([x_axis[0], x_axis[len(x_axis) - 1]],
                          [j * offset, j * offset], line_color='#a0a0a0'))
        temp = []
        for x in x_axis:
            v = data[j + x * n]
            temp.append(v)
        temp = np.array(temp)
        avg = np.average(temp)
        color = '#{:02x}00{:02x}'.format(int(avg * 255), int(avg * 125))

        temp = (temp + j) * offset
        lines.append(Line(x_axis, temp, line_color=color))
        # text='actions%({}-{})'.format(j * offset, (j + 1) * offset)))

    lines.append(Line([x_axis[0], x_axis[len(x_axis) - 1]],
                      [n * offset, n * offset], line_color='#a0a0a0'))

    plot_lines(lines, labels=False, grid_flag=False)


class Action_space_data(Data):

    def compute_densities(self, n=9):
        spaces = self.break_into_episodes('space')

        self.add_array('densities')
        self.add_to_array('densities', n)

        for ep in spaces:
            density = np.zeros(n)
            for action in ep:
                index = min(int(action * n), n - 1)
                density[index] += 1
            density /= len(ep)
            self.add_to_array('densities', density)

    def break_into_episodes(self, field):
        data = self.get_data(field)
        lenghts = self.get_data('lenght')
        if field == 'actors_action':
            usage = self.break_into_episodes('usage')
            lenghts = []
            for ep in usage:
                lenghts.append(np.sum(ep))

        batches = []
        count = 0
        for l in lenghts:
            # try:
            batches.append(data[count:count + int(l)])
            count += int(l)
            # except Exception as e:
            # print(e)
            # break
        return batches

    def get_total_scores(self):
        scores = self.break_into_episodes('usage')
        totals = []
        i = 0
        for s in scores:
            totals.append(np.sum(s))
            i += 1

        return totals

    def get_number_of_unique_actions_used(self):
        scores = self.break_into_episodes('usage')
        totals = []
        i = 0
        for s in scores:
            totals.append(len(np.where(s > 0)[0]))
            i += 1

        return totals


if __name__ == '__main__':
    monitor = Action_space_data('action_space_5000_101')
    monitor.load()
    # monitor.print_data()

    # plot_action_density(monitor)
    # plot_space_adaption_history(monitor)
    plot_lenghts(monitor)
    # plot_space_and_actions_across_episodes(monitor)

    # print(monitor.break_into_episodes('space'))
