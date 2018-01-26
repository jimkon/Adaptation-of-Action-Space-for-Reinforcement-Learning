#!/usr/bin/python3
import numpy as np
from my_plotlib import *
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

    plot_lines(lines)


def plot_space_across_episodes(mon):
    lines = []
    print('processing data')
    spaces = mon.break_into_episodes('space')
    scores = np.array(mon.break_into_episodes('usage'))
    max_scores = []
    for s in scores:
        if len(s) > 0:
            max_scores.append(np.max(s))
        else:
            max_scores.append(-1)

    episodes = len(spaces)
    print('start plotting data')
    x = np.arange(episodes)
    for i in x:
        space = spaces[i]
        for j in range(len(space)):
            f = scores[i][j] / max_scores[i]
            color = '#{:02x}0000'.format(int(255 * f * f))
            f += .7
            lines.append(Point(i, space[j],
                               line_color=color, line_width=f))

    plot_lines(lines, labels=False)


def plot_actors_actions_across_episodes(mon):
    lines = []
    print('processing data')
    actions = mon.break_into_episodes('actors_action')

    episodes = len(actions)
    print('start plotting data')
    x = np.arange(episodes)
    for i in x:
        ep = actions[i]
        for j in range(len(ep)):
            # f = scores[i][j] / max_scores[i]
            # color = '#{:02x}0000'.format(int(255 * f))
            lines.append(Point(i, ep[j],
                               line_color='#00ff00'))

    plot_lines(lines, labels=False)


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
                               line_color='#00ff00'))
    # plotting spaces
    for i in x:
        space = spaces[i]
        for j in range(len(space)):
            f = scores[i][j] / max_scores[i]
            color = '#{:02x}0000'.format(int(255 * f * f))
            f += .7
            lines.append(Point(i, space[j],
                               line_color=color, line_width=f))

    plot_lines(lines, labels=False)


class Action_space_data(Data):

    def break_into_episodes(self, field):
        data = self.get_data(field)
        lenghts = self.get_data('lenght')

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


if __name__ == '__main__':
    monitor = Action_space_data('temp')
    monitor.load()
    monitor.print_data()

    # plot_lenghts(monitor)
    # plot_space_across_episodes(monitor)
    # plot_actors_actions_across_episodes(monitor)
    plot_space_and_actions_across_episodes(monitor)

    # print(monitor.break_into_episodes('space'))
