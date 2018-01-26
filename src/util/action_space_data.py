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

    plot_lenghts(monitor)
