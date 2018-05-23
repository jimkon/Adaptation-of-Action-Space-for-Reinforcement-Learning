#!/usr/bin/python3
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


class Monitor():

    STATE_COL = 1
    ACTION_COL = 2
    REWARD_COL = 3

    def __init__(self, size, state_size, action_size, refresh_rate=0, action_limits=None):

        self.size = size
        self.refresh_rate = refresh_rate

        self.t = 0
        self.t_offset = 0
        self.last_repaint = self.t_offset

        self.states = []  # list(np.zeros(size * state_size).reshape(size, state_size))
        self.actions = []  # list(np.zeros(size * action_size).reshape(size, action_size))

        self.rewards = []  # list(np.zeros(size))
        self.rewards_avg = []  # list(np.zeros(size))
        self.total_rewards = []

        self.action_size = action_size
        self.state_size = state_size

        if action_limits is not None:
            self.action_low = np.ones(2) * action_limits[0]
            self.action_high = np.ones(2) * action_limits[1]
        else:
            self.action_low = None
            self.action_high = None

        self.episodes = [0]

        plt.ion()
        self.fig = plt.figure()
        self.state_ax = self.fig.add_subplot(4, 1, 1)
        self.action_ax = self.fig.add_subplot(4, 1, 2)

        self.reward_ax = self.fig.add_subplot(4, 1, 3)
        # self.line,  = self.reward_ax.plot(np.random.uniform(0, 1, size=size))
        self.reward_avg_ax = self.fig.add_subplot(4, 1, 4)

        # an = anim.FuncAnimation(self.fig, self.repaint, interval=10)
        plt.legend()
        plt.show()

    def repaint(self):
        if self.t - self.last_repaint < self.refresh_rate:
            return
        self.last_repaint = self.t

        x = np.arange(self.t_offset, self.t)

        self.state_ax.clear()
        for j in range(self.state_size):
            y = list(a[j] for a in self.states)
            self.state_ax.plot(x, y, label='{}'.format(j), linewidth=0.7)

        self.episodes = [ep for ep in self.episodes if ep >= self.t_offset]
        self.state_ax.plot(self.episodes, np.zeros(len(self.episodes)), 'ro')

        self.state_ax.set_title('states')
        if self.state_size <= 4:
            self.state_ax.legend()

        ####
        self.action_ax.clear()
        for j in range(self.action_size):
            y = list(a[j] for a in self.actions)
            self.action_ax.plot(x, y, label='{}'.format(j), linewidth=0.7)

        if self.action_low is not None:
            t_x = [x[0], x[len(x) - 1]]
            self.action_ax.plot(t_x, self.action_low, 'r', linewidth=0.7)
            self.action_ax.plot(t_x, self.action_high, 'r', linewidth=0.7)

        self.action_ax.set_title('actions')
        if self.action_size <= 4:
            self.action_ax.legend()

        ###
        self.reward_ax.clear()

        y = self.rewards
        self.reward_ax.plot(x, y, linewidth=1)
        self.reward_ax.set_title('reward')

        ####
        self.reward_avg_ax.clear()

        y = self.rewards_avg
        self.reward_avg_ax.plot(x, y, linewidth=1, label='running average')
        y = np.divide(self.total_rewards, x + 1)
        self.reward_avg_ax.plot(x, y, label='total average')
        self.reward_avg_ax.legend()
        self.reward_avg_ax.set_title('reward history')

        # self.fig.draw()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

    def add_data(self, state, action, reward):

        self.t += 1
        if self.t - self.t_offset > self.size:
            self.t_offset = self.t - self.size

        self.states.append(state)
        if len(self.states) > self.size:
            self.states.pop(0)

        self.actions.append(action)
        if len(self.actions) > self.size:
            self.actions.pop(0)

        self.rewards.append(reward)
        if len(self.rewards) > self.size:
            self.rewards.pop(0)

        self.rewards_avg.append(np.average(self.rewards))
        if len(self.rewards_avg) > self.size:
            self.rewards_avg.pop(0)

        if len(self.total_rewards) == 0:
            self.total_rewards.append(reward)
        else:
            self.total_rewards.append(self.total_rewards[len(self.total_rewards) - 1] + reward)

        if len(self.total_rewards) > self.size:
            self.total_rewards.pop(0)

        # print(self.states)
        # print(self.actions)
        # print(self.rewards)
        # exit()
        # self.repaint(0)

    def end_of_episode(self):
        self.episodes.append(self.t)


if __name__ == '__main__':
    mon = Monitor(100, 2, 1)
    mon.start()
    while True:
        mon.add_data([0.1, 5], [1], 2)
        print('asdasd')
        # mon.repaint()
