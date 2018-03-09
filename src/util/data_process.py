#!/usr/bin/python3

from data import *
import numpy as np
import matplotlib.pyplot as plt
import math

import sys
sys.path.insert(0, '../')
import action_space


"""
DATA_TEMPLATE = '''
{
    "id":0,
    "agent":{
      "name":"default_name",
      "max_actions":0,
      "k":0,
      "version":0
    },
    "experiment":{
      "name":"no_exp",
      "actions_low":null,
      "actions_high":null,
      "number_of_episodes":0
    },
    "simulation":{
      "episodes":[]
    }

}
'''

EPISODE_TEMPLATE = '''
{
    "id":0,
    "states":[],
    "actions":[],
    "actors_actions":[],
    "ndn_actions":[],
    "rewards":[],
    "action_space_sizes":[]
}
'''
"""


def average_timeline(x):
    res = []
    count = 0
    total = 0
    for i in x:
        total += i
        count += 1
        res.append(total / count)
    return res


def apply_func_to_window(data, window_size, func):
    data_lenght = len(data)
    window_size = min(window_size, data_lenght)
    if window_size == 0:
        window_size = (data_lenght * .1)
    res = []
    for i in range(data_lenght):
        start = int(max(i - window_size / 2, 0))
        end = int(min(i + window_size / 2, data_lenght - 1))
        if start == end:
            continue
        res.append(func(data[start:end]))

    return res


def break_into_batches(data, batches):
    l = len(data)
    batch_size = int(math.ceil(l / (batches)))
    res = []
    for i in range(batches):
        res.append(data[i * batch_size:(i + 1) * batch_size])

    return res


def plot_3d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for x, y, z in points:

        ax.scatter(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


class Data_handler:

    def __init__(self, filename):
        self.data = load(filename)
        self.episodes = self.data.data['simulation']['episodes']

    def get_episode_data(self, field):
        result = []
        for i in self.episodes:
            data = i[field]
            if len(data) == 0:
                continue
            if isinstance(data, list) and isinstance(data[0], list):
                result.extend(data)
            else:
                result.append(data)
        return result

    def get_full_episode_rewards(self):
        rewards = self.get_episode_data('rewards')
        total_rewards = []
        for episode_rewards in rewards:
            total_rewards.append(np.sum(episode_rewards))
        return total_rewards

    def get_adaption_episode(self, reward_threshold=10, window=100):
        rewards = self.get_full_episode_rewards()
        avg = np.array(apply_func_to_window(rewards, window, np.average))

        adaption = np.where(avg > reward_threshold)
        if len(adaption[0]) == 0:
            return 0

        return adaption[0][0]

    def get_min_number_of_actions(self):
        min_size = self.get_episode_data("action_space_sizes")[0][0]
        lenght_of_first_episode = len(self.episodes[0]['actions'])
        return min_size - lenght_of_first_episode

    def get_actions_space_dimensions(self):
        return len(self.data.data['experiment']['actions_low'])

    def create_action_history(self, action_space_check=True):
        before = []
        after = []

        actions = self.data.data['agent']['max_actions']
        init_actions = self.get_min_number_of_actions()
        init_ratio = init_actions / actions
        low = self.data.data['experiment']['actions_low']
        high = self.data.data['experiment']['actions_high']

        space = action_space.Space(low, high, actions, init_ratio)
        tree = space._action_space_module

        sizes = self.get_episode_data("action_space_sizes") if action_space_check else None

        episode_number = 0
        for episode in self.episodes:

            before.append(space.get_space())

            for search_point in episode['actors_actions']:
                space.search_point(search_point, 1)

            # tree.plot()

            after.append(space.get_space())

            size_before_prune = tree.get_size()
            space.update()

            if sizes is not None:
                size_after_prune = tree.get_size()

                expected_sizes = sizes[episode_number]

                print(size_before_prune, '==',
                      expected_sizes[0], ' and ', size_after_prune, '==', expected_sizes[1])
                if size_before_prune != expected_sizes[0] or size_after_prune != expected_sizes[1]:
                    print('Data_process: recreate_action_history: sizes do not match')
                    return None, None

            episode_number += 1
            if episode_number == 30:
                return before, after

        return before, after


# plots

    def plot_rewards(self):

        rewards = self.get_full_episode_rewards()
        # print(rewards)
        episodes = len(rewards)
        batch_size = int(episodes * .01)

        plt.subplot(211)

        total_avg = average_timeline(rewards)
        plt.plot(total_avg, 'm', label='total avg: {}'.format(total_avg[len(total_avg) - 1]))

        avg = apply_func_to_window(rewards, batch_size, np.average)
        plt.plot(avg, 'g', label='batch avg')

        maxima = apply_func_to_window(rewards, batch_size, np.max)
        plt.plot(maxima, 'r', linewidth=1, label='max')

        minima = apply_func_to_window(rewards, batch_size, np.min)
        plt.plot(minima, 'b', linewidth=1, label='min')

        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.legend()
        plt.grid(True)

        plt.subplot(212)

        hist = plt.hist(rewards, facecolor='g', alpha=0.75,  rwidth=0.8)
        max_values = int(hist[0][len(hist[0]) - 1])
        x_max_values = int(hist[1][len(hist[0]) - 1])
        plt.annotate(str(max_values), xy=(x_max_values, int(max_values * 1.1)))

        plt.ylabel("Distribution")
        plt.xlabel("Value")
        plt.yscale("log")

        plt.grid(True)
        plt.show()

    def plot_average_reward(self):
        rewards = self.get_full_episode_rewards()

        window_size = int(len(rewards) * .05)
        w_avg = apply_func_to_window(rewards, window_size, np.average)
        plt.plot(w_avg, 'g--', label='widnowed avg (w_size {})'.format(window_size))

        avg = average_timeline(rewards)
        plt.plot(avg, label='average: {}'.format(avg[len(avg) - 1]))

        adaption_time = self.get_adaption_episode()
        plt.plot(adaption_time, avg[adaption_time], 'bo',
                 label='adaption time: {}'.format(adaption_time))

        avg_ignore_adaption = np.array(average_timeline(rewards[adaption_time:]))
        plt.plot(np.arange(adaption_time, len(rewards)), avg_ignore_adaption,
                 label='average(ignore adaption): {}'.format(avg_ignore_adaption[len(avg_ignore_adaption) - 1]))

        argmax = np.argmax(avg_ignore_adaption)
        # print(argmax, adaption_time, len(avg_ignore_adaption))
        plt.plot(argmax + adaption_time, avg_ignore_adaption[argmax], 'ro',
                 label='max: {}'.format(avg_ignore_adaption[argmax]))

        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_actions(self):
        picked_actions = np.array(self.get_episode_data('actions'))
        actors_actions = np.array(self.get_episode_data('actors_actions'))
        ndn = np.array(self.get_episode_data('ndn_actions'))

        plt.plot(ndn, 'r1', label='Nearest discrete neighbor'.format())
        plt.plot(actors_actions, 'g1', label='Actors actions'.format())
        plt.plot(picked_actions, 'b1', label='Final actions'.format())

        plt.ylabel("Action value")
        plt.xlabel("Episodes")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_action_distribution(self):
        assert len(self.data.data['experiment']['actions_low']
                   ) == 1, 'This function works only for 1-dimensional action space'
        picked_actions = np.array(self.get_episode_data('actions'))
        actors_actions = np.array(self.get_episode_data('actors_actions'))

        picked_actions = picked_actions.flatten()
        actors_actions = actors_actions.flatten()

        plt.hist([picked_actions, actors_actions], bins=100,
                 label=['{} actions'.format(len(picked_actions)),
                        'continuous actions'])

        plt.ylabel("logN")
        plt.xlabel("Action space")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_action_distribution_over_time(self, number_of_batches=5, n_bins=30):
        assert len(self.data.data['experiment']['actions_low']
                   ) == 1, 'This function works only for 1-dimensional action space'
        picked_actions = np.array(self.get_episode_data('actions'))
        batches = break_into_batches(picked_actions, number_of_batches)
        low = self.data.data['experiment']['actions_low'][0]
        high = self.data.data['experiment']['actions_high'][0]
        res = []
        count = 0
        for batch in batches:
            hist, bins = np.histogram(batch, bins=np.linspace(low, high, n_bins))
            count += 1
            plt.plot(bins[1:], hist, linewidth=1, label='t={}%'.format(
                100 * count / number_of_batches))
            # plt.hist(batch, bins=30, histtype='stepfilled', label=str(count))

        plt.ylabel("logN")
        plt.xlabel("Action space")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_action_error(self):
        ndn = np.array(self.get_episode_data('ndn_actions'))
        actors_actions = np.array(self.get_episode_data('actors_actions'))

        error = np.sqrt(np.sum(np.square(ndn - actors_actions), axis=1))  # square error
        # plt.plot(error, label='error')
        print('Ploting actions might take a while: number of actions to plot {}:'.format(len(ndn)))
        w_avg = apply_func_to_window(error, 1000, np.average)
        plt.plot(w_avg, linewidth=1, label='w error')

        avg_error = average_timeline(error)
        plt.plot(avg_error, label='avg_error :{}'.format(
            avg_error[len(avg_error) - 1]))

        avg_number_of_actions = self.data.data['agent']['max_actions']
        mean_expected_error = 1 / (4 * avg_number_of_actions)
        plt.plot([0, len(ndn)], [mean_expected_error] * 2,
                 label='mean expected error={}'.format(mean_expected_error))

        plt.ylabel("Error")
        plt.xlabel("Episode")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_action_space_size(self):
        min_max = self.get_episode_data("action_space_sizes")
        size = np.array(min_max).flatten()

        # grad = np.gradient(size)
        # adaption_point = np.where(grad < 0)[0][0]

        x = np.arange(len(size))
        plt.plot(x, size, '--')

        s_max = average_timeline(apply_func_to_window(size, 0, np.max))
        plt.plot(x, s_max, 'r',
                 label='max {}'.format(s_max[len(s_max) - 1]))
        avg = average_timeline(size)
        plt.plot(x, avg, 'g',
                 label='avg {}'.format(avg[len(avg) - 1]))

        s_min = average_timeline(apply_func_to_window(size, 0, np.min))
        plt.plot(x, s_min, 'b',
                 label='min {}'.format(s_min[len(s_min) - 1]))

        plt.ylabel("Size")
        plt.xlabel("Episode")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    dh = Data_handler('data_10000_Wolp4_Inv127k12#0.json')
    # dh = Data_handler('data_10000_agen4_exp1000k10#0.json.zip')
    # dh = Data_handler('saved/data_2500_Wolp4_Inv7k1#0.json.zip')
    print("loaded")

    # dh.plot_rewards()
    # dh.plot_average_reward()
    # dh.plot_actions()
    # dh.plot_action_distribution()
    # dh.plot_action_distribution_over_time()
    # dh.plot_action_error()

    print(dh.create_action_history())
