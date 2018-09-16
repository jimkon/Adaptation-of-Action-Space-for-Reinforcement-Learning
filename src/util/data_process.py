from data import *
import numpy as np
import matplotlib.pyplot as plt
import math

import sys
sys.path.insert(0, '../')
# print(sys.path)
# import action_space


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
        self.filename = filename
        self.data = load(filename)
        self.episodes = self.data.data['simulation']['episodes']

    def get_episode_data(self, field):
        result = []
        for i in self.episodes:
            data = i[field]

            if isinstance(data, list) and len(data) == 0:
                continue
            if isinstance(data, list) and isinstance(data[0], list):
                result.extend(data)
            else:
                result.append(data)

        # if len(np.array(result).shape) < 2:
        #     print('ERROR getting episode data:', field)
        #     exit()
        return result

    def get_full_episode_rewards(self):
        rewards = self.get_episode_data('rewards')
        total_rewards = []
        for episode_rewards in rewards:
            total_rewards.append(np.sum(episode_rewards))
        return total_rewards

    def get_adaption_episode(self, reward_threshold=15, window=100):
        rewards = self.get_full_episode_rewards()
        window_size = int(len(rewards) * .05)
        w_avg = np.array(apply_func_to_window(rewards, window_size, np.average))

        adaption = np.where(w_avg > reward_threshold)
        if len(adaption[0]) == 0:
            return 0

        return adaption[0][0]

    def get_min_number_of_actions(self):
        min_size = self.get_episode_data("action_space_sizes")[0][0]
        lenght_of_first_episode = len(self.episodes[0]['actions'])
        return min_size - lenght_of_first_episode

    def get_actions_space_dimensions(self):
        return len(self.data.data['experiment']['actions_low'])

    def get_average_action_space_size(self):
        sizes = self.get_episode_data("action_space_sizes")
        return np.average(sizes)

    def get_prune_episodes(self):
        sizes = np.array(self.get_episode_data("action_space_sizes")).flatten()
        size_1 = sizes[1:]
        size_2 = sizes[:len(sizes) - 1]
        return np.where(size_1 < size_2)[0] / 2 - 1


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

    def plot_sensitivity_efficiency(self):
        rewards = np.array(self.get_full_episode_rewards()) / 200

        # rewards = []
        # temp = np.array(self.get_episode_data('rewards')).flatten()
        # for r in temp:
        #     rewards.extend(r)
        # rewards = np.array(rewards)

        argsort = np.argsort(rewards)
        rewards = rewards[argsort]

        plt.subplot(311)

        density = [[0, 0]]
        for i in np.arange(0.1, 1, 0.1):
            index = int(i * len(rewards))
            plt.plot([0, 1], [rewards[index]] * 2, 'r', linewidth=0.5)
            density.append([i, rewards[index]])
        density.append([1, rewards[len(rewards) - 1]])

        plt.plot(np.linspace(0, 1, len(rewards)), rewards, label='sorted')

        # plt.plot(list(i[0] for i in density), list(i[1] for i in density), 'ro')
        t_adaption = self.get_adaption_episode() / len(rewards)
        plt.plot([t_adaption] * 2,
                 [0, rewards[len(rewards) - 1]], '--', label='adaption at {}'.format(t_adaption))

        avg = np.average(rewards)
        plt.plot([0, 1], [avg] * 2, '--', label='average {}'.format(avg))

        plt.xticks(list(i[0] for i in density))
        plt.yticks(list(i[1] for i in density))
        plt.ylabel("Reward")
        plt.xlabel("Episodes")
        plt.legend()
        plt.grid(True)

        plt.subplot(312)
        rewards = []
        temp = np.array(self.get_episode_data('rewards')).flatten()
        # for r in temp:
        #     rewards.extend(r)
        rewards = temp
        rewards = np.array(rewards)
        plt.hist(rewards, label='hist')

        avg = np.average(rewards)
        plt.plot(avg, 0, 'o', label='average {}'.format(avg))
        plt.legend()
        plt.grid(True)
        plt.ylabel("Steos")
        plt.xlabel("Reward")

        plt.subplot(313)
        thetas = np.array(list(i[2] for i in np.array(self.get_episode_data('states'))))
        thetas = np.abs(thetas * (360 / (2 * np.pi)))
        plt.hist(thetas, label='deg histogram', bins=400)
        avg_th = np.average(thetas)
        plt.plot(avg_th, 0, 'o', label='avg={:.03f}'.format(avg_th))

        plt.legend()
        plt.grid(True)
        # plt.xscale('log')
        plt.ylabel("Steps")
        plt.xlabel("Degrees")
        plt.tight_layout()
        plt.show()

    def plot_actions(self):
        picked_actions = np.array(self.get_episode_data('actions'))
        actors_actions = np.array(self.get_episode_data('actors_actions'))
        ndn = np.array(self.get_episode_data('ndn_actions'))

        plt.plot(ndn, 'r1', label='Nearest discrete neighbor'.format())
        plt.plot(actors_actions, 'g2', label='Actors actions'.format())
        plt.plot(picked_actions, 'b3', label='Final actions'.format())

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

    def plot_action_distribution_over_time(self, number_of_batches=4, n_bins=30):
        assert len(self.data.data['experiment']['actions_low']
                   ) == 1, 'This function works only for 1-dimensional action space'
        picked_actions = np.array(self.get_episode_data('actions'))
        batches = break_into_batches(picked_actions, number_of_batches)
        low = self.data.data['experiment']['actions_low'][0]
        high = self.data.data['experiment']['actions_high'][0]
        res = []
        count = 0
        for batch in batches:
            hist, bins = np.histogram(batch, bins=np.linspace(low, high, n_bins + 1))
            count += 1
            plt.plot(np.linspace(bins[0], bins[len(bins) - 1], n_bins), hist, linewidth=1, label='t={}%'.format(
                100 * count / number_of_batches))
            # plt.hist(batch, bins=30, histtype='stepfilled', label=str(count))

        plt.ylabel("N")
        plt.xlabel("Action space")
        # plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_discretization_error(self):
        ndn = np.array(self.get_episode_data('ndn_actions'))
        actors_actions = np.array(self.get_episode_data('actors_actions'))

        error = np.sqrt(np.sum(np.square(ndn - actors_actions), axis=1))  # square error
        # plt.plot(error, label='error')
        print('Ploting actions might take a while: number of actions to plot {}:'.format(len(ndn)))
        w_avg = apply_func_to_window(error, int(self.get_average_action_space_size()), np.average)
        plt.plot(w_avg, linewidth=1, label='running avg error(window {})'.format(
            int(self.get_average_action_space_size())))

        avg_error = np.average(error)
        plt.plot([0, len(ndn)], [avg_error] * 2,
                 label='avg error={}'.format(avg_error))

        avg_number_of_actions = self.get_average_action_space_size()
        mean_expected_error = 1 / (4 * avg_number_of_actions)
        plt.plot([0, len(ndn)], [mean_expected_error] * 2,
                 label='mean expected error={}'.format(mean_expected_error))

        plt.ylabel("Error")
        plt.xlabel("Episode")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_discretization_error_distribution(self):
        ndn = np.array(self.get_episode_data('ndn_actions'))
        actors_actions = np.array(self.get_episode_data('actors_actions'))

        inv_idexes = np.where(actors_actions > 3)[0]
        np.put(actors_actions, inv_idexes, np.ones(len(inv_idexes)) * 3)

        inv_idexes = np.where(actors_actions < -3)[0]
        np.put(actors_actions, inv_idexes, np.ones(len(inv_idexes)) * -3)

        size = len(actors_actions)

        error = np.sqrt(np.sum(np.square(ndn - actors_actions), axis=1))  # square error

        sorting_indexes = np.argsort(actors_actions, axis=0)

        sorted_actions = np.reshape(actors_actions[sorting_indexes], actors_actions.shape)

        error = error[sorting_indexes]
        w_error = apply_func_to_window(error, int(size * .01), np.average)

        action_distr, _ = np.histogram(actors_actions, bins=1000)
        action_distr = action_distr / size
        plt.plot(np.linspace(sorted_actions[0], sorted_actions[size - 1], 1000),
                 action_distr, linewidth=0.5, label='action usage distr')

        plt.plot(sorted_actions, w_error, label='Error distribution')

        weighted_error = np.copy(error)
        i = 0
        min_a = np.min(actors_actions)
        max_a = np.max(actors_actions)

        w_i = np.interp(weighted_error, [min_a, max_a], [0, len(action_distr) - 1]).astype(int)
        weighted_error = np.multiply(weighted_error, action_distr[w_i])
        weighted_error = apply_func_to_window(weighted_error, int(size * 0.01), np.average)
        plt.plot(sorted_actions, weighted_error, linewidth=1, label='weighted error distr')

        # argmin = np.argmin(w_error)
        # plt.plot(sorted_actions[argmin], w_error[argmin],
        #          'bo', linewidth=1, label='min={}'.format(w_error[argmin]))

        avg = np.average(error)
        plt.plot([sorted_actions[0], sorted_actions[size - 1]],
                 [avg, avg], label='avg={}'.format(avg))

        avg_number_of_actions = self.get_average_action_space_size()
        mean_expected_error = 1 / (4 * avg_number_of_actions)
        plt.plot([sorted_actions[0], sorted_actions[size - 1]],
                 [mean_expected_error, mean_expected_error], linewidth=1,
                 label='mean expected error={}'.format(mean_expected_error))

        plt.ylabel("Error")
        plt.xlabel("Space")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_actor_critic_error(self):
        actions = np.array(self.get_episode_data('actions'))
        actors_actions = np.array(self.get_episode_data('actors_actions'))

        error = np.sqrt(np.sum(np.square(actions - actors_actions), axis=1))  # square error
        # plt.plot(error, label='error')
        print('Ploting actions might take a while: number of actions to plot {}:'.format(len(actions)))
        w_avg = apply_func_to_window(error, 1000, np.average)
        plt.plot(w_avg, linewidth=1, label='w error')

        avg_error = average_timeline(error)
        plt.plot(avg_error, label='avg_error :{}'.format(
            avg_error[len(avg_error) - 1]))

        plt.ylabel("Error")
        plt.xlabel("Episode")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_action_space_size(self):
        min_max = self.get_episode_data("action_space_sizes")
        size = np.array(min_max).flatten()

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

    def plot_action_space_timeline(self):
        h = History(self.filename)
        prune_episodes = h.data_p.get_prune_episodes()
        print(prune_episodes)

        all_actions = np.array(self.get_episode_data('actors_actions')).flatten()
        prev_step_for_actions = 0
        average_action_space = []
        previous_action_space = None
        for ep in prune_episodes:
            actions = h.go_to_episode(ep)
            h.end_episode()
            h.print_step_info()

            cur_action_space = h.current_action_space()

            # average_action_space.extend(np.array(cur_action_space).flatten())
            # plt.hist(average_action_space, bins=1000, histtype='step', label='average')

            plt.hist(cur_action_space,
                     bins=1000,
                     linewidth=0.5,
                     histtype='step',
                     label='current space')

            actions = []
            for action in all_actions[prev_step_for_actions:h._current_total_steps - 1]:
                actions.extend(h._action_space._import_point(action))
            prev_step_for_actions = h._current_total_steps - 1
            plt.hist(actions,
                     bins=1000,
                     histtype='step',
                     label='actions')

            if previous_action_space is not None:
                plt.hist(previous_action_space,
                         bins=1000,
                         histtype='step',
                         label='prev space')
            previous_action_space = cur_action_space

            plt.grid(True)
            plt.legend()
            plt.show()


if __name__ == "__main__":

    dh = Data_handler(
        "D:\dip\Adaptation-of-Action-Space-for-Reinforcement-Learning\\results\Wolp5\InvertedPendulum-v2\data\def\data_1000_Wolp4_Inv1000k100#0.json.zip")
    # dh = Data_handler('data_10000_Wolp4_Inv1000k51#0.json.zip')
    # dh = Data_handler('data_10000_Wolp4_Inv255k25#0.json.zip')
    # # dh = Data_handler('data_5000_Wolp4_Inv10000k1000#0.json.zip')
    # dh = Data_handler('data_2000_Wolp4_Inv511k51#0.json.zip')
    # # dh = Data_handler('data_100_Wolp4_Inv127k12#0.json.zip')
    # print("loaded")

    dh.plot_rewards()
    # dh.plot_average_reward()
    # dh.plot_actions()
    # dh.plot_action_distribution()
    # dh.plot_action_distribution_over_time()
    # dh.plot_discretization_error()
    # dh.plot_actor_critic_error()
    # dh.plot_discretization_error_distribution()
    # dh.plot_action_space_timeline()
    # print(dh.get_prune_episodes())

    # dh.plot_action_space_size()
    # b, a = dh.create_action_history()
    # print(len(b), len(a))
