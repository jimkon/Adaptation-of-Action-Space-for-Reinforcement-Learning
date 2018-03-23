#!/usr/bin/python3
import numpy as np
import data
import data_process
import action_space
import matplotlib.pyplot as plt


class History:

    def __init__(self, data_path, check_data=True):
        data_p = data_process.Data_handler(data_path)
        data = data_p.data

        self._check_data = check_data

        self._k = data.data['agent']['k']
        self._actions_low = data.data['experiment']['actions_low']
        self._actions_high = data.data['experiment']['actions_high']
        self._episodes = sorted(data.data['simulation']['episodes'], key=lambda k: k['id'])
        self._action_space_size = data.data['agent']['max_actions']

        # min_size = data_p.get_episode_data("action_space_sizes")[0][0]
        # lenght_of_first_episode = len(self._episodes[0]['actions'])
        # min_size -= lenght_of_first_episode

        self._action_space = action_space.Space(self._actions_low,
                                                self._actions_high,
                                                self._action_space_size)
        # , min_size_ratio = min_size / self._action_space_size)

        self._current_episode = 0
        self._current_step = 0
        self._current_total_steps = 0
        self._current_total_reward = 0
        self._current_episode_ended = False

        self.data_p = data_p
        self.data = data

        if self._check_data and data.data['experiment']['number_of_episodes'] != self.total_episodes():
            print('Difference on number of episodes')

    def current_state(self):
        return self._episodes[self._current_episode]['states'][self._current_step]

    def current_action(self):
        return self._episodes[self._current_episode]['actions'][self._current_step]

    def current_actors_action(self):
        return self._episodes[self._current_episode]['actors_actions'][self._current_step]

    def current_knns(self):
        return self._knns

    def current_reward(self):
        return self._episodes[self._current_episode]['rewards'][self._current_step]

    def current_total_reward(self):
        return self._current_total_reward

    def current_totatl_steps(self):
        return self._current_total_steps

    def current_action_space(self):
        return self._action_space.get_space()

    def current_episode_lenght(self):
        return len(self._episodes[self._current_episode]['rewards'])

    def is_end_of_episode(self):
        return self._current_episode_ended

    def total_episodes(self):
        return len(self.data_p.get_episode_data('id'))

    def is_end(self):
        return self._current_episode == self.total_episodes() - 1 and self.is_end_of_episode()

    def next_step(self):

        if self.is_end_of_episode():
            return False

        knns, indexes = self._action_space.search_point(self.current_actors_action(), self._k)

        self._knns = knns
        self._current_total_reward += self.current_reward()
        if self._check_data:
            self._check_step()

        # self.print_step_info()

        if self._current_step == self.current_episode_lenght() - 1:
            self._current_episode_ended = True
        else:
            self._current_step += 1
            self._current_total_steps += 1

        return not self.is_end_of_episode()

    def end_episode(self):
        while self.next_step():
            continue

    def next_episode(self):
        self.end_episode()

        if self.is_end():
            return

        self._current_episode += 1
        self._current_step = 0
        self._current_total_reward = 0
        self._current_episode_ended = False
        # self.plot_current_action_space()

        self._action_space.update()

    def go_to_episode(self, ep, collect_data_func=None):
        res = []
        while self._current_episode < ep and not self.is_end():
            if collect_data_func is not None:
                res.append(collect_data_func(self))

            self.next_episode()

        return res

    def _check_step(self):
        # if self.current_action() not in self._knns:
        #     print('Action picked not in k-nns: Episode',
        #           self._current_episode, 'Step', self._current_step)
        #     # self.print_current_episode()
        #     print(self.current_action())
        #     print(self._knns)
        #     print(self.current_action_space())
        #     self.plot_current_action_space()
        #
        #     exit()
        return True

    def print_current_episode(self):
        import json
        print(json.dumps(self._episodes[self._current_episode], indent=2, sort_keys=True))

    def print_step_info(self):
        print('Episode: {}, Step: {}/{}, End={}'.format(self._current_episode,
                                                        self._current_step,
                                                        self.current_episode_lenght() - 1,
                                                        self.is_end_of_episode()))

    def plot_current_action_space(self):
        self._action_space._action_space_module.plot()

    def plot_current_action_space_distr(self):
        space = self.current_action_space()
        plt.hist(space, bins=100, histtype='step')

        # actions_till_now = []
        # last_actions = []
        #
        # last_prune_i = np.where(np.array(h.data_p.get_prune_episodes()) < self._current_episode)[0]
        # last_prune = (h.data_p.get_prune_episodes()[last_prune_i[len(
        #     last_prune_i) - 1]] + 1) if len(last_prune_i) > 0 else 0
        # print(last_prune)
        # for ep in self._episodes:
        #     if ep['id'] > self._current_episode:
        #         break
        #     actions_till_now.extend((self._action_space._import_point(action)
        #                              for action in ep['actors_actions']))
        #     if ep['id'] >= last_prune:
        #         last_actions.extend((self._action_space._import_point(action)
        #                              for action in ep['actors_actions']))
        #
        # last_actions = np.array(last_actions).flatten()
        # plt.hist(last_actions, bins=100, histtype='step')
        #
        # actions_till_now = np.array(actions_till_now).flatten()
        # plt.hist(actions_till_now, bins=100, histtype='step')

        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # dh = Data_handler('data_10000_Wolp4_Inv127k12#0.json')
    # dh = Data_handler('data_10000_Wolp4_Inv1000k51#0.json.zip')
    # dh = Data_handler('data_10000_Wolp4_Inv255k25#0.json.zip')
    h = History('data_2000_Wolp4_Inv2047k204#0.json', check_data=False)
    # h = History('data_100_Wolp4_Inv127k12#0.json.zip', check_data=True)
    print("loaded")
    rewards = h.data_p.get_full_episode_rewards()
    print(rewards)
    prune_episodes = h.data_p.get_prune_episodes()
    count = 0
    for ep in prune_episodes:
        h.go_to_episode(ep)
        h.end_episode()
        # h.plot_current_action_space()
        h.plot_current_action_space_distr()

    h.plot_current_action_space()
