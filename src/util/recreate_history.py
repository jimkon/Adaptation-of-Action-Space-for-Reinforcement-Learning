#!/usr/bin/python3
import numpy as np
import data
import data_process
import action_space


class History:

    def __init__(self, data_path, check_data=True):
        data_p = data_process.Data_handler(data_path)
        data = data_p.data

        self._check_data = check_data

        self._k = data.data['agent']['k']
        self._actions_low = data.data['experiment']['actions_low']
        self._actions_high = data.data['experiment']['actions_high']
        self._episodes = data.data['simulation']['episodes']
        self._action_space_size = data.data['agent']['max_actions']

        min_size = data_p.get_episode_data("action_space_sizes")[0][0]
        lenght_of_first_episode = len(self._episodes[0]['actions'])
        min_size -= lenght_of_first_episode

        self._action_space = action_space.Space(self._actions_low,
                                                self._actions_high,
                                                self._action_space_size,
                                                min_size_ratio=min_size / self._action_space_size)

        self._current_episode = 0
        self._current_step = -1
        self._total_episodes = len(self._episodes)
        if self._check_data and self._total_episodes != self.total_episodes():
            print('Difference on number of episodes')
        self.next_step()

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

    def current_action_space(self):
        return self._action_space.get_space()

    def current_episode_lenght(self):
        return len(self._episodes[self._current_episode]['rewards'])

    def is_end_of_episode(self):
        return self._current_step == self.current_episode_lenght()

    def total_episodes(self):
        return len(self._episodes['id'])

    def is_end(self):
        return self._current_episode == self.total_episodes()

    def next_step(self):
        done = False
        if self.is_end():
            return True
        self._current_step += 1
        if self.is_end_of_episode():
            self._action_space.update()
            self._current_step = 0
            self._current_episode += 1
            done = True

        knns, indexes = self._action_space.search_point(self.current_actors_action(), self._k)
        if self._check_data and self.current_action() not in knns:
            print('Action picked not in k-nns')
        self._knns = knns

        return done

    def next_episode(self):
        if self.is_end():
            return
        while not self.next_step():
            continue

    def go_to_episode(self, ep):
        while self._current_episode < ep:
            self.next_episode()


    # def create_action_history(self, action_space_check=False):
    #     before = []
    #     after = []
    #
    #     actions = self.data.data['agent']['max_actions']
    #     init_actions = self.get_min_number_of_actions()
    #     init_ratio = init_actions / actions
    #     low = self.data.data['experiment']['actions_low']
    #     high = self.data.data['experiment']['actions_high']
    #
    #     space = action_space.Space(low, high, actions, init_ratio)
    #     tree = space._action_space_module
    #
    #     sizes = self.get_episode_data("action_space_sizes")
    #
    #     episode_number = 0
    #     for episode in self.episodes:
    #
    #         before.append(space.get_space())
    #         print(len(space.get_space()))
    #
    #         for search_point in episode['actors_actions']:
    #             space.search_point(search_point, 1)
    #         print('added points', len(episode['actors_actions']), episode['actors_actions'])
    #
    #         # tree.plot()
    #
    #         after.append(space.get_space())
    #
    #         size_before_prune = tree.get_size()
    #         space.update()
    #         print(len(space.get_space()))
    #
    #         size_after_prune = tree.get_size()
    #         expected_sizes = sizes[episode_number]
    #
    #         print('Data_process: recreate_action_history: sizes do not match => episode',
    #               episode_number, end=', ')
    #         print(size_before_prune, '==',
    #               expected_sizes[0], ' and ', size_after_prune, '==', expected_sizes[1])
    #
    #         if size_before_prune != expected_sizes[0] or size_after_prune != expected_sizes[1]:
    #             # print('Data_process: recreate_action_history: sizes do not match => episode',
    #             #       episode_number, end=', ')
    #             # print(size_before_prune, '==',
    #             #       expected_sizes[0], ' and ', size_after_prune, '==', expected_sizes[1])
    #             #
    #             # print('added points', len(episode['actors_actions']))
    #             exit()
    #             if action_space_check:
    #                 return None, None
    #
    #         episode_number += 1
    #
    #     return before, after
if __name__ == "__main__":
    # dh = Data_handler('data_10000_Wolp4_Inv127k12#0.json')
    # dh = Data_handler('data_10000_Wolp4_Inv1000k51#0.json.zip')
    # dh = Data_handler('data_10000_Wolp4_Inv255k25#0.json.zip')
    h = History('data_2000_Wolp4_Inv2047k204#0.json.zip')
    print("loaded")
    print(h.current_state())
    print(h.current_action())
    print(h.current_actors_action())
    print(h.current_knns())
    print(h.current_reward())
