import numpy as np
import pyflann
from gym.spaces import Box
from ddpg import agent
import action_space

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class WolpertingerAgent(agent.DDPGAgent):

    def __init__(self, env, max_actions=1e5, k_ratio=0.1, training_flag=True,
                 action_space_config=['auto', 'square', 10000, 10]):

        self.action_space_config = action_space_config

        super().__init__(env, training_flag=training_flag)

        assert self.continious_action_space, "this version doesn't work for discrete actions spaces"

        self.action_space = action_space.Space(self.low, self.high, max_actions,
                                               adaptation=action_space_config[0], arg1=action_space_config[1],
                                               arg2=action_space_config[2], arg3=action_space_config[3])

        self.k_nearest_neighbors = max(1, int(max_actions * k_ratio))

        self.update_count = 0

    def get_short_name(self):
        return "Wolp"

    def get_version(self):
        adaptation = self.action_space_config[0]
        if adaptation is 'auto':
            return 6
        elif adaptation is 'custom':
            return 5
        else:
            return 4

    def get_specs(self):
        return '{}k{}'.format(self.action_space.get_size(), self.k_nearest_neighbors)

    def get_action_space(self):
        return self.action_space

    def get_action_space_size(self):
        return self.action_space.get_current_size()

    def act(self, state):
        # taking a continuous action from the actor
        proto_action = super().act(state)

        # return the best neighbor of the proto action
        return self.wolp_action(state, proto_action)

    def observe(self, episode):
        super().observe(episode)
        if episode['done'] == 1:
            min_action_space_size = self.action_space.get_current_size()
            adapted_action_space = self.action_space.update()
            if adapted_action_space:
                print("Adapting action space")
                self.update_count += 1

            max_action_space_size = self.action_space.get_current_size()

    def wolp_action(self, state, proto_action):
        # get the proto_action's k nearest neighbors
        actions, indexes = self.action_space.search_point(proto_action, self.k_nearest_neighbors)
        actions = actions[0]

        # make all the state-action pairs for the critic
        states = np.tile(state, [len(actions), 1])
        # evaluate each pair through the critic
        actions_evaluation = self.critic_net.evaluate_critic(states, actions)
        # find the pair with the maximum value
        max_index = np.argmax(actions_evaluation)
        result_action = actions[max_index]
        result_index = indexes[max_index]
        # return index to action space module
        self.action_space.action_selected(result_index)
        # return the best action
        return result_action
