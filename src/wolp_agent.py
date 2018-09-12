import numpy as np
import pyflann
from gym.spaces import Box
from ddpg import agent
import action_space


class WolpertingerAgent(agent.DDPGAgent):

    ACTION_SPACE_SAMPLE_BUFFER_SIZE = 10000

    def __init__(self, env, result_dir, max_actions=1e5, k_ratio=0.1, adapted_action_space=True,
                 training_flag=True):
        super().__init__(env, result_dir, training_flag=training_flag)

        assert self.continious_action_space, "this version doesn't work for discrete actions spaces"

        self.action_space = action_space.Space(self.low, self.high, max_actions)

        # max_actions = self.action_space.get_number_of_actions()

        self.k_nearest_neighbors = max(1, int(max_actions * k_ratio))

        self.adapted_action_space = adapted_action_space
        self.sample_count = 0
        self.update_count = 0

    def get_name(self):
        return "Wolp5"

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

            if self.adapted_action_space and self.sample_count > self.ACTION_SPACE_SAMPLE_BUFFER_SIZE:
                print("Adapting action space")
                self.action_space.update()
                self.update_count += 1
                self.action_space.plot_space(
                    filename="/home/jim/Desktop/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/results/pics/a{}.png".format(self.update_count))
                self.sample_count = 0
                # print('new action space:\n', self.action_space.get_space())

            max_action_space_size = self.action_space.get_current_size()
            self.data_fetch.set_action_space_size(min_action_space_size, max_action_space_size)

    def wolp_action(self, state, proto_action):
        # get the proto_action's k nearest neighbors
        actions, indexes = self.action_space.search_point(proto_action, self.k_nearest_neighbors)
        self.sample_count += 1
        actions = actions[0]
        self.data_fetch.set_ndn_action(actions[0].tolist())
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
