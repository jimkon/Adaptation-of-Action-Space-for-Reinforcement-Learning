import numpy as np
import pyflann
from gym.spaces import Box
from ddpg import agent
import action_space


class WolpertingerAgent(agent.DDPGAgent):

    def __init__(self, env, max_actions=1e5, k_ratio=0.1):
        super().__init__(env)
        self.experiment = env.spec.id
        if self.continious_action_space:
            self.action_space = action_space.Space(self.low, self.high, max_actions)
            max_actions = self.action_space.get_number_of_actions()
        else:
            print("this version doesn't work for discrete actions spaces")
            exit()

        self.k_nearest_neighbors = max(1, int(max_actions * k_ratio))

    def get_name(self):
        return 'Wolp4_{}k{}_{}'.format(self.action_space.get_max_size(),
                                       self.k_nearest_neighbors, self.experiment)

    def get_action_space(self):
        return self.action_space

    def get_action_space_size(self):
        return self.action_space.get_size()

    def act(self, state):
        # taking a continuous action from the actor
        proto_action = super().act(state)

        # return the best neighbor of the proto action
        return self.wolp_action(state, proto_action)

    def observe(self, episode):
        super().observe(episode)
        if episode['done'] == 1:
            min_action_space_size = self.action_space.get_size()

            self.action_space.update()

            max_action_space_size = self.action_space.get_size()
            self.data_fetch.set_action_space_size(min_action_space_size, max_action_space_size)

    def wolp_action(self, state, proto_action):
        # get the proto_action's k nearest neighbors
        actions, indexes = self.action_space.search_point(proto_action, self.k_nearest_neighbors)
        actions = actions[0]
        self.data_fetch.set_ndn_action(actions[0].tolist())
        # make all the state-action pairs for the critic
        states = np.tile(state, [len(actions), 1])
        # evaluate each pair through the critic
        actions_evaluation = self.critic_net.evaluate_critic(states, actions)
        # find the pair with the maximum value
        max_index = np.argmax(actions_evaluation)
        result_action = actions[max_index]
        result_index = -1  # indexes[max_index]
        # return index to action space module
        self.action_space.action_selected(result_index)
        # return the best action
        return result_action
