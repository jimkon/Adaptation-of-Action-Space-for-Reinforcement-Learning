import numpy as np
import pyflann
from gym.spaces import Box
from ddpg import agent
import action_space


class WolpertingerAgent(agent.DDPGAgent):

    def __init__(self, env, max_actions=1e5, k_ratio=0.1, action_space_monitor=None):
        super().__init__(env)
        self.experiment = env.spec.id
        if self.continious_action_space:
            self.action_space = action_space.Space(
                self.low, self.high, max_actions, action_space_monitor)
        else:
            print('This version works only for continuous action space')
            exit()

        self.k_nearest_neighbors = int(max_actions * k_ratio)

    def get_name(self):
        return 'Wolp3_{}k{}_{}'.format(self.action_space.get_number_of_actions(),
                                       self.k_nearest_neighbors, self.experiment)

    def get_action_space(self):
        return self.action_space

    def act(self, state):
        # taking a continuous action from the actor
        proto_action = super().act(state)
        if self.k_nearest_neighbors < 1:
            return proto_action

        # return the best neighbor of the proto action
        return self.wolp_action(state, proto_action)

    def wolp_action(self, state, proto_action):
        # get the proto_action's k nearest neighbors
        actions, indexes = self.action_space.search_point(proto_action, self.k_nearest_neighbors)
        # make all the state-action pairs for the critic
        states = np.tile(state, [len(actions), 1])
        # evaluate each pair through the critic
        actions_evaluation = self.critic_net.evaluate_critic(states, actions)
        # find the pair with the maximum value
        max_index = np.argmax(actions_evaluation)
        result_action = actions[max_index]
        result_index = indexes[max_index]
        # return index to action space module
        self.action_space.action_selected(result_index, proto_action)
        # return the best action
        return result_action
