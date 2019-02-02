import random
import numpy as np


import gym
from gym.spaces import Box, Discrete

from actor_net import ActorNet
from critic_net import CriticNet
from actor_net_bn import ActorNet_bn
from critic_net_bn import CriticNet_bn
from tensorflow_grad_inverter import grad_inverter

from collections import deque

import os


"""Some agent implementations including stevenpjg's DDPG agent"""


class Agent:

    def __init__(self, env, result_dir):

        self.env = env
        self.result_dir = result_dir
        # checking state space
        if isinstance(env.observation_space, Box):
            self.observation_space_size = env.observation_space.shape[0]
        else:
            self.observation_space_size = env.observation_space.n

        # checking action space
        if isinstance(env.action_space, Box):
            self.action_space_size = env.action_space.shape[0]
            self.continious_action_space = True
            self.low = env.action_space.low
            self.high = env.action_space.high
        else:  # dont need in this implementation
            self.action_space_size = 1
            self.continious_action_space = False
            self.low = np.array([0])
            self.high = np.array([env.action_space.n])

        if not os.path.exists(self.get_dir()):
            os.makedirs(self.get_dir(), exist_ok=True)

    def act(self, state):
        pass

    def observe(self, episode):
        pass

    def get_name(self):
        return self.get_short_name()+str(self.get_version())

    def get_short_name(self):
        return 'Agent'

    def get_version(self):
        return 0

    def _np_shaping(self, array, is_state):

        number_of_elements = array.shape[0] if len(array.shape) > 1 else 1
        size_of_element = self.observation_space_size if is_state else self.action_space_size

        res = np.array(array)
        res.shape = (number_of_elements, size_of_element)
        return res

    def get_dir(self):
        return "{}/{}/{}".format(self.result_dir, self.get_name(), self.env.spec.id)


class DDPGAgent(Agent):
    ''' stevenpjg's implementation of DDPG algorithm '''

    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 64
    GAMMA = 0.99

    def __init__(self, env, dir, is_batch_norm=False, is_grad_inverter=True,
                 training_flag=True):
        super().__init__(env, dir)
        assert isinstance(env.action_space, Box), "action space must be continuous"

        if is_batch_norm:
            self.critic_net = CriticNet_bn(self.observation_space_size,
                                           self.action_space_size)
            self.actor_net = ActorNet_bn(self.observation_space_size,
                                         self.action_space_size)

        else:
            self.critic_net = CriticNet(self.observation_space_size,
                                        self.action_space_size)
            self.actor_net = ActorNet(self.observation_space_size,
                                      self.action_space_size)

        self.is_grad_inverter = is_grad_inverter
        self.training_flag = training_flag
        self.replay_memory = deque()

        self.time_step = 0

        action_max = np.array(self.high).tolist()
        action_min = np.array(self.low).tolist()
        action_bounds = [action_max, action_min]
        self.grad_inv = grad_inverter(action_bounds)

        self.data_fetch = None

    def add_data_fetch(self, df):
        self.data_fetch = df

    def get_short_name(self):
        return 'DDPG'

    def act(self, state):
        state = self._np_shaping(state, True)
        result = self.actor_net.evaluate_actor(state).astype(float)

        if self.data_fetch:
            self.data_fetch.set_actors_action(result[0].tolist())

        return result

    def observe(self, episode):
        episode['obs'] = self._np_shaping(episode['obs'], True)
        episode['action'] = self._np_shaping(episode['action'], False)
        episode['obs2'] = self._np_shaping(episode['obs2'], True)

        self.add_experience(episode)

    def add_experience(self, episode):
        self.replay_memory.append(episode)

        self.time_step += 1
        if len(self.replay_memory) > type(self).REPLAY_MEMORY_SIZE:
            self.replay_memory.popleft()

        if len(self.replay_memory) > type(self).BATCH_SIZE:
            res = self.train()
            return res
        else:
            return None

    def minibatches(self):
        batch = random.sample(self.replay_memory, type(self).BATCH_SIZE)
        # state t
        state = self._np_shaping(np.array([item['obs'] for item in batch]), True)
        # action
        action = self._np_shaping(np.array([item['action'] for item in batch]), False)
        # reward
        reward = np.array([item['reward'] for item in batch])
        # state t+1
        state_2 = self._np_shaping(np.array([item['obs2'] for item in batch]), True)
        # doneA
        done = np.array([item['done'] for item in batch])

        return state, action, reward, state_2, done

    def train(self):
        if not self.training_flag:
            return

        # sample a random minibatch of N transitions from R
        state, action, reward, state_2, done = self.minibatches()

        actual_batch_size = len(state)

        target_action = self.actor_net.evaluate_target_actor(state)

        # Q'(s_i+1,a_i+1)
        q_t = self.critic_net.evaluate_target_critic(state_2, target_action)

        y = []
        for i in range(0, actual_batch_size):

            if done[i]:
                y.append(reward[i])
            else:
                y.append(reward[i] + type(self).GAMMA * q_t[i][0])  # q_t+1 instead of q_t

        y = np.reshape(np.array(y), [len(y), 1])

        # Update critic by minimizing the loss
        self.critic_net.train_critic(state, action, y)

        # Update actor proportional to the gradients:
        # action_for_delQ = self.act(state)  # was self.evaluate_actor instead of self.act
        action_for_delQ = self.actor_net.evaluate_actor(state)  # dont need wolp action

        if self.is_grad_inverter:
            del_Q_a = self.critic_net.compute_delQ_a(state, action_for_delQ)  # /BATCH_SIZE
            del_Q_a = self.grad_inv.invert(del_Q_a, action_for_delQ)
        else:
            del_Q_a = self.critic_net.compute_delQ_a(state, action_for_delQ)[0]  # /BATCH_SIZE

        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(state, del_Q_a)

        # Update target Critic and actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()

    def save_agent(self, force=False, comment="default"):
        path = "{}/weights/{}".format(self.get_dir(), comment)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print("Saving agent in", path)
            self.actor_net.save_model(path + '/actor.ckpt')
            self.critic_net.save_model(path + '/critic.ckpt')
        else:
            if force:
                print("Overwrite old agent in", path)
                self.actor_net.save_model(path + '/actor.ckpt')
                self.critic_net.save_model(path + '/critic.ckpt')
            else:
                print("Save aborted. An agent is already saved in ", path)

    def load_agent(self, agent_name=None, comment="default"):
        if agent_name is None:
            path = "{}/weights/{}".format(self.get_dir(), comment)
        else:
            path = "{}/{}/{}/weights/{}".format(self.result_dir,
                                                agent_name, self.env.spec.id, comment)
        if os.path.exists(path):
            print("Loading agent saved in", path)
            self.actor_net.load_model(path + '/actor.ckpt')
            self.critic_net.load_model(path + '/critic.ckpt')
        else:
            print("Agent not found in", path)

    def close_session(self):
        self.actor_net.close()
        self.critic_net.close()
