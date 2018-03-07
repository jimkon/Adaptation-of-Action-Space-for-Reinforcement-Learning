#!/usr/bin/python3
import numpy as np
import pyflann


class State_sequence:

    STATE_REACHED_DISTANCE = .1  # not final

    def __init__(self, state_size):
        self.state_size = state_size

        self._sequence = []
        self._total_reward = 0

        self._flann = pyflann.FLANN()

        self._reset_current_sequence()
        pass

    def add_state_reward(self, state, reward, done):
        assert len(state) == self.state_size, 'Input state has different size: {} != {}'.format(
            self.state_size, len(state))

        self._current_sequence.append(state)
        self._current_total_reward += reward

        if done:
            self._compare_sequences()

    def goal_state(self, state):
        state = np.array(state).astype(float)
        if len(self._sequence) == 0:
            return np.zeros(self.state_size)

        print('state', state)
        index, distance = self._flann.nn_index(state, 2)
        print('indexes', index)
        print('d', distance)

        index = index[0]
        distance = distance[0][0]

        if distance > self.STATE_REACHED_DISTANCE:
            print('result1', self._sequence[index[0]])
            return self._sequence[index[0]]
        else:
            if index[0] == len(self._sequence) - 1:
                print('result2', self._sequence[index[1]])

                return self._sequence[index[1]]
            else:
                print('result3', self._sequence[index[0] + 1])

                return self._sequence[index[0] + 1]

    def get_size(self):
        return len(self._sequence)

    def get_reward(self):
        return self._total_reward

    def _reset_current_sequence(self):
        self._current_sequence = []
        self._current_total_reward = 0

    def _compare_sequences(self):
        if self._current_total_reward > self._total_reward:
            self._flann.delete_index()

            self._total_reward = self._current_total_reward
            self._sequence = self._current_sequence

            self._index = self._flann.build_index(np.copy(self._sequence).astype(float))

        self._reset_current_sequence()
