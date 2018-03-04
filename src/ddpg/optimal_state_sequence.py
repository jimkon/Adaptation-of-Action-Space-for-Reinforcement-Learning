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

    def goal_state(self, current_state):
        if len(self._sequence) == 0:
            return np.zeros(self.state_size)

        index, distance = self._flann.nn_index(state, 2)

        if distance[0] < self.STATE_REACHED_DISTANCE:
            return self._sequence[index[0]]
        else:
            if index == len(self._sequence):
                return self._sequence[index[1]]
            else:
                return self._sequence[index[0] + 1]

    def _reset_current_sequence(self):
        self._current_sequence = []
        self._current_total_reward = 0

    def _compare_sequences(self):
        if self._current_total_reward > self._total_reward:
            self._flann.delete_index()

            self._total_reward = self._current_total_reward
            self._sequence = self._current_sequence

            self._index = self._flann.build_index(self._sequence)

        self._reset_current_sequence()
