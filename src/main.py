from run import *
from run_multiproc import *

from multiprocessing import Pool
import itertools
import os
import json
import math
import pickle


def number_of_actions(dims, max_level=10):
    def actions_per_level(level, dims):
        return 2**(level*dims)

    def actions_per_dim(level):
        return 2**(level+1)-1

    def total_actions(level, dims):
        apl = actions_per_level(level, dims)
        if level == 0:
            return apl
        else:
            return apl+total_actions(level-1, dims)

    for i in range(0, max_level):
        level = i
        apd = actions_per_dim(i)
        ta = total_actions(i, dims)
        string = "Level {}: actions per dimension: {}, total actions: {}".format(i, apd, ta)
        print(string)


if __name__ == '__main__':
    # gather_results("InvertedDoublePendulum-v2",
    #                10,
    #                7,
    #                0.1)
    # number_of_actions(6)
    #
    # test_run("BipedalWalker-v2",
    #          100,
    #          max_actions=16513,
    #          knn=0.1,
    #          render=True,
    #          agent_to_load=["Wolp4", "training"],
    #          training_flag=False)
    training(experiment="HalfCheetah-v2",
             reset_after_episodes=100,
             max_batches=100,
             max_actions=4161,
             knn=0.1,
             start_id=0)
    # produce_combos(func="test_run",
    #                args={
    #                    "experiment": ["InvertedPendulum-v2"],
    #                    "episodes": [100, 150, 200],
    #                    "render": [False],
    #                    "save_data": [False],
    #                    "max_actions": [1000],
    #                    "knn": [0.1],
    #                    "silent": [True]})
