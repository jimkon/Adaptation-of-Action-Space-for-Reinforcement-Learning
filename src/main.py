from run import *
from run_multiproc import *

from multiprocessing import Pool
import itertools
import os
import json
import math
import pickle


if __name__ == '__main__':
    # gather_results("LunarLanderContinuous-v2",
    #                2000,
    #                1365,
    #                0.1,
    #                action_space_config=['auto', 'square', 20000, -1],
    #                save_action_space=True,
    #                agent_to_load=["Wolp4", "training/prev/batch200_47"])
    #
    # test_run("InvertedPendulum-v2",
    #          100,
    #          max_actions=8191,
    #          knn=0.1,
    #          render=True,
    #          agent_to_load=["Wolp4", "training"],
    #          training_flag=False)
    training(experiment="InvertedDoublePendulum-v2",
             reset_after_episodes=1000,
             max_batches=10,
             max_actions=4095,
             knn=0.25,
             start_id=0)

    # produce_combos(func="gather_results",
    #                args={
    #                    "experiment": ["InvertedPendulum-v2"],
    #                    "episodes": [2000],
    #                    "max_actions": [7, 15, 31, 63, 127, 255, 511],
    #                    "knn": [.1, .2, .3, .4, .5, 1],
    #                    "save_action_space": [True],
    #                    "silent": [True]})

    # -------------------------------
    # produce_combos(func="gather_results",
    #                args={
    #                    "experiment": ["HalfCheetah-v2"],
    #                    "episodes": [250],
    #                    "max_actions": [1000, 10000, 50000, 266305],
    #                    "knn": [.1, .25, .5],
    #                    "save_action_space": [True],
    #                    "agent_to_load": [["Wolp4", "training/prev/batch100_23"]]})

    # produce_combos(func="gather_results",
    #                args={
    #                    "experiment": ["LunarLanderContinuous-v2"],
    #                    "episodes": [200],
    #                    "max_actions": [100, 1000, 10000],
    #                    "knn": [.1, .2, .3, .4, .5, 1],
    #                    "save_action_space": [True],
    #                    "agent_to_load": [["Wolp4", "training/prev/batch200_47"]]})
