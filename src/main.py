from run import *
from run_multiproc import *

from multiprocessing import Pool
import itertools
import json
import math
import pickle
import os

if __name__ == '__main__':
    # test_run("CartPole-v1",
    #          100,
    #          max_actions=31,
    #          knn=0.25,
    #          render=True,
    #          # training_flag=False,
    #          # load_agent=True,
    #          # action_space_config=['auto', 'square', 2000, 10],
    #          # save_data=True,
    #          agent_to_load=["Wolp4", "training-1d"])
    # test_run("CartPole-v1",
    #          1000,
    #          max_actions=127,
    #          knn=0.1,
    #          render=True,
    #          training_flag=True,
    #          load_agent=True,
    #          agent_to_load=["Wolp4", "training-1d"])
    #
    # gather_results("CartPole-v1",
    #                episodes=1000,
    #                max_actions=21,
    #                knn=0.1,
    #                save_action_space=False,
    #                id=2)

    produce_combos(func="gather_results",
                   args={
                       "experiment": ["CartPole-v1"],
                       "episodes": [1000],
                       "max_actions": [15, 31, 63, 127, 255],
                       "knn": [.1, .2, .3],
                       "save_action_space": [False],
                       "id": [1, 2]})

    # training(experiment="CartPole-v1",
    #          reset_after_episodes=2000,
    #          max_batches=15,
    #          max_actions=4095,
    #          knn=0.25,
    #          start_id=0,
    #          agent_to_load=None,
    #          comment="training-1d")

    # gather_results("LunarLanderContinuous-v2",
    #                episodes=2000,
    #                max_actions=73,
    #                knn=0.4,
    #                id=0)
    # gather_results("LunarLanderContinuous-v2",
    #                episodes=2000,
    #                max_actions=585,
    #                knn=0.4,
    #                id=0)

    # gather_results("LunarLanderContinuous-v2",
    #                episodes=2000,
    #                max_actions=585,
    #                knn=0.3,
    #                id=0)
    # gather_results("LunarLanderContinuous-v2",
    #                episodes=2000,
    #                max_actions=73,
    #                knn=0.4,
    #                id=0)

    #
    # test_run("LunarLanderContinuous-v2",
    #          100,
    #          max_actions=5461,
    #          knn=0.1,
    #          render=True,
    #          agent_to_load=["Wolp4", "training"],
    #          training_flag=False)
    # training(experiment="InvertedPendulum-v2",
    #          reset_after_episodes=200,
    #          max_batches=5,
    #          max_actions=4095,
    #          knn=0.5,
    #          start_id=10)
    # training(experiment="LunarLanderContinuous-v2",
    #          reset_after_episodes=1000,
    #          max_batches=10,
    #          max_actions=5461,
    #          knn=0.2,
    #          start_id=5)

    # produce_combos(func="gather_results",
    #                args={
    #                    "experiment": ["InvertedPendulum-v2"],
    #                    "episodes": [2000],
    #                    "max_actions": [127],
    #                    "knn": [.1],
    #                    "save_action_space": [True],
    #                    "agent_to_load": [["Wolp4", "training/prev/batch1000_9"]]})

    # produce_combos(func="gather_results",
    #                args={
    #                    "experiment": ["InvertedPendulum-v2"],
    #                    "episodes": [2000],
    #                    "max_actions": [15, 31, 63, 127, 255],
    #                    "knn": [.1, .2, .3],
    #                    "save_action_space": [False],
    #                    "id": [3, 4, 5]})

    # -------------------------------
    # produce_combos(func="gather_results",
    #                args={
    #                    "experiment": ["HalfCheetah-v2"],
    #                    "episodes": [250],
    #                    "max_actions": [1000, 10000, 50000, 266305],
    #                    "knn": [.1, .25, .5],
    #                    "save_action_space": [True],
    #                    "agent_to_load": [["Wolp4", "training/prev/batch100_23"]]})

    # produce_coms
