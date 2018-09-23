from run import *
from run_multiproc import *

from multiprocessing import Pool
import itertools
import os
import json
import math
import pickle


if __name__ == '__main__':
    gather_results("InvertedPendulum-v2",
                   1000,
                   63,
                   0.1,
                   action_space_config=['auto', 'square', 10000, -1],
                   save_action_space=True)
    #
    # test_run("HalfCheetah-v2",
    #          10,
    #          max_actions=4161,
    #          knn=0.1,
    #          render=True,
    #          agent_to_load=["Wolp4", "training"],
    #          training_flag=False)
    # training(experiment="Pusher-v2",
    #          reset_after_episodes=100,
    #          max_batches=50,
    #          max_actions=16513,
    #          knn=0.1,
    #          start_id=100)
    # produce_combos(func="gather_results",
    #                args={
    #                    "experiment": ["InvertedPendulum-v2"],
    #                    "episodes": [100, 150, 200],
    #                    "render": [False],
    #                    "save_data": [False],
    #                    "max_actions": [1000],
    #                    "knn": [0.1],
    #                    "silent": [True]})
