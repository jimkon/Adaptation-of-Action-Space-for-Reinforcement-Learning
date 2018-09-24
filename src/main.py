from run import *
from run_multiproc import *

from multiprocessing import Pool
import itertools
import os
import json
import math
import pickle


if __name__ == '__main__':
    # gather_results("InvertedPendulum-v2",
    #                2000,
    #                63,
    #                0.5,
    #                action_space_config=['auto', 'square', 20000, -1],
    #                save_action_space=True)
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
    produce_combos(func="gather_results",
                   args={
                       "experiment": ["InvertedPendulum-v2"],
                       "episodes": [2000],
                       "max_actions": [7, 15, 31, 63, 127, 255, 511],
                       "knn": [.1, .2, .3, .4, .5, 1],
                       "save_action_space": [True],
                       "silent": [True]})
