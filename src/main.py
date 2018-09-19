from run import *
from run_multiproc import *

from multiprocessing import Pool
import itertools
import os
import json
import math
import pickle

FILE = "D:/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/src/proc.json"


if __name__ == '__main__':
    # produce(json.dumps([math.pow, [2]]))
    # training("LunarLanderContinuous-v2", 200, 5, 1023, comment="training", start_id=1)
    # test_run("InvertedDoublePendulum-v2", 501, render=False)
    # produce(func="test_run",
    #         args={
    #             "experiment": "LunarLanderContinuous-v2",
    #             "episodes": 200,
    #             "render": False,
    #             "save_data": False,
    #             "max_actions": 1000,
    #             "knn": 0.1,
    #             "silent": False})
    produce_combos(func="test_run",
                   args={
                       "experiment": ["InvertedPendulum-v2"],
                       "episodes": [100, 150, 200],
                       "render": [False],
                       "save_data": [False],
                       "max_actions": [1000],
                       "knn": [0.1],
                       "silent": [True]})
