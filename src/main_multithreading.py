#!/usr/bin/python3

# import gym
#
# import numpy as np
#
# from wolp_agent import *
# from ddpg.agent import DDPGAgent
# import util.data
# from util.timer import Timer
# from monitor import *
# from ddpg.ou_noise import *

from multiprocessing import Pool
import itertools, os
import random, time
from tqdm import tqdm


def run(episodes=10000,
        render=False,
        experiment='InvertedPendulum-v1',
        max_actions=1000,
        adapted_action_space=True,
        knn=0.1):

    start = time.time()
    count = 0
    iters = int(10e6)
    # pbar = tqdm(total=iters)
    for i in range(iters):
        count+=1
        # if count%100000==0:
        #     pbar.update(1)
            # print('process id:', os.getpid(), "%", 100*i/iters)

    elapsed_time = time.time()-start
    print(episodes, render, experiment, max_actions, adapted_action_space, knn,
        "elapsed time", elapsed_time, "for workload:", count )
    return elapsed_time



if __name__ == '__main__':
    pool = Pool(processes=8)

    episodes = [1000]
    render = [False]
    experiment=['InvertedPendulum-v1']
    max_actions = [1, 2, 3, 4]
    adapted_action_space = [True]
    knn = [0.1, 0.2, 0.5]

    args_compinations = itertools.product(episodes, render, experiment, max_actions, adapted_action_space, knn)
    # print(args_compinations)
    # print(len(list(args_compinations)))



    result = sum(pool.starmap(run, args_compinations))
    

    pool.close()
    pool.join()
    print("total time needed", result)

    # run()
