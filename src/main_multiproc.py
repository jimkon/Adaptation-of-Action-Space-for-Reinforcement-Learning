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
import itertools
import os
import random
import time
import json
import main
import sys
import tensorflow as tf

FILE = "D:/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/src/proc.json"
CPUS = 4


def produce(procs):
    file = None
    with open(FILE, 'r') as f:
        file = json.load(f)

    with open(FILE, 'w') as f:
        if file:
            file['queued'].append(procs)
            json.dump(file, f, indent=2, sort_keys=True)


def consume():

    procs = None
    file = None
    with open(FILE, 'r') as f:
        file = json.load(f)
        procs = file['queued']

    with open(FILE, 'w') as f:
        if file:
            file['queued'] = []
            json.dump(file, f, indent=2, sort_keys=True)

    return procs


def log(procs, failed=False):
    if len(procs) == 0:
        return

    file = None
    with open(FILE, 'r') as f:
        file = json.load(f)

    with open(FILE, 'w') as f:
        if file:
            if failed:
                file['failed'].extend(procs)
            else:
                file['consumed'].extend(procs)
            json.dump(file, f, indent=2, sort_keys=True)


def batch_procs(all_procs):
    from collections import Counter
    import numpy as np

    if len(all_procs) == 0:
        return [], all_procs

    proc_names = list(p[0] for p in all_procs)
    # print(proc_names)
    unique_procs = list(Counter(proc_names).keys())
    unique_procs_count = list(Counter(proc_names).values())
    # print(unique_procs)
    # print(unique_procs_count)
    # print(max(unique_procs_count))
    max_proc = unique_procs[np.argmax(unique_procs_count)]
    # print(max_proc)

    result_procs = []
    for p in all_procs:

        if p[0] == max_proc:
            result_procs.append(p)
        if(len(result_procs) >= CPUS):
            break
    for p in result_procs:
        all_procs.remove(p)

    return result_procs, all_procs


def star_test_run(args):
    main.test_run(experiment=args['experiment'],
                  episodes=args['episodes'],
                  knn=args['knn'],
                  max_actions=args['max_actions'],
                  render=args['render'],
                  save_data=args['save_data'],
                  silent=args['silent'])


def caller(proc):
    print("run proc", proc)
    try:
        func, args = proc
        getattr(sys.modules[__name__], "star_{}".format(func))(args)
        return proc, None
    except Exception as e:
        return proc, str(e)


if __name__ == '__main__':

    print("Preload tensorflow")
    pre_load = tf.constant("Preload finished")
    sess = tf.Session()
    print(sess.run(pre_load))

    total_time = 0

    all_procs = consume()

    procs, all_procs = batch_procs(all_procs)
    while(len(procs) > 0):
        print("processes:", len(procs), "left:", len(all_procs))
        # for p in procs:
        #     print(p)

        start_time = time.time()

        # func = procs[0][0]
        # args = (p[1] for p in procs)
        # print("mapping", func, "("+str(len(procs))+") with args", args)

        ps = (p for p in procs)
        pool = Pool(processes=CPUS)

        results = pool.map(caller, ps)

        pool.close()
        pool.join()
        #
        map_time = time.time()-start_time
        total_time += map_time
        print("elapsed time, map", map_time, "for processes", len(procs))

        succeed = []
        failed = []
        # print(results)
        for proc, fail_message in results:
            if fail_message:
                proc.append(json.dumps(fail_message))
                failed.append(proc)
            else:
                succeed.append(proc)

        log(succeed)
        log(failed, failed=True)

        procs, all_procs = batch_procs(all_procs)

        new_procs = consume()
        if len(new_procs) > 0:
            all_procs.extend(new_procs)

    print("Total time", total_time)
    # exit()

    # exit()
    #
    #
    # episodes = [1000]
    # render = [["a", 'b'], ['c', 'd']]
    # experiment = ['InvertedPendulum-v2', 'InvertedDoublePendulum-v2']
    # max_actions = [1, 2, 3, 4]
    # adapted_action_space = [True]
    # knn = [0.1, 0.2, 0.5]
    #
    # args_compinations = itertools.product(
    #     episodes, render, experiment, max_actions, adapted_action_space, knn)
    # # print(args_compinations)
    # # for i in list(args_compinations):
    # #     print(i)
    # # exit()
    #
    # result = sum(pool.starmap(run, args_compinations))

    # pool.close()
    # pool.join()
    # print("total time needed", result)

    # run()
