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
import sys
import tensorflow as tf

from run import *


FILE = "D:/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/src/proc.json"
CPUS = 3

FILE_TEMPLATE = {
    "consumed": [],
    "failed": [],
    "queued": []
}


def create_file_if_not_exists():
    if os.path.exists(FILE):
        return
    else:
        with open(FILE, 'w') as f:
            json.dump(FILE_TEMPLATE, f, indent=2, sort_keys=True)


def produce(func, args):
    create_file_if_not_exists()
    file = None
    with open(FILE, 'r') as f:
        file = json.load(f)

    with open(FILE, 'w') as f:
        if file:
            file['queued'].append([func, args])
            print("Adding ", [func, args], "to proc.json")
            json.dump(file, f, indent=2, sort_keys=True)


def produce_combos(func, args):
    from collections import Counter
    from itertools import product

    keys = list(Counter(args).keys())
    values = list(Counter(args).values())
    # print(keys)
    # print(values)
    combos = product(*values)
    dicts = []
    count = 0
    for c in combos:
        # print(c)
        d = dict()
        for i in range(len(keys)):
            d[keys[i]] = c[i]
        count += 1
        dicts.append(d)
        # print(d)

    for d in dicts:
        produce(func=func, args=d)

    print("Added", count, "processes in queue")


def consume():
    create_file_if_not_exists()
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
    create_file_if_not_exists()
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


def caller(proc):
    print("run proc", proc)
    try:
        func, args = proc
        getattr(sys.modules[__name__], func)(**args)
        return proc, None
    except Exception as e:
        print(e)
        return proc, str(e)


if __name__ == '__main__':

    total_time = 0

    all_procs = consume()

    print("Processes found on proc.json:", len(all_procs))

    # procs, all_procs = batch_procs(all_procs)

    pool = Pool(processes=CPUS)
    procs = all_procs

    print("processes:", len(procs), "left:", len(all_procs))
    # for p in procs:
    #     print(p)

    start_time = time.time()

    # func = procs[0][0]
    # args = (p[1] for p in procs)
    # print("mapping", func, "("+str(len(procs))+") with args", args)

    ps = (p for p in procs)

    results = pool.map(caller, ps, CPUS)

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

    print("Total time", total_time)
