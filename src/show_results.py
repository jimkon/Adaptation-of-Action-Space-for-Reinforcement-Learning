#!/usr/bin/python3
import numpy as np
from util.data_process import *


def show(folder='saved/',
         episodes=2500,
         actions=7,
         k=1,
         experiment='InvertedPendulum-v1',
         v=4,
         id=0):

    name = '{}data_{}_Wolp{}_{}{}k{}#{}.json.zip'.format(folder,
                                                         episodes,
                                                         v,
                                                         experiment[:3],
                                                         actions,
                                                         k,
                                                         id
                                                         )

    # template = '/home/jim/Desktop/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/results/obj/saved/data_2000_Wolp{}_Car{}k{}#{}.json.zip'
    #
    # data_process = Data_handler(template.format(4, 2047, 204, 1), abs_path=True)
    data_process = Data_handler(name)

    print("Data file is loaded")

    data_process.plot_sensitivity_efficiency()
    data_process.plot_rewards()
    data_process.plot_average_reward()
    data_process.plot_actions()
    data_process.plot_action_distribution()
    data_process.plot_action_distribution_over_time()
    data_process.plot_discretization_error()
    data_process.plot_action_space_size()


if __name__ == '__main__':
    show()
