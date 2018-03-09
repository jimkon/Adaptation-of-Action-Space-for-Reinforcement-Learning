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

    data_process = Data_handler(name)

    print("Data file is loaded")

    data_process.plot_rewards()
    data_process.plot_average_reward()
    data_process.plot_actions()
    data_process.plot_action_distribution()
    data_process.plot_action_distribution_over_time()
    data_process.plot_action_error()
    data_process.plot_action_space_size()


if __name__ == '__main__':
    show(folder='/', episodes=10000, actions=127, k=12,
         experiment='InvertedPendulum-v1', v=4, id=0)
    # show()
