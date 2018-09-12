#!/usr/bin/python3

import gym

import numpy as np

from wolp_agent import *
from ddpg.agent import DDPGAgent
import util.data
from util.timer import Timer
from monitor import *
from ddpg.ou_noise import *


def run(episodes=10000,
        render=False,
        experiment='InvertedPendulum-v1',
        max_actions=1000,
        adapted_action_space=True,
        knn=0.1,
        load_path=None,
        save_path=None,
        training_flag=True,
        id=0):

    env = gym.make(experiment)

    print(env.observation_space)
    print(env.action_space)

    steps = env.spec.timestep_limit

    # agent = DDPGAgent(env)
    agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn,
                              adapted_action_space=adapted_action_space,
                              training_flag=training_flag)

    if load_path is not None:
        agent.load_agent(load_path)

    timer = Timer()

    data = util.data.Data()
    data.set_agent(agent.get_name(), int(agent.action_space.get_size()),
                   agent.k_nearest_neighbors, 4 if not adapted_action_space else 5)
    data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)
    data.set_id(id)

    agent.add_data_fetch(data)
    print(data.get_file_name())

    if render:
        monitor = Monitor(400, env.observation_space.shape[0], env.action_space.shape[0], 50,
                          [agent.low.tolist()[0], agent.high.tolist()[0]])

    full_epoch_timer = Timer()
    reward_sum = 0

    ou = OUNoise(1, mu=0, theta=0.5, sigma=.1)

    # temp_buffer = [0] * 150

    for ep in range(episodes):

        timer.reset()
        observation = env.reset()

        total_reward = 0
        print('Episode ', ep, '/', episodes - 1, end='. ')
        for t in range(steps):

            if render:
                env.render()

            action = agent.act(observation)

            data.set_action(action.tolist())

            data.set_state(observation.tolist())

            prev_observation = observation
            # some environments need the action as scalar valua, and other as array
            # for scalar: action[0] if len(action) == 1 else action
            observation, reward, done, info = env.step(action if len(action) == 1 else action)

            if render:
                monitor.add_data(observation, action, reward)
                monitor.repaint()

            data.set_reward(reward)
            # if render:
            #     monitor.add_data(observation, action, reward)

            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}
            agent.observe(episode)

            total_reward += reward

            # print(episode['obs'], episode['action'], episode['obs2'], episode['reward'])
            if done or (t == steps - 1):
                if render:
                    monitor.end_of_episode()

                t += 1
                reward_sum += total_reward
                time_passed = timer.get_time()
                print('Reward:{} Steps:{} t:{} ({}/step) Curr avg={}, {} actions({})'.format(total_reward, t,
                                                                                             time_passed,
                                                                                             round(
                                                                                                 time_passed / t),
                                                                                             round(
                                                                                                 reward_sum / (ep + 1)),
                                                                                             agent.get_action_space_size(),
                                                                                             agent.get_action_space_size() / max_actions))

                data.finish_and_store_episode()

                break

        # temp_buffer.append(total_reward)
        # temp_buffer.pop(0)
        # temp_avg = np.average(temp_buffer)
        # print('temp average = ', temp_avg)
        # if temp_avg > 750:
        #     break
    # end of episodes
    time = full_epoch_timer.get_time()
    print('Run {} episode in {} seconds and got {} average reward'.format(
        episodes, time / 1000, reward_sum / episodes))

    data.save()
    if save_path is not None:
        print('Saving agent\'s vaiables')
        agent.save_agent(save_path)


# def run_episode(render=False,
#                 experiment='InvertedPendulum-v1',
#                 max_actions=1000,
#                 adapted_action_space=True,
#                 knn=0.1):
#
#     env = gym.make(experiment)
#
#     print(env.observation_space)
#     print(env.action_space)
#
#     steps = env.spec.timestep_limit
#
#     # agent = DDPGAgent(env)
#     agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn)
#
#     timer = Timer()
#
#     data = util.data.Data()
#     data.set_agent(agent.get_name(), int(agent.action_space.get_size()),
#                    agent.k_nearest_neighbors, 4)
#     data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), 1)
#
#     agent.add_data_fetch(data)
#     print(data.get_file_name())
#
#     if render:
#         monitor = Monitor(400, env.observation_space.shape[0], env.action_space.shape[0], 50)
#
#     full_epoch_timer = Timer()
#     total_reward = 0
#
#     timer.reset()
#     observation = env.reset()
#
#     total_reward = 0
#     reward_hist = []
#     reward_hist_size = 15
#
#     for t in range(steps):
#
#         if render:
#             env.render()
#
#         action = agent.act(observation)
#
#         # data.set_action(action.tolist())
#
#         # data.set_state(observation.tolist())
#
#         prev_observation = observation
#         # some environments need the action as scalar valua, and other as array
#         # for scalar: action[0] if len(action) == 1 else action
#         observation, reward, done, info = env.step(action if len(action) == 1 else action)
#
#         # data.set_reward(reward)
#         if render:
#             monitor.add_data(observation, action, reward)
#             monitor.repaint()
#
#         episode = {'obs': prev_observation,
#                    'action': action,
#                    'reward': reward,
#                    'obs2': observation,
#                    'done': done,
#                    't': t}
#
#         agent.observe(episode)
#
#         total_reward += reward
#         reward_hist.append(reward)
#         if len(reward_hist) > reward_hist_size:
#             reward_hist.pop(0)
#
#         t += 1
#         time_passed = timer.get_time()
#         print('Reward:{}, Total reward={}, Step:{} ({}/step) Total avg={}, W average={}, {} actions({})'.format(reward,
#                                                                                                                 round(
#                                                                                                                     total_reward), t,
#                                                                                                                 round(
#                                                                                                                     time_passed / t),
#                                                                                                                 round(
#                                                                                                                     total_reward / (t + 1)),
#                                                                                                                 round(np.average(
#                                                                                                                     reward_hist)),
#                                                                                                                 agent.get_action_space_size(),
#                                                                                                                 agent.get_action_space_size() / max_actions))
#
#         # data.finish_and_store_episode()
#
#     # end of episodes
#     time = full_epoch_timer.get_time()
#     print('Run {} steps in {} seconds and got reward {}'.format(
#         t, time / 1000, total_reward))
#
#     data.save()
#
#
# if __name__ == '__main__':
#     run()
