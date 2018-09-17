import gym

import numpy as np

from wolp_agent import *
from ddpg.agent import DDPGAgent
import util.data
from util.timer import Timer
from monitor import *
from ddpg.ou_noise import *

PROJECT_DIR = "D:/dip/Adaptation-of-Action-Space-for-Reinforcement-Learning/results/"


def run(experiment,
        episodes,
        max_actions,
        knn,
        action_space_config=['off', 'square', 1000, 10],
        result_dir=PROJECT_DIR,
        render=False,
        load_agent=True,
        save_agent=True,
        training_flag=True,
        id=0,
        comment="default"):

    env = gym.make(experiment)

    print(env.observation_space)
    print(env.action_space)

    steps = env.spec.timestep_limit

    agent = WolpertingerAgent(env, result_dir, max_actions=max_actions, k_ratio=knn,
                              training_flag=training_flag,
                              action_space_config=action_space_config)

    if load_agent:
        agent.load_agent(comment=comment)

    timer = Timer()

    data = util.data.Data(agent.result_dir)
    data.set_agent(agent.get_name(), int(agent.action_space.get_size()),
                   agent.k_nearest_neighbors, agent.get_version())
    data.set_experiment(experiment, agent.low.tolist(), agent.high.tolist(), episodes)
    data.set_id(id)

    agent.add_data_fetch(data)
    print(data.get_file_name())

    # if render:
    #     monitor = Monitor(400, env.observation_space.shape[0], env.action_space.shape[0], 50,
    #                       [agent.low.tolist()[0], agent.high.tolist()[0]])

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
            observation, reward, done, info = env.step(action.flatten())

            # if render:
            #     monitor.add_data(observation, action, reward)
            #     monitor.repaint()

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
                # if render:
                #     monitor.end_of_episode()

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
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        episodes, time / 1000, reward_sum / episodes))

    data.save(comment=comment)
    if save_agent:
        agent.save_agent(force=True, comment=comment)
    agent.close_session()


def training(experiment, reset_after_episodes, max_batches, max_actions, knn=0.1, comment="training", finalize=True):

    # start training batches
    for i in range(max_batches):
        run(experiment=experiment,
            episodes=reset_after_episodes,
            max_actions=max_actions,
            knn=knn,
            id=i,
            comment=comment)

    if finalize:
        path_to_dir = "{}/Wolp4/{}/data/{}/".format(PROJECT_DIR, experiment, comment)
        util.data.merge(path_to_dir)

        import os
        import shutil

        jup_template = "{}/jupyter_templates/training.ipynb".format(PROJECT_DIR)
        if os.path.exists(jup_template):
            shutil.copyfile(jup_template, path_to_dir+'/training.ipynb')


if __name__ == '__main__':
    training("InvertedPendulum-v2", 10, 5, 1023, comment="test")
