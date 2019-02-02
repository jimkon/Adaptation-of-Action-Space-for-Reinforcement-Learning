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
        agent_to_load=None,
        save_agent=True,
        save_data=True,
        training_flag=True,
        id=0,
        comment="run",
        close_session=True,
        silent=False,
        tempsave=True,
        save_action_space=False):

    env = gym.make(experiment)

    print(env.observation_space)
    print(env.action_space)

    steps = env.spec.timestep_limit

    agent = WolpertingerAgent(env, result_dir, max_actions=max_actions, k_ratio=knn,
                              training_flag=training_flag,
                              action_space_config=action_space_config,
                              save_action_space=save_action_space)
    if load_agent:
        if agent_to_load is not None:
            agent.load_agent(agent_name=agent_to_load[0], comment=agent_to_load[1])
        else:
            agent.load_agent(comment=comment)
    timer = Timer()

    if save_data:
        data = util.data.Data(agent.get_dir(), comment=comment, tempsave=tempsave)
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
        if not silent:
            print('Episode ', ep, '/', episodes - 1, end='. ')

        for t in range(steps):

            if render:
                env.render()

            action = agent.act(observation)

            if save_data:
                data.set_action(action.tolist())
                data.set_state(observation.tolist())

            prev_observation = observation
            # some environments need the action as scalar valua, and other as array
            # for scalar: action[0] if len(action) == 1 else action
            observation, reward, done, info = env.step(action.flatten())

            # if render:
            #     monitor.add_data(observation, action, reward)
            #     monitor.repaint()
            if save_data:
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
                if not silent:
                    print('Reward:{} Steps:{} t:{} ({}/step) Curr avg={}, {} actions({})'.format(total_reward, t,
                                                                                                 time_passed,
                                                                                                 round(
                                                                                                     time_passed / t),
                                                                                                 round(
                                                                                                     reward_sum / (ep + 1)),
                                                                                                 agent.get_action_space_size(),
                                                                                                 agent.get_action_space_size() / max_actions))
                if save_data:
                    data.finish_and_store_episode()

                break

    # end of episodes
    time = full_epoch_timer.get_time()
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        episodes, time / 1000, reward_sum / episodes))

    if save_data:
        data.save()
    if save_agent:
        agent.save_agent(force=True, comment=comment)

    if close_session:
        agent.close_session()
    print("END")
    return agent


def training(experiment,
             reset_after_episodes,
             max_batches,
             max_actions,
             knn=0.1,
             comment="training",
             agent_to_load=None,
             finalize=True,
             start_id=0,
             tempsave=False):

    # start training batches
    for i in range(start_id, start_id+max_batches):

        agent = run(experiment=experiment,
                    episodes=reset_after_episodes,
                    max_actions=max_actions,
                    knn=knn,
                    id=i,
                    comment=comment,
                    agent_to_load=agent_to_load,
                    close_session=False,
                    silent=False,
                    tempsave=tempsave)

        agent.save_agent(comment="{}/{}/batch{}_{}".format(comment,
                                                           'prev', reset_after_episodes, i))
        agent.close_session()

    if finalize:
        path_to_dir = "{}/data/{}/".format(agent.get_dir(), comment)
        util.data.merge(path_to_dir)

        # import os
        # import shutil
        #
        # jup_file = "training.ipynb"
        # jup_template = "{}/jupyter_templates/{}".format(PROJECT_DIR, jup_file)
        # dest_file = "{}/".format(path_to_dir)
        # if os.path.exists(jup_template) and not os.path.exists(dest_file):
        #     print("Adding training notebook")
        #     shutil.copyfile(jup_template, dest_file)


def gather_results(experiment,
                   episodes,
                   max_actions,
                   knn=.1,
                   # action_space_config=['off', 'square', 20000, 10],
                   action_space_config=['auto', 'square', 20000, 10],
                   agent_to_load=["Wolp4", "training-1d"],
                   id=0,
                   comment="results",
                   silent=False,
                   tempsave=False,
                   save_action_space=False):

    t_comment = "{}/{}/{}".format(comment, action_space_config[0], action_space_config[1])
    run(experiment=experiment,
        episodes=episodes,
        max_actions=max_actions,
        knn=knn,
        action_space_config=action_space_config,
        render=False,
        load_agent=True,
        agent_to_load=agent_to_load,
        save_agent=False,
        save_data=True,
        training_flag=False,
        id=id,
        comment=t_comment,
        close_session=True,
        silent=silent,
        tempsave=tempsave,
        save_action_space=save_action_space)

    # import os
    # import shutil
    #
    # path_to_dir = "{}/data/{}/".format(agent.get_dir(), comment)
    #
    # jup_file = "results.ipynb"
    # jup_template = "{}/jupyter_templates/{}".format(PROJECT_DIR, jup_file)
    # dest_file = "{}/".format(path_to_dir)
    # if os.path.exists(jup_template) and not os.path.exists(dest_file):
    #     print("Adding results notebook")
    #     shutil.copyfile(jup_template, dest_file)


def test_run(experiment,
             episodes,
             render=True,
             load_agent=True,
             agent_to_load=None,
             save_data=False,
             max_actions=1000,
             knn=0.1,
             silent=False,
             training_flag=True,
             action_space_config=['auto', 'square', 20000, 10],
             tempsave=False):

    run(experiment=experiment,
        episodes=episodes,
        max_actions=max_actions,
        knn=knn,
        render=render,
        load_agent=load_agent,
        agent_to_load=agent_to_load,
        save_agent=False,
        save_data=save_data,
        training_flag=training_flag,
        comment="test_run",
        silent=silent,
        tempsave=tempsave,
        action_space_config=action_space_config,)
