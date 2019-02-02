import numpy as np
import gym
from wolp_agent import WolpertingerAgent

env = gym.make("Pendulum-v0")

agent = WolpertingerAgent(env)


for episode in range(10):
    state = env.reset()
    episode_reward = 0
    steps = 0
    done = False

    while not done:

        state, reward, done, _ = env.step(agent.act(state))
        steps += 1
        episode_reward += reward

        env.render()
        if done:
            print("Episode {} finished after {} steps with total reward {}".format(
            episode, steps, episode_reward
            ))
