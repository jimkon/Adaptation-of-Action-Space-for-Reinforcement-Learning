# Adaptation of Action Space for Reinforcement Learning

## Summary
Reinforcement Learning is a Machine Learning technique, where a decision-making algorithm, also known as an autonomous agent, interacts with an (unknown) environment by making observations and taking actions, while it is receiving positive or negative rewards at each step based on its performance. During this process, the agent tries to learn an optimal decision-making policy, namely which action selections at each state will help to maximize the expected total reward in the long term. This technique is ideal for optimal control problems, games and many other domains. Many RL architectures use a discrete set of actions to represent a continuous Cartesian action space and the agent is called to select one of these discrete actions at each time step. Usually, this discretization of a continuous action space reduces the ability of the agent to take actions that perform best, since the agent is forced to choose among the discrete actions. There are two alternative solutions to this problem: either increase the density of discrete points, which affects the responsiveness of the agent, or adopt a discretization of variable resolution which adapts to the needs of the problem. In this thesis, we present a method for creating discretizations able to adapt dynamically according to the use of the action space. The proposed adaptive discretization can match automatically a wide variety of different patterns in a few adaptation steps while maintaining a constant number of discrete points. We embed this adaptive discretization method into the action space of a particular Deep RL agent performing in specific environments that require precision. Our adaptive discretizations take advantage of the selective use the agent makes over the action space and adjust the density of the discrete points in the space, giving an increased number of discrete actions and thus higher resolution to regions where it is needed. As a result, the agent’s precision and learning performance is increased, without a significant increase in computational resources. 


## References
My [Diploma Thesis PDF](http://purl.tuc.gr/dl/dias/33218A13-C811-425E-BC8B-8D5226842B6F)

Based on [this paper](https://arxiv.org/abs/1512.07679)

## Code implementations
*  [Deep Reinforcement Learning in Large Discrete Action Spaces](https://github.com/jimkon/Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces)

*  [Adaptive Discretization](https://github.com/jimkon/adaptive-discretization)

*  [Deep Deterministic Policy Gradient](https://github.com/stevenpjg/ddpg-aigym) ([stevenpjg's implementation](https://github.com/stevenpjg))

## Dependecies (pip packages)
*  numpy
*  pyflann
*  gym
*  tensorflow
*  [adiscr](https://github.com/jimkon/adaptive-discretization)

