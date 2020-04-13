import gym
import pandas as pd
import numpy as np
from common.utils import mini_batch_train
from agents.pl_methods.ddpg import DDPGAgent

env = gym.make("Pendulum-v0")

max_episodes = 10
max_steps = 500
batch_size = 32

gamma = 0.99
tau = 1e-2
buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-3

agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size)

episode_rewards = np.asarray(episode_rewards)

