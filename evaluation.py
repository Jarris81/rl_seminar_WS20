import gym
import pandas as pd
import numpy as np

# agents
from agents.pg_methods.ddpg import DDPGAgent
from agents.pg_methods.sac2019 import SACAgent
from agents.dqn_methods.dqn import DQNAgent
from agents.dqn_methods.common.utils import mini_batch_train_dqn
from agents.pg_methods.common.utils import mini_batch_train
from agents.pg_methods.td3 import TD3Agent

env_list = [
            "Pendulum-v0",
            "MountainCarContinuous-v0",
            "LunarLanderContinuous-v2",
            "BipedalWalker-v3"]

agent_list = ["dqn1", "dqn2", "ddpg", "sac1", "td31"]

# training values
max_episodes = 100000
max_steps = 1000
batch_size = 32
save_step = 5000

# dqn params
learning_rate_dqn = 3e-4
gamma_dqn = 0.99

# same buffer for all
buffer_maxlen = 1000000



# a2c params
gamma_a2c = 0.99
lr_a2c = 1e-4

# ddpg params
gamma_ddpg = 0.99
tau_ddpg = 1e-2
buffer_maxlen_ddpg = 100000
critic_lr_ddpg = 1e-3
actor_lr_ddpg = 1e-3

# SAC 2019 Params
gamma_ddpg = 0.99
tau_ddpg = 0.01
alpha_ddpg = 0.2
a_lr_ddpg = 3e-4
q_lr_ddpg = 3e-4
p_lr_ddpg = 3e-4

# td3 params
gamma_td3 = 0.99
tau_td3 = 1e-2
noise_std_td3 = 0.2
bound_td3 = 0.5
delay_step_td3 = 2
buffer_maxlen_td3 = 100000
critic_lr_td3 = 1e-3
actor_lr_td3 = 1e-3



for env_name in env_list:

    env = gym.make(env_name)

    df = pd.DataFrame()

    for agent_name in agent_list:
        if agent_name == "ddpg":
            agent = DDPGAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg)
            result = np.asarray(mini_batch_train(env, agent, max_episodes, max_steps, batch_size, save_step))
        elif agent_name == "sac1":
            agent = SACAgent(env, gamma_ddpg, tau_ddpg, alpha_ddpg, q_lr_ddpg, p_lr_ddpg, a_lr_ddpg, buffer_maxlen)
            result = np.asarray(mini_batch_train(env, agent, max_episodes, max_steps, batch_size, save_step))
        elif agent_name == "dqn1":
            agent = DQNAgent(env, learning_rate_dqn, gamma_dqn, buffer_maxlen, 3, use_conv=False)
            result = np.asarray(mini_batch_train_dqn(env, agent, max_episodes, max_steps, batch_size, save_step))
        elif agent_name == "dqn2":
            agent = DQNAgent(env, learning_rate_dqn, gamma_dqn, buffer_maxlen, 9, use_conv=False)
            result = np.asarray(mini_batch_train_dqn(env, agent, max_episodes, max_steps, batch_size, save_step))
        elif agent_name == "td31":
            agent = TD3Agent(env, gamma_td3, tau_td3, buffer_maxlen, delay_step_td3, noise_std_td3, bound_td3,
                             critic_lr_td3, actor_lr_td3)
            result = np.asarray(mini_batch_train(env, agent, max_episodes, max_steps, batch_size, save_step))
        else:
            print("ERROR: agent not in specified")
            break

        df[agent_name] = result[0]
        df[agent_name+"_time"] = result[1]

    df.to_csv("data/" + env_name+".csv")
