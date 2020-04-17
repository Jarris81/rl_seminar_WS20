import gym
import numpy as np
import pandas as pd
import time

from agents.pg_methods.ddpg_hier import DDPGHAgent
from agents.pg_methods.ddpg import DDPGAgent
from agents.pg_methods.sac2019 import SACAgent
from agents.dqn_methods.dqn import DQNAgent
from agents.pg_methods.td3 import TD3Agent


learning_rate_dqn = 3e-4
gamma_dqn = 0.99

# same buffer for all
buffer_maxlen = 100000

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
gamma_sac = 0.99
tau_sac = 0.01
alpha_sac = 0.2
a_lr_sac = 3e-4
q_lr_sac = 3e-4
p_lr_sac = 3e-4

# td3 params
gamma_td3 = 0.99
tau_td3 = 1e-2
noise_std_td3 = 0.2
bound_td3 = 0.5
delay_step_td3 = 2
buffer_maxlen_td3 = 100000
critic_lr_td3 = 1e-3
actor_lr_td3 = 1e-3

#range_leg = 0.01
#offset_leg = 0.0
buffer_maxlen = 100000

# weight side leg

# ddpg params
gamma_ddpg = 0.99
tau_ddpg = 1e-2
buffer_maxlen_ddpg = 100000
critic_lr_ddpg = 1e-3
actor_lr_ddpg = 1e-3

period = 100

def make_step(t, params):
# Functon used to make a step for the ant in Ant-V3

    # values for two parameter step function
    if len(params) == 2:
        w_r = params[0]
        w_l = params[1]
        range_hip = 0.01
        offset_hip = 0.0
        range_leg = 0.01
        offset_leg = 0.0

    # values for six parameters step function
    elif len(params) == 6:
        w_r = params[0]
        w_l = params[1]
        range_hip = params[2]
        offset_hip = params[3]
        range_leg = params[4]
        offset_leg = params[5]

    # initial with 0
    values = np.zeros(8)

    # front right leg
    values[0] = w_r * -range_hip * np.sin(t * 2 * np.pi * 1/period) - offset_hip
    values[1] = np.abs(w_r) * range_leg * np.sin(t * 2 * np.pi * 1/period) + offset_leg

    # front left leg
    values[2] = w_l * -range_hip * np.sin(t * 2 * np.pi * 1/period) - offset_hip
    values[3] = np.abs(w_l) * -range_leg * np.sin(t * 2 * np.pi * 1/period) + offset_leg

    # back left leg
    values[4] = w_r * range_hip * np.sin(t * 2 * np.pi * 1/period) + offset_hip
    values[5] = np.abs(w_r) * -range_leg * np.sin(t * 2 * np.pi * 1/period) - offset_leg

    # back right leg
    values[6] = w_l * range_hip * np.sin(t * 2 * np.pi * 1/period) + offset_hip
    values[7] = np.abs(w_l) * range_leg * np.sin(t * 2 * np.pi * 1/period) - offset_leg

    return values


if __name__ == "__main__":

    # training values
    # train_max_episodes = 100000
    # train_max_steps = 1000
    # train_batch_size = 32
    # train_save_step = 5000

    agent_name = "dpg6"
    env_name = "Ant-v3"

    env = gym.make(env_name)

    if agent_name == 'dpg2':
        agent = DDPGHAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg, 2, agent_name)
    elif agent_name == "ddpg":
        agent = DDPGAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg)
    elif agent_name == "dpg6":
        agent = DDPGHAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg, 6, agent_name)
    elif agent_name == "sac1":
        agent = SACAgent(env, gamma_sac, tau_ddpg, alpha_sac, q_lr_sac, p_lr_sac, a_lr_sac, buffer_maxlen)
    elif agent_name == "dqn1":
        agent = DQNAgent(env, learning_rate_dqn, gamma_dqn, buffer_maxlen, 3, use_conv=False)
    elif agent_name == "dqn2":
        agent = DQNAgent(env, learning_rate_dqn, gamma_dqn, buffer_maxlen, 9, use_conv=False)
    elif agent_name == "td31":
        agent = TD3Agent(env, gamma_td3, tau_td3, buffer_maxlen, delay_step_td3, noise_std_td3, bound_td3,
                         critic_lr_td3, actor_lr_td3)

    agent.load()

    state = env.reset()

    for step in range(100):

        action_values = agent.get_action(state)
        #action_values = [1, 1]
        print((action_values))
        for t_step in range(period):
            action = make_step(t_step, action_values)
            #print(action)
            env.render()
            next_state, reward, done, _ = env.step(action)
            print(reward)
        state = next_state
    env.close()



