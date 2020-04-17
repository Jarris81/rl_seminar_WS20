import gym
import numpy as np
import pandas as pd
import os

from agents.pg_methods.ddpg import DDPGAgent
from agents.pg_methods.ddpg_hier import DDPGHAgent
from agents.pg_methods.common.utils import mini_batch_train
from agents.pg_methods.common.utils import mini_batch_train_hier

env_list = ['Ant-v3']

agent_list = [
    'ddpg',
    'dpg2',
    'dpg6'
]

# TODO store parameters in seperate file
period = 100

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
    else:
        print("Error, param size not correct")
        return None

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


def train_all(envs, agents, max_episodes=10000, max_steps=1000, batch_size=32, save_step=1000):

    for env_name in envs:
        # create environment
        env = gym.make(env_name)
        # create new empty dataframe
        df = pd.DataFrame()
        # iterate over each agent, create and start mini batch
        for agent_name in agents:
            if agent_name == 'dpg2':
                agent = DDPGHAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg, 2, agent_name)
                result = np.asarray(
                    mini_batch_train_hier(
                        env, agent, max_episodes, max_steps, batch_size, save_step, make_step, period))
                print(result.shape)
                print(len(result[0]))
            elif agent_name == "ddpg":
                agent = DDPGAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg)
                result = np.asarray(mini_batch_train(env, agent, max_episodes, max_steps, batch_size, save_step))
                print(result.shape)
                print(len(result[0]))
            elif agent_name == "dpg6":
                agent = DDPGHAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg, 6, agent_name)
                result = np.asarray(mini_batch_train_hier(env, agent, max_episodes, max_steps, batch_size, save_step, make_step, period))
            else:
                print("ERROR: agent not in specified")
                break
            # add values to dataframe
            df[agent_name] = result[0]
            df[agent_name + "_time"] = result[1]
            # save values in csv file

            path = "data/" + env_name

            if not os.path.exists(path):
                os.makedirs(path)
            df.to_csv(path + agent_name + ".csv")


# main function
if __name__ == "__main__":

    # define envs ids
    env_list = ['Ant-v3']

    # define agents
    agent_list = [
        'ddpg',
        'dpg2',
        'dpg6'
    ]

    # training values
    train_max_episodes = 5000
    train_max_steps = 500
    train_batch_size = 32
    train_save_step = 5000

    # start training
    train_all(env_list, agent_list,
              max_episodes=train_max_episodes,
              max_steps=train_max_steps,
              batch_size=train_batch_size,
              save_step=train_save_step)






