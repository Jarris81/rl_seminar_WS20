import gym
import numpy as np
import pandas as pd
import time

from agents.pg_methods.ddpg import DDPGAgent
from agents.pg_methods.ddpg_hier import DDPGHAgent
from agents.pg_methods.common.utils import mini_batch_train

env_list = ['Ant-v3']

agent_list = [
    #'ddpg',
    'dpg2',
    'dpg6'
]

period = 100

#range_hip = 0.01
#offset_hip = 0.0

#range_leg = 0.01
#offset_leg = 0.0
buffer_maxlen = 100000

# weight side leg
#w_l = 1
#w_r = 1

# training values
max_episodes = 10000
max_steps = 1000
batch_size = 32
save_step = 5000

# ddpg params
gamma_ddpg = 0.99
tau_ddpg = 1e-2
buffer_maxlen_ddpg = 100000
critic_lr_ddpg = 1e-3
actor_lr_ddpg = 1e-3

def make_step(t, params):

    if len(params) == 2:
        w_r = params[0]
        w_l = params[1]
        range_hip = 0.01
        offset_hip = 0.0
        range_leg = 0.01
        offset_leg = 0.0

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


def mini_batch_train_hier(env, agent, max_episodes, max_steps, batch_size, save_step):
    episode_rewards = []

    start_episode = agent.load()

    time_stamp = []
    time_start = time.time()

    for episode in range(start_episode, max_episodes+1):
        state = env.reset()
        episode_reward = 0
        step = 0

        while step < max_steps:
            initial_state = state
            action_params = agent.get_action(initial_state)
            accumulated_reward = 0
            next_state = None
            done = False

            for t_step in range(0, period):
                step += 1
                action = make_step(t_step, action_params)
                next_state, reward, done, _ = env.step(action)
                accumulated_reward += reward
                if done:
                    break

            agent.replay_buffer.push(initial_state, action_params, accumulated_reward, next_state, done)
            episode_reward += accumulated_reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                time_stamp.append(time.time() - time_start)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

        if not episode % save_step and episode:
            print("Saving model at Episode ", episode)
            agent.save_model(episode)

    return episode_rewards, time_stamp

for env_name in env_list:

    env = gym.make(env_name)

    df = pd.DataFrame()
    for agent_name in agent_list:
        if agent_name == 'dpg2':
            agent = DDPGHAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg, 2, agent_name)
            result = np.asarray(mini_batch_train_hier(env, agent, max_episodes, max_steps, batch_size, save_step))
            print(result.shape)
            print(len(result[0]))
        elif agent_name == "ddpg":
            agent = DDPGAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg)
            result = np.asarray(mini_batch_train(env, agent, max_episodes, max_steps, batch_size, save_step))
            print(result.shape)
            print(len(result[0]))
        elif agent_name == "dpg6":
            agent = DDPGHAgent(env, gamma_ddpg, tau_ddpg, buffer_maxlen, critic_lr_ddpg, actor_lr_ddpg, 6, agent_name)
            result = np.asarray(mini_batch_train_hier(env, agent, max_episodes, max_steps, batch_size, save_step))
        else:
            print("ERROR: agent not in specified")
            break

        df[agent_name] = result[0]
        df[agent_name + "_time"] = result[1]

        df.to_csv("data/" + env_name + agent_name + ".csv")








