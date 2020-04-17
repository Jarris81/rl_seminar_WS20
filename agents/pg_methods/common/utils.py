import numpy as np
import math
import gym
import torch
import time


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size, save_step):
    episode_rewards = []

    start_episode = agent.load()

    time_stamp = []
    time_start = time.time()

    for episode in range(start_episode, max_episodes+1):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                time_stamp.append(time.time() - time_start)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

        if not episode % save_step and episode:
            print("Saving model at Episode ", episode)
            agent.save_model(episode)

    return episode_rewards, time_stamp

def mini_batch_train_frames(env, agent, max_frames, batch_size):
    episode_rewards = []
    state = env.reset()
    episode_reward = 0

    for frame in range(max_frames):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        episode_reward += reward

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)   

        if done:
            episode_rewards.append(episode_reward)
            print("Frame " + str(frame) + ": " + str(episode_reward))
            state = env.reset()
            episode_reward = 0
        
        state = next_state
            
    return episode_rewards

# process episode rewards for multiple trials
def process_episode_rewards(many_episode_rewards):
    minimum = [np.min(episode_reward) for episode_reward in episode_rewards]
    maximum = [np.max(episode_reward) for episode_reward in episode_rewards]
    mean = [np.mean(episode_reward) for episode_reward in episode_rewards]

    return minimum, maximum, mean

def mini_batch_train_hier(env, agent, max_episodes, max_steps, batch_size, save_step, model_func, period):
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
                action = model_func(t_step, action_params)
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
