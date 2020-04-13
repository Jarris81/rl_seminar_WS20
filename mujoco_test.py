import gym
import numpy as np

env = gym.make('Ant-v3')
print(env.action_space)
print(env.action_space.sample())
print(env.observation_space.shape)
start_value = [0, 1, 0, 1, 0, -1, 0, -1]

env.reset()
env.step(start_value)

speed = 1/100

range_hip = 0.01
offset_hip = 0.0

range_leg = 0.015
offset_leg = 0.005

# weight side leg
w_l = 1
w_r = 1

for t in range(10000):
    env.render()

    # front right leg
    start_value[0] = w_r * -range_hip * np.sin(t * 2 * np.pi * speed) - offset_hip
    start_value[1] = np.abs(w_r) * range_leg * np.sin(t * 2 * np.pi * speed) + offset_leg

    # front left leg
    start_value[2] = w_l * -range_hip * np.sin(t * 2 * np.pi * speed) - offset_hip
    start_value[3] = np.abs(w_l) * -range_leg * np.sin(t * 2 * np.pi * speed) + offset_leg

    # back left leg
    start_value[4] = w_r * range_hip * np.sin(t * 2 * np.pi * speed) + offset_hip
    start_value[5] = np.abs(w_r) * -range_leg * np.sin(t * 2 * np.pi * speed) - offset_leg

    # back right leg
    start_value[6] = w_l * range_hip * np.sin(t * 2 * np.pi * speed) + offset_hip
    start_value[7] = np.abs(w_l) * range_leg * np.sin(t * 2 * np.pi * speed) - offset_leg

    # # ankles: left front and right back



    #start_value[1] = 0.3 * np.sin((t*2*np.pi)/500)
    #start_value[3] = np.sin(t / 10)

    #start_value[4] = 1 * np.sin((t * 2 * np.pi) / 100)
    #start_value[5] = 1 * - np.sin((t * 2 * np.pi) / 100)
    #start_value[5] = -np.sin(t / 10)
    #start_value[7] = -np.sin(t / 10)
    #start_value[3] = _ * 1e-3
    obs, reward, done, info = env.step(start_value) # take a random action
    #print(info["x_position"], info["y_position"])
    #print(reward)
env.close()
