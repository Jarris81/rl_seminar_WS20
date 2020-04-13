import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from common.replay_buffers import BasicBuffer

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import gym
import os

class DQNAgent:

    def __init__(self, env, learning_rate, gamma, buffer_size, discrete_num, use_conv=True):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)

        self.path = "model/" + self.env.unwrapped.spec.id + "/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.discrete_num = discrete_num

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.isContinuous = False
        # check if discrete,
        if type(env.action_space) == gym.spaces.Box:
            self.isContinuous = True
            self.action_dim = env.action_space.shape[0]
            print(self.action_dim)

            low = np.asarray(env.action_space.low)
            high = np.asarray(env.action_space.high)

            steps_size = (high - low) / self.discrete_num

            steps = np.arange(self.discrete_num) + 0.5
            steps = steps.reshape((self.discrete_num, 1))
            #print(steps)
            values = steps * steps_size + low
            #print(values)
            list_values = np.split(values, self.action_dim, axis=1)
            #print(list_values)
            value_comb = np.array(np.meshgrid(*list_values)).T.reshape(-1, self.action_dim)

            self.mapping = {key: value for (key, value) in enumerate(value_comb)}

        else:
            self.action_dim = env.action_space.n
            self.mapping = None

        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(env.observation_space.shape, self.action_dim).to(self.device)
        else:
            self.model = DQN(env.observation_space.shape, self.action_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state, eps=0.20):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        if np.random.randn() < eps:
            return np.random.randint(self.action_dim), self.mapping


        return action, self.mapping

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))

        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load(self):
        files = [x for x in os.listdir(self.path) if "dqn"+str(self.discrete_num) in x]

        if len(files) == 0:
            return 0
        else:
            max_episode = max([(int(x[:-4][8:]), i) for i, x in enumerate(files)])

            checkpoint = torch.load(self.path + files[max_episode[1]])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print("Loading model at episode: ", max_episode[0])
            return max_episode[0]


    def save_model(self, episode):

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.path+"dqn"+str(self.discrete_num)+"_"+"epi"+str(episode)+".pth")


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_input_dim = self.feature_size()

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        features = self.conv_net(state)
        features = features.view(features.size(0), -1)
        qvals = self.fc(features)
        return qvals

    def feature_size(self):
        return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals