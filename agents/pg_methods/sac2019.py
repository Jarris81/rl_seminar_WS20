import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions import Normal

#from models import SoftQNetwork, PolicyNetwork
from common.replay_buffers import BasicBuffer
import os

class SACAgent:
  
    def __init__(self, env, gamma, tau, alpha, q_lr, policy_lr, a_lr, buffer_maxlen):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env

        self.path = "model/" + self.env.unwrapped.spec.id + "/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.action_range = [env.action_space.low, env.action_space.high]
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2
        
        # initialize networks 
        self.q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net1 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.target_q_net2 = SoftQNetwork(self.obs_dim, self.action_dim).to(self.device)
        self.policy_net = PolicyNetwork(self.obs_dim, self.action_dim).to(self.device)

        # copy params to target param
        for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            target_param.data.copy_(param)

        for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            target_param.data.copy_(param)

        # initialize optimizers 
        self.q1_optimizer = optim.Adam(self.q_net1.parameters(), lr=q_lr)
        self.q2_optimizer = optim.Adam(self.q_net2.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        # entropy temperature
        self.alpha = alpha
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=a_lr)

        self.replay_buffer = BasicBuffer(buffer_maxlen)

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()
        
        return self.rescale_action(action)
    
    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
            (self.action_range[1] + self.action_range[0]) / 2.0
   
    def update(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)
        
        next_actions, next_log_pi = self.policy_net.sample(next_states)
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)        
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update q networks        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # delayed update for policy network and target q networks
        new_actions, log_pi = self.policy_net.sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions)
            )
            policy_loss = (self.alpha * log_pi - min_q).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
            # target networks
            for target_param, param in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        self.update_step += 1


    def load(self):
        files = [x for x in os.listdir(self.path) if "sac1" in x]

        if len(files) == 0:
            return 0
        else:
            max_episode = max([(int(x[:-4][8:]), i) for i, x in enumerate(files)])
            print(max)
            checkpoint = torch.load(self.path + files[max_episode[1]])
            self.q_net1.load_state_dict(checkpoint['q_net1_state_dict'])
            self.target_q_net1.load_state_dict(checkpoint['q_net1_t_state_dict'])
            self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
            self.q_net2.load_state_dict(checkpoint['q_net2_state_dict'])
            self.target_q_net2.load_state_dict(checkpoint['q_net2_t_state_dict'])
            self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            print("Loading model at episode: ", max_episode[0])
            return max_episode[0]

    def save_model(self, episode):
        torch.save({
            'q_net1_state_dict': self.q_net1.state_dict(),
            'q_net1_t_state_dict': self.target_q_net1.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q_net2_state_dict': self.q_net2.state_dict(),
            'q_net2_t_state_dict': self.target_q_net2.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'policy_net_state_dict':self.policy_net.state_dict(),
            'policy_optimizer_state_dict':self.policy_optimizer.state_dict()
        }, self.path + "sac1_epi" + str(episode) + ".pth")


class ValueNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class SoftQNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi