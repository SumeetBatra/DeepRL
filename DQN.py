'''My implementation of DQN based on the original paper https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf applied to the
Open AI gym environment 'Cartpole-v0'
Hyperparameter selection taken from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import gym
import numpy as np
import random

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()
episode_durations = []


class QNetwork(nn.Module):
    def __init__(self, n_obs, hidden_size, n_actions):
        super(QNetwork, self).__init__()

        self.linear1 = nn.Linear(n_obs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, n_actions)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_action(self, obs):
        q_vals = self.forward(obs)
        act = np.argmax(q_vals)
        return act

class ReplayMemory(object):
    def __init__(self, N):
        super(ReplayMemory, self).__init__()

        self.capacity = N
        self.memory = []
        self.position = 0

    def save(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def learn(model, optim, replay_mem, batch_size=64, gamma=0.999):
    if len(replay_mem) < batch_size:
        return
    transitions = replay_mem.sample(batch_size) # batch of [state, act, reward, next_state, done]
    batch_states, batch_acts, batch_rews, batch_next_states, batch_terminal = zip(*transitions)
    batch_states = torch.cat(batch_states)
    batch_acts = torch.cat(batch_acts)
    batch_rews = torch.cat(batch_rews)
    batch_next_states = torch.cat(batch_next_states)
    batch_terminal = torch.cat(batch_terminal)



    '''This block is most similar to the paper in that terminal states only get the reward for that state,
    and not the reward + gamma * action-value. However, this performs worse than doing reward + gamma * action-value
    for everything so....idk  ¯\_(ツ)_/¯ '''
    ####################################################################################################################
    # q_vals, target_q_vals = torch.zeros(len(batch_rews)).to(device), torch.zeros(len(batch_rews)).to(device)
    # terminal_inds = torch.where(batch_terminal==True)[0]
    # nonterminal_inds = torch.where(batch_terminal==False)[0]
    #
    # target_q_vals[terminal_inds] = torch.tensor(batch_rews[terminal_inds]) # terminal states only get reward of that state
    # target_q_vals[nonterminal_inds] = torch.tensor(batch_rews[nonterminal_inds]) + gamma * torch.max(model(batch_next_states), 1)[0][nonterminal_inds] # nonterminal states get reward + action-value
    # q_vals = model(torch.tensor(batch_states)).gather(1, batch_acts.unsqueeze(1))
    # loss = F.smooth_l1_loss(q_vals.squeeze(), target_q_vals) # Huber loss, more stable than MSE and performs similarly
    ####################################################################################################################

    q_vals = model(batch_states).gather(1, batch_acts.unsqueeze(1))
    target_q_vals = batch_rews + gamma * torch.max(model(batch_next_states), 1)[0] # I see other people do .detach() here, but this breaks everything for me????
    loss = F.smooth_l1_loss(q_vals.squeeze(), target_q_vals)

    optim.zero_grad()
    loss.backward()
    optim.step()

# Thanks to https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html for this function
def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



def train(env, device, num_episodes=200, eps_start=0.9, eps_end=0.05, eps_decay=200):
    replay_mem = ReplayMemory(N=10000)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    QNet = QNetwork(obs_dim, 16, n_acts).to(device)
    optimizer = optim.Adam(QNet.parameters(), lr=0.01)

    ep = 0
    steps_done = 0 # counter for eps decay
    while ep < num_episodes:
        steps = 0 # counter for steps in one rollout
        obs = env.reset()
        state = torch.FloatTensor([obs]).to(device)

        done = False
        while not done:
            eps_thresh = eps_end + (eps_start - eps_end) * np.exp(-1. * steps_done / eps_decay) # decay eps every time an action is taken
            steps_done += 1
            # select random action with prob. eps, else choose greedily
            p = np.random.uniform()
            if p < eps_thresh:
                action = np.random.randint(0, n_acts)
            else:
                with torch.no_grad():
                    # gym env doesn't like tensor wrapper for some reason
                    action = torch.max(QNet(state), 1)[1].cpu().numpy()[0] # torch.max returns maxval in 1st array, index of maxval in 2nd
            obs, rew, done, _ = env.step(action)
            next_state = torch.FloatTensor([obs]).to(device)
            transition = (state, torch.LongTensor([action]).to(device), torch.FloatTensor([rew]).to(device), next_state, torch.Tensor([done]).to(torch.bool).to(device))
            replay_mem.save(transition)
            learn(model=QNet, optim=optimizer, replay_mem=replay_mem)
            state = next_state
            steps += 1

            # compute target
            if done:
                episode_durations.append(steps)
                print("Episode: {} Steps: {}".format(ep, steps))
                ep += 1









if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(env, device)
    plot_durations()