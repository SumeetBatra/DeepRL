'''Implementation of the REINFORCE algorithm applied to the the Open AI gym environment 'Cartpole-v0
Credit to https://towardsdatascience.com/learning-reinforcement-learning-reinforce-with-pytorch-5e8ad7fc7da0
for the algorithm and inspiration
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
import gym
import numpy as np


class PolicyNet(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_size):
        super(PolicyNet, self).__init__()

        self.n_actions = n_actions
        self.linear1 = nn.Linear(n_obs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_actions)

    def forward(self, input):
        x = F.relu(self.linear1(input))
        out = F.softmax(self.linear2(x))
        return out

    def get_action(self, obs):
        logits = self.forward(obs)
        action_dist = Categorical(logits=logits)
        return action_dist.sample().item()

    def get_policy(self, obs):
        logits = self.forward(obs)
        return Categorical(logits=logits)

def discount_rewards(rewards, gamma=0.99):
    '''Infinite horizon discounted return
        I believe this is a state-value function V(s) ???
    '''
    r = np.array([(gamma**i) * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1] # reverse array, do cumsum, then revert back to original order
    return r - r.mean() # for stability

def get_loss(acts, obs, rewards, policy_net):
    '''
    :param acts: (batch) actions taken during the rollout
    :param obs: (batch) observations made during the rollout
    :param rewards: (batch) discounted rewards from the rolout
    :param policy_net: policy neural network
    '''
    logp = policy_net.get_policy(obs).log_prob(acts)
    loss = -(rewards * logp).mean()
    return loss

def reinforce(env, policy_net, device,  epoch=0, num_episodes=5000, batch_size=128, gamma=0.99):
    # set up lists to hold results of a rollout
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_obs = []
    batch_counter = 1

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    action_space = np.arange(env.action_space.n)
    ep = 0
    while ep < num_episodes:
        obs = env.reset()
        observations = []
        rewards = []
        actions = []
        done = False
        finished_rendering_this_epoch = False
        while not done:
            if (not finished_rendering_this_epoch):
                # env.render()
                pass

            observations.append(obs)
            with torch.no_grad():
                action = policy_net.get_action(torch.as_tensor(obs, dtype=torch.float32).to(device))
            obs, rew, done, _ = env.step(action)

            rewards.append(rew)
            actions.append(action)

            if done:
                finished_rendering_this_epoch = True
                batch_rewards.extend(discount_rewards(rewards, gamma))
                batch_obs.extend(observations)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # if batch is complete, update the network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    obs_tensor = torch.FloatTensor(batch_obs).to(device)
                    reward_tensor = torch.FloatTensor(batch_rewards).to(device)
                    action_tensor = torch.LongTensor(batch_actions).to(device) # actions are used as indices, must be LongTensor
                    loss = get_loss(action_tensor, obs_tensor, reward_tensor, policy_net)
                    loss.backward() # calculate gradients
                    optimizer.step() # apply gradients
                    # reset counters
                    batch_rewards = []
                    batch_actions = []
                    batch_obs = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                # print(ep)
                print("\rEpisode: {} Average of last 100 Episodes: {:.2f}".format((ep + 1), avg_rewards), end="")
                ep += 1

    return avg_rewards







if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy_net = PolicyNet(n_obs=env.observation_space.shape[0], n_actions=env.action_space.n, hidden_size=16).to(device)
    epochs = 1
    for i in range(epochs):
     avg_rewards = reinforce(env, policy_net, device, epoch=i+1)
     print("\nEpoch: {} Average reward: {:.2f}".format(i + 1, avg_rewards), end="")
