import torch
import gym
import os
import matplotlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.distributions.categorical import Categorical
from stable_baselines.common import set_global_seeds
from stable_baselines.common import make_vec_env
from stable_baselines import results_plotter



# hyperparams
NUM_WORKERS = 8 # equal to number of logical cores on your machine
TMAX = 5 # number of timesteps before update
TOTAL_STEPS  = 2e6 # total number of timesteps



class Actor(nn.Module): # policy network
    def __init__(self, n_obs, n_actions, hidden_size):
        super(Actor, self).__init__()

        self.n_actions = n_actions
        self.linear1 = nn.Linear(n_obs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_actions)

    def forward(self, input):
        x = F.relu(self.linear1(input))
        out = F.softmax(self.linear2(x), dim=1)
        return out

    def get_action(self, obs):
        logits = self.forward(obs)
        action_dist = Categorical(logits)
        return action_dist.sample()

    def get_policy(self, obs):
        '''Return the (discrete) action distribution of the current policy'''
        logits = self.forward(obs)
        return Categorical(logits)


class Critic(nn.Module): # Neural Network implemented value function used as the baseline
    def __init__(self, n_obs, hidden_size):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(n_obs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

    def forward(self, input):
        '''Take in observations, output one approx. value for each possible output action'''
        x = F.relu(self.linear1(input))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x



def train(env, device, gamma=0.99, entropy_coef=0.01):
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    actor = Actor(obs_dim, n_acts, hidden_size=16).to(device)
    critic = Critic(obs_dim, hidden_size=16).to(device)
    ep_rewards = [0] * NUM_WORKERS
    ep_steps = [0] * NUM_WORKERS
    policy_optim = optim.Adam(actor.parameters(), lr=0.01)
    value_optim = optim.Adam(critic.parameters(), lr=0.01)

    obs = torch.FloatTensor(env.reset()).to(device)
    step = 0
    while step < TOTAL_STEPS:
        acts = []
        log_probs = []
        vals = []
        observs = []
        rewards = []
        entropies = []
        done_masks = []
        for t in range(TMAX):
            action = actor.get_action(obs)
            logp = actor.get_policy(obs).log_prob(action)
            entropy = actor.get_policy(obs).entropy()
            val = critic(obs)
            obs, rews, dones, _ = env.step(action.cpu().numpy()) # NOTE: ENVS WILL RESET AUTOMATICALLY WHEN DONE!
            obs = torch.FloatTensor(obs).to(device)
            step += NUM_WORKERS

            observs.append(obs)
            acts.append(action)
            log_probs.append(logp)
            vals.append(val)
            rewards.append(torch.FloatTensor(rews).to(device))
            entropies.append(entropy)
            done_masks.append(1-dones)

            for i, done in enumerate(dones):
                ep_rewards[i] += rews[i]
                ep_steps[i] += 1
                if done:
                    ep_rewards[i] = 0

        if step % 2000 == 0:
            print("Step: {} Avg. Episode Reward: {}".format(step, np.array(ep_rewards).mean()))

        # perform an update on the policy and value networks
        policy_optim.zero_grad()
        value_optim.zero_grad()
        returns = torch.zeros((TMAX, NUM_WORKERS)).to(device)
        Q = vals[TMAX-1].squeeze() * torch.FloatTensor(done_masks[TMAX-1]).to(device) # terminal states get 0 future reward
        returns[TMAX-1] = Q
        for t in reversed(range(TMAX-1)): # Exclusive so 1st is tmax-2, then tmax-3, ...
            Q = rewards[t] + gamma * Q * torch.FloatTensor(done_masks[t]).to(device)
            returns[t] = Q

        log_probs = torch.cat(log_probs).reshape(TMAX, NUM_WORKERS)
        vals = torch.cat(vals).reshape(TMAX, NUM_WORKERS)
        returns = returns.reshape(TMAX, NUM_WORKERS)
        entropies = torch.cat(entropies).reshape(TMAX, NUM_WORKERS)
        advantage = returns - vals
        policy_loss = -(log_probs * advantage.detach() + entropy_coef*entropies).mean()
        value_loss = advantage.pow(2).mean()

        policy_loss.backward()
        policy_optim.step()
        value_loss.backward()
        value_optim.step()

        # print(ep_rewards)





if __name__ == '__main__':
    env_id = "CartPole-v1"
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    env = make_vec_env('CartPole-v1', n_envs=NUM_WORKERS, monitor_dir=log_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(env, device)
    results_plotter.plot_results([log_dir], TOTAL_STEPS, results_plotter.X_TIMESTEPS, "A2C CartPole-v1")
    plt.show()