import torch
import gym

from PPO import ActorCritic

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env_id = 'LunarLanderContinuous-v2'
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.shape[0]

    model_dir = './saved_models/'
    model_name = 'ppo_lunarlander.pt'
    model = ActorCritic(obs_dim, n_acts).to(device)
    model.load_state_dict(torch.load(model_dir + model_name))

    obs = torch.FloatTensor(env.reset()).to(device)
    done = False
    while not done:
        means, _ = model(obs)
        action = model.get_action(means)
        obs, rewards, dones, info = env.step(action.cpu().detach().numpy())
        obs = torch.FloatTensor(obs).to(device)
        env.render()

        if done:
            obs = env.reset()