from policy.ppo import PPO
from policy.train import Episode, OnPolicyTrainer
from policy.net import Actor, Critic
from policy.test import Test

import torch

import gymnasium as gym

device = torch.device('cuda')


# env = gym.make("LunarLander-v2")
env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1", render_mode='human')

actor = Actor(obs_dim=4, action_dim=2).to(device)
actor_optim = torch.optim.AdamW(actor.parameters(), lr=0.001)
critic = Critic(obs_dim=4).to(device)
critic_optim = torch.optim.AdamW(critic.parameters(), lr=0.01)


agent = PPO(actor, critic, actor_optim, critic_optim, gamma=1.0)

trainer = OnPolicyTrainer(env, agent)
test = Test(test_env, agent)


for turn in range(100):
    print(f'> turn {turn}')
    trainer.train(num_episodes=100)

    test.run()
    








