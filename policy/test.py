from .ppo import PPO
from .train import Episode, OnPolicyTrainer
from .net import Actor, Critic
import torch
import gymnasium as gym
import time


class Test:
    def __init__(self, env: gym.Env, agent):
        self.env = env
        self.agent = agent


    def run(self, interval=False):
        done = False
        observation, info = self.env.reset()

        while not done:

            action, dist = self.agent.take_action_distribution(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)

            print(f'action: {action}, distribution = {dist.tolist()}')

            done = terminated or truncated

            if interval:
                time.sleep(interval)








