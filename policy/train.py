import torch
import gymnasium as gym
import einops as ein
import numpy as np
from tqdm import tqdm


class Episode:
    def __init__(self):
        self.reward_list = []
        # observations
        self.state_list = []
        self.next_state_list = []
        self.action_list = []

        # log probability of the event: "the i-th actions is the chosen action"
        # self.log_prob_list = [] # this is computed dynamically
        self.prev_log_prob_list = []

        self.length = None
        

    def collate(self):
        self.states = torch.tensor(np.array(self.state_list), dtype=torch.float32)
        self.next_states = torch.tensor(np.array(self.next_state_list), dtype=torch.float32)
        
        self.prev_log_probs = torch.tensor(self.prev_log_prob_list, dtype=torch.float32).detach()
        self.rewards = torch.tensor(self.reward_list, dtype=torch.float32)
        
        self.actions = torch.tensor(self.action_list, dtype=int)

        # done[i] == this episode terminates at i
        self.done = torch.zeros(self.length, dtype=torch.float32)
        self.done[-1] = 1.0


    def to(self, device: torch.device):
        self.states = self.states.to(device)
        self.next_states = self.next_states.to(device)
        self.actions = self.actions.to(device)
        self.prev_log_probs = self.prev_log_probs.to(device)
        self.rewards = self.rewards.to(device)
        self.done = self.done.to(device)



class OnPolicyTrainer:
    def __init__(self, env: gym.Env, agent):
        self.env = env
        self.agent = agent



    def _sample_episode(self) -> Episode:

        e = Episode()
        done = False
        count = 0
        observation, info = self.env.reset()

        while not done:

            e.state_list.append(observation)

            action, prob = self.agent.take_action(observation, return_prob=True)
            observation, reward, terminated, truncated, info = self.env.step(action)
            # print(action, reward)

            e.reward_list.append(reward)
            e.action_list.append(action)
            e.next_state_list.append(observation)
            e.prev_log_prob_list.append(torch.log(prob).detach())
            # print(f'action: {action}, prob = {prob}')
            

            done = terminated or truncated
            count += 1
        # end while

        e.length = count
        return e

    def train(self, num_episodes):
        for episode in tqdm(range(num_episodes)):
            e = self._sample_episode()
            e.collate()
            e.to(self.agent.device)

            self.agent.update_episode(e)

        


















