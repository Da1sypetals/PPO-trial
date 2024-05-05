import numpy as np
from .train import Episode
import torch
import einops as ein
import random


class PPO:

    def __init__(self, 
                 actor_net, 
                 critic_net, 
                 actor_optim, 
                 critic_optim, 
                 gamma=.99, # discount rate
                 lmbda = 0.95,
                 epsilon=.2, 
                 epoch_per_eposide = 10,
                 relax_prob = .01,
                 device=torch.device('cuda')):
        
        self.actor = actor_net
        self.critic = critic_net
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.relax_prob = relax_prob

        self.device = device

        self.epoch_per_eposide = epoch_per_eposide
        


    def take_action(self, state, return_prob=False):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_distribution = torch.distributions.Categorical(probs)
        action = action_distribution.sample()

        if return_prob:
            action = action.item()
            return action, probs[0, action]                

        return action.item()
    

    def take_action_distribution(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_distribution = torch.distributions.Categorical(probs)
        action = action_distribution.sample()

        return action.item(), probs
    

    def update_episode(self, e: Episode):
        if random.random() > self.relax_prob:
            self._update_episode_clipped(e)
        else:
            self._update_episode_unlimited(e)


    
    def _update_episode_clipped(self, e: Episode):

        # print(e.rewards.size())
        # print((self.gamma * self.critic(e.next_states)).size())
        # print((1 - e.done).size())

        # approximations is made here: use the previous critic.
        timediff_target = e.rewards + (self.gamma * self.critic(e.next_states)).view(-1) * (1 - e.done)
        timediff_delta = timediff_target - self.critic(e.states)
        # advantage is deterministic given episode.
        advantage = self._advantage(timediff_delta).to(self.device)

        for epoch in range(self.epoch_per_eposide):

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            
            # print(e.actions.size())
            # print(self.actor(e.states).size())

            # use NEW actor/critic here.
            log_probs = torch.log(self.actor(e.states).gather(1, ein.rearrange(e.actions, 'n -> n 1')))
            log_probs = ein.rearrange(log_probs, 'n k -> (n k)')

            prob_ratio = torch.exp(log_probs - e.prev_log_probs)
            actor_loss = torch.mean(-torch.min(
                                    prob_ratio * advantage, 
                                    torch.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
                                    ))
            
            # detach here to avoid backprop-ing the original TDtarget multiple times.
            critic_loss = torch.nn.functional.mse_loss(timediff_target.detach(), 
                                                       self.critic(e.states).view(-1), 
                                                       reduction='mean')

            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()


    
    def _update_episode_unlimited(self, e: Episode):
        # print(e.rewards.size())
        # print((self.gamma * self.critic(e.next_states)).size())
        # print((1 - e.done).size())

        # approximations is made here: use the previous critic.
        timediff_target = e.rewards + (self.gamma * self.critic(e.next_states)).view(-1) * (1 - e.done)
        timediff_delta = timediff_target - self.critic(e.states)
        # advantage is deterministic given episode.
        advantage = self._advantage(timediff_delta).to(self.device)

        for epoch in range(self.epoch_per_eposide):

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            
            # print(e.actions.size())
            # print(self.actor(e.states).size())

            # use NEW actor/critic here.
            log_probs = torch.log(self.actor(e.states).gather(1, ein.rearrange(e.actions, 'n -> n 1')))
            log_probs = ein.rearrange(log_probs, 'n k -> (n k)')

            prob_ratio = torch.exp(log_probs - e.prev_log_probs)
            # unclipped
            actor_loss = torch.mean(-prob_ratio * advantage)

            # detach here to avoid backprop-ing the original TDtarget multiple times.
            critic_loss = torch.nn.functional.mse_loss(timediff_target.detach(), 
                                                       self.critic(e.states).view(-1), 
                                                       reduction='mean')

            actor_loss.backward()
            critic_loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()

        

    def _advantage(self, timediff_delta):
        timediff_delta = timediff_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in timediff_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)


        


























