import os
import sys

# Set the working directory
os.chdir("D:\\RL_Finance\\RL sutton book\\flappy bird")
sys.path.append("D:/RL_Finance/RL sutton book/flappy_bird")


import flappy_bird_gymnasium
import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DQN import DQN
from experience_replay import ReplayMemory
from collections import deque
import itertools
import yaml
import random


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, hyperparameter_set):
        with open('hyperparameter.yaml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']

        
    def run(self, is_train=True, is_render=False):
        env = gymnasium.make("CartPole-v1", render_mode="human" if is_render else None)
        
        state_dim=env.observation_space.shape[0]
        action_dim=env.action_space.n

        policy_qnet=DQN(state_dim=state_dim, action_dim=action_dim, hidden_dim=128).to(device)
        if is_train:
            memory=ReplayMemory(10000)

        reward_per_episode=[]
        epsilon_per_episode=[]
        epsilon=self.epsilon_init
        
        
        
        for episode in itertools.count():

            state, _ = env.reset()
            terminated=False
            episode_reward=0
            
            ## Within an episode
            while not terminated:
                state=torch.tensor(state, dtype=torch.float).to(device)

                # Next action:
                if is_train and random.random() < epsilon:
                   action = env.action_space.sample()
                   action=torch.tensor(action, dtype=torch.int32,device=device)
                else:
                    with torch.no_grad():
                        action = policy_qnet(state.unsqueeze(0)).squeeze().argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                
                
                episode_reward += reward

                new_state=torch.tensor(new_state, dtype=torch.float,device=device)
                reward=torch.tensor(reward, dtype=torch.float,device=device)

                if is_train:
                    memory.append((state, action, new_state, reward, terminated))
                


                state=new_state


        reward_per_episode=reward_per_episode.append(episode_reward)
        epsilon=max(epsilon*self.epsilon_decay, self.epsilon_min)
        epsilon_per_episode.append(epsilon)


if __name__ == "__main__":
    agent = Agent("cartpole1")
    agent.run(is_train=True, is_render=False)
