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
from DQN import DQN1, DQN2
from experience_replay import ReplayMemory
from collections import deque
import itertools
import yaml
import random
import matplotlib.pyplot as plt


# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Agent():
    def __init__(self, hyperparameter_set):
        with open('hyperparameter.yaml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.env_id = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.gamma = hyperparameters['gamma']
        self.loss_fn = nn.MSELoss()
        self.lr=hyperparameters['lr']
        self.network_update_frequency=hyperparameters['network_update_frequency']
        self.maximum_reward=hyperparameters['maximum_reward']
        self.hidden_dim=hyperparameters['hidden_dim']
        self.enable_double_dqn=hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn=hyperparameters['enable_dueling_dqn']


        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')




    def play(self, num_episodes=20, is_render=False):
        env=gymnasium.make(self.env_id, render_mode="human" if is_render else None)
        state_dim=env.observation_space.shape[0]
        action_dim=env.action_space.n

        ## load the pre_trained best model
        policy_qnet = DQN2(state_dim=state_dim, action_dim=action_dim, hidden_dim=self.hidden_dim, enable_dueling_dqn=self.enable_dueling_dqn).to(device)
        policy_qnet.load_state_dict(torch.load(self.MODEL_FILE))  # Load trained weights
        policy_qnet.eval()  # Set model to evaluation mode
        

        

        for episode in range(num_episodes):
            state, _ = env.reset()
            terminated=False
            episode_reward=0

            while not terminated:
                state=torch.tensor(state, dtype=torch.float).to(device)
                with torch.no_grad():
                    action = policy_qnet(state.unsqueeze(0)).squeeze().argmax()
                new_state, reward, terminated, _, info = env.step(action.item())
                episode_reward += reward
                state=new_state
            print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
        env.close()
        

        
    def run(self, is_train=True, is_render=False, use_saved_model=False):
        env = gymnasium.make(self.env_id, render_mode="human" if is_render else None)
        
        state_dim=env.observation_space.shape[0]
        action_dim=env.action_space.n
        
        ## if training mode is on, we define the model with opimizer and target net 
        ## if not use saved model, we create a new model
        if is_train:
            policy_qnet=DQN2(state_dim=state_dim, action_dim=action_dim, hidden_dim=self.hidden_dim, enable_dueling_dqn=self.enable_dueling_dqn).to(device)
            if use_saved_model==False:
                print("Creating new model")
            else:
                print("Loading saved model")
                policy_qnet.load_state_dict(torch.load(self.MODEL_FILE))
        

            self.optimizer = torch.optim.Adam(policy_qnet.parameters(), lr=self.lr)
            memory=ReplayMemory(10000)
            target_qnet=DQN2(state_dim=state_dim, action_dim=action_dim, hidden_dim=self.hidden_dim, enable_dueling_dqn=self.enable_dueling_dqn).to(device)
            target_qnet.load_state_dict(policy_qnet.state_dict())
            
            best_reward=-99999


        reward_per_episode=[]
        epsilon_per_episode=[]
        epsilon=self.epsilon_init
        step = 0 ## step records how many episodes we have so far

        for episode in range(1,200000+1):
            ## Start an episode:
            state, _ = env.reset()
            terminated=False
            episode_reward=0

            
            
            ## Within an episode and if the current reward is high enough, we stop
            while (not terminated and episode_reward < self.maximum_reward):
                state=torch.tensor(state, dtype=torch.float).to(device)

                # Episilon-greedy
                if is_train and random.random() < epsilon:
                   action = env.action_space.sample()
                   action=torch.tensor(action, dtype=torch.int64,device=device)
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
                    step += 1

                if is_train and episode_reward > best_reward:
                    best_reward=episode_reward
                    torch.save(policy_qnet.state_dict(), self.MODEL_FILE)
                    with open(self.LOG_FILE, "a") as log_file:
                        log_file.write(f"Episode {episode}: New best reward {best_reward} - Model saved\n")
                    print(f"New best reward {best_reward} in Episode {episode} - Model saved")


                state=new_state

        
            reward_per_episode.append(episode_reward)
            epsilon=max(epsilon*self.epsilon_decay, self.epsilon_min)
            epsilon_per_episode.append(epsilon)

            if episode % 20 == 0:
                self.save_graph(reward_per_episode)




            if len(memory)>self.mini_batch_size and is_train:
                mini_batch=memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_qnet, target_qnet, episode)

                if step > self.network_update_frequency:
                    target_qnet.load_state_dict(policy_qnet.state_dict())
                    step=0
    
    ## plot the rewards that we find so far
    def save_graph(self, reward_per_episode):

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(reward_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(reward_per_episode[max(0, x-99):(x+1)])

        fig, ax = plt.subplots(figsize=(8, 5))
    
        # Plot Mean Rewards per Episode
        ax.plot(mean_rewards, label="Mean Reward", color="blue")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Mean Rewards")
        ax.set_title("Training Progress")
        ax.legend()
        ax.grid(True)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_qnet, target_qnet, episode):

        '''
        ## This is only for optimize for each observation (stocastic gradient descent)

        for state, action, new_state, reward, terminated in mini_batch:
            
            if terminated:
                target= reward
            else:
                with torch.no_grad():
                    target= reward+ self.gamma * (target_qnet(new_state).max())
            
            q_value= policy_qnet(state)[action.item()]

            self.loss=self.loss_fn(q_value, target)

            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        '''

    
        ## this runs for batch with mean squared error, so only one loss scale presented for the mini batch
        state, action, new_state, reward, terminated= zip(*mini_batch)

        state=torch.stack(state)
        action=torch.stack(action)
        new_state=torch.stack(new_state)
        reward=torch.stack(reward)
        terminated=torch.tensor(terminated,dtype=torch.int64, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_action= policy_qnet(new_state).max(dim=1)[1]
                target= reward+ self.gamma * (target_qnet(new_state).gather(1, best_action.unsqueeze(1)).squeeze())*(1-terminated)
            else:
                target= reward+ self.gamma * (target_qnet(new_state).max(dim=1)[0])*(1-terminated)
        
        q_value= policy_qnet(state).gather(1, action.unsqueeze(1)).squeeze()

        self.loss=self.loss_fn(q_value, target)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        if episode % 100 == 0:
           print(f"loss: {self.loss}, episode: {episode}")



        


if __name__ == "__main__":
    agent = Agent("cartpole1")
    agent.run(is_train=True, is_render=False)
