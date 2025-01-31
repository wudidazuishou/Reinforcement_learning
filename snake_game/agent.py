import torch
import matplotlib
from snake_game.game import SnakeGameAI, Direction, Point
from collections import deque
import numpy as np


import random

'''
Whole training process:

1. Get Current State (11 values)

2. Get action from current state (using learnable model)


P(next_state, reward| state, action), the transition prob matrix usually told by game env:

3. reward, game_over, score= game.play_step(action)
   
4. Get next state (11 values)

5. Save the data for (state, reward, score, game_over..)

6. model.train, in order to maximize the reward by tunning parameter (change the behaviour of action | state)

'''


MAX_memory=100000
Batch_sz=1000
lr=0.001

class Agent:

    def __init__(self):
        self.n_games=0
        self.epsilon=0
        self.dis_factor=0
        self.model=None  ## TODO need to implement the structure of model
        self.trainer=None ## TODO need to implement the trainer using lightening
        self.memory= deque(maxlen=MAX_memory)

    def get_state(self, game):
        head=game.snake[0]

        ## Depending on the value of head, we need to decide what is all the directions to the head

        point_l=Point(head.x-20, head.y)
        point_r=Point(head.x+20,head.y)
        point_u=Point(head.x,head.y+20)
        point_d=Point(head.x,head.y-20)

        dir_l= game.direction==Direction.LEFT
        dir_r= game.direction==Direction.RIGHT
        dir_u= game.direction==Direction.UP
        dir_d= game.direction==Direction.DOWN

        state=[  ## Straight danger
                 (dir_r and game._is_collision(point_r)) 
                 or (dir_l and game._is_collision(point_l)) 
                 or (dir_u and game._is_collision(point_u)) 
                 or (dir_d and game._is_collision(point_d)),
                 ## Right danger
                 (dir_r and game._is_collision(point_d)) 
                 or (dir_u and game._is_collision(point_r)) 
                 or (dir_l and game._is_collision(point_u)) 
                 or (dir_d and game._is_collision(point_r)),
                 ## Left danger
                 (dir_r and game._is_collision(point_u)) 
                 or (dir_u and game._is_collision(point_l)) 
                 or (dir_l and game._is_collision(point_d)) 
                 or (dir_d and game._is_collision(point_r)),
                 ## Move direction
                 dir_l, dir_r, dir_u, dir_d,
                 ## Food location
                 (game.food.x< head.x), (game.food.x> head.x), (game.food.y< head.y), (game.food.y> head.y)
                 ]
        return np.array(state, dtype=int)

    def get_action(self, state):
        ## if we play more games, the smaller the epsilon will be ( less likely we do random action)
        ## if we play less games, the larger the epsilon will be  (more likely we do random action)

        ## this step is to control how we would like to explore the action space
        self.epsilon=80-self.n_games 
        
        final_move=[0,0,0]

        if random.randint(0,200) < self.epsilon:
            ## if rare event, random search space
            idx=random.randint(0,2)
            final_move[idx]=1
        else:
            ## if common event, then we use our model
            state0=torch.tensor(state0, dtype=torch.float)
            pred=self.model(state0)
            idx=torch.argmax(pred)
            final_move[idx]=1

        return(final_move)


    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def train_short_time(self,state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_time(self):
        if len(self.memory) > Batch_sz:
            sample=random.sample(self.memory, Batch_sz)
        else:
            sample= self.memory
        
        states, actions,rewards, next_states, dones= zip(*sample)
        self.trainer.train_step(states, actions,rewards, next_states, dones)


def train():
    plot_scores=[]
    plot_mean_scores=[]
    game=SnakeGameAI()
    agent=Agent()
    total_score=0
    record=0

    while True:
        state= agent.get_state(game)
        action=agent.get_action(state)

        reward, game_over, score= game.play_step(action)
        new_state=agent.get_state(game)

        ## train the model
        agent.train_short_time(state, action, reward, new_state, game_over)
        agent.remember(state, action, reward, new_state, game_over)
        
        ## if game is over, we would like to start the new game
        ## The purpose of train long time is to read from the previous completed game and learn 
        if game_over:
          game.reset()
          ## do one more game
          agent.n_games += 1
          ## agent
          agent.train_long_time()

          if score > record:
              record=score ## update the record to the highest score so far.
          print("Games:", agent.n_games, 'Score:', score, "Record:", record)
              








