from game import SnakeGameAI, Point, Direction
import torch
from model import Linear_model
import numpy as np



class Evaluator:
    def __init__(self, model_path='model.pth'):
        self.model = Linear_model(11, 256, 3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set model to evaluation mode
        self.game = SnakeGameAI()
    
    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            dir_l, dir_r, dir_u, dir_d,
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def evaluate(self, num_games=10):
        scores = []
        for _ in range(num_games):
            self.game.reset()
            done = False
            while not done:
                state = self.get_state(self.game)
                final_move = self.get_action(state)
                _ , done, score = self.game.play_step(final_move)
            
            scores.append(score)
            print(f'Game {_ + 1}: Score = {score}')
        
        avg_score = sum(scores) / len(scores)
        print(f'Average Score over {num_games} games: {avg_score}')

evaluator = Evaluator(model_path='trained_model/best_model.pth')
evaluator.evaluate(num_games=10)