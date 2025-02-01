import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os




## Notice that the linear model would have an output like [x1, x2, x3]
## which is a approximation of action-value function Q corresponding to 3 actions


class Linear_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1=nn.Linear(input_size, hidden_size)
        self.layer2=nn.Linear(hidden_size, hidden_size)
        self.layer3=nn.Linear(hidden_size, output_size)


    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=self.layer3(x)

        return x
    
    def save(self, file_path='best_model.pth'):
        model_folder_path='trained_model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        model_path=os.path.join(model_folder_path,file_path)
        torch.save(self.state_dict(),model_path)
    

    

class RL_Trainer:
    def __init__(self, model, lr, gamma):
        self.model=model
        self.lr=lr
        self.gamma=gamma
        self.optimizer=optim.Adam(self.model.parameters(), lr= self.lr)
        self.loss=nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        
        ## for short time training, all the inputs should have batch size 1
        ## for long time training, all the inputs should have n as batch size

        # so we need to unsqueeze the input into batch size (1, input)

        state= torch.tensor(state, dtype=torch.float)
        action= torch.tensor(action, dtype=torch.float)
        next_state= torch.tensor(next_state, dtype=torch.float)
        reward= torch.tensor(reward, dtype=torch.float)
        
        ## transfer them into batch data (1,input)
        if len(state.shape) == 1:
           state=torch.unsqueeze(state, 0)
           action=torch.unsqueeze(action, 0)
           next_state=torch.unsqueeze(next_state, 0)
           reward=torch.unsqueeze(reward, 0)
           done= (done, )

        ## pred is the action-value function at current state

        Q_old=self.model(state) 

        target=Q_old.clone()
        
        ## we need to iterate through all the state
        for idx in range(len(done)):
            ## so we define the value for Q_new to be: R + \gamma * max(Q(next_state))
            Q_new=reward[idx]
            if not done[idx]:
                Q_new=reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            ## to update Q value, we should reassign the new Q value to the current Q value for current action

            target[idx][torch.argmax(action).item()]= Q_new
        
        self.optimizer.zero_grad()

        loss=self.loss(target, Q_old)
        loss.backward()

        self.optimizer.step()

        

            


        






