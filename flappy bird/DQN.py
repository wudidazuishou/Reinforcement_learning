import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(AttentionModule, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # Layer normalization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5))
        attention_output = torch.matmul(attention_scores, V)
        attention_output = self.dropout(attention_output)  # Dropout for regularization

        return self.layer_norm(x + attention_output)  # Residual connection + LayerNorm

class DQN1(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, dropout=0.1):
        super(DQN1, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.attention = AttentionModule(hidden_dim, dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)  # LayerNorm after first FC
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.layer_norm1(x)  # Normalize activations
        x = self.attention(x)  # Apply attention mechanism
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout before final layer
        x = self.fc3(x)
        return x
    


class DQN2(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, enable_dueling_dqn=False):
        super(DQN2, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.enable_dueling_dqn = enable_dueling_dqn
        
        if self.enable_dueling_dqn:
            self.layer1_value = nn.Linear(state_dim,64)
            self.layer2_value = nn.Linear(64, 128)
            self.layer3_value = nn.Linear(128, self.hidden_dim)
            self.layer4_value = nn.Linear(self.hidden_dim, 1)

            self.layer1_advantage = nn.Linear(state_dim,64)
            self.layer2_advantage = nn.Linear(64, 128)
            self.layer3_advantage = nn.Linear(128, self.hidden_dim)
            self.layer4_advantage = nn.Linear(self.hidden_dim, self.action_dim)
        else:
            self.layer1 = nn.Linear(state_dim,64)
            self.layer2 = nn.Linear(64, 128)
            self.layer3 = nn.Linear(128, self.hidden_dim)
            self.layer4 = nn.Linear(self.hidden_dim, 512) 
            self.layer5 = nn.Linear(512, 512)
            self.layer6 = nn.Linear(512, action_dim)
  
        

    def forward(self, x):
        if self.enable_dueling_dqn:
            x_value = F.relu(self.layer1_value(x))
            x_value = F.relu(self.layer2_value(x_value))
            x_value = F.relu(self.layer3_value(x_value))
            x_value = F.relu(self.layer4_value(x_value))

            x_advantage = F.relu(self.layer1_advantage(x))
            x_advantage = F.relu(self.layer2_advantage(x_advantage))
            x_advantage = F.relu(self.layer3_advantage(x_advantage))
            x_advantage = F.relu(self.layer4_advantage(x_advantage))

            Q = x_value + (x_advantage - torch.mean(x_advantage, dim=1, keepdim=True))
        else:
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            x = F.relu(self.layer4(x))
            x = F.relu(self.layer5(x))
            Q= self.layer6(x)
        return Q