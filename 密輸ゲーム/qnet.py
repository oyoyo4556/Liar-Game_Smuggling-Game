import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 2 * dim)
        
        self.ln2 = nn.LayerNorm(2 * dim)
        self.fc2 = nn.Linear(2 * dim, dim)

    def forward(self, x):
        residual = x
        
        # 1層目: Norm -> Linear -> ReLU
        out = self.ln1(x)
        out = self.fc1(out)
        out = F.relu(out)
        
        # 2層目: Norm -> Linear
        out = self.ln2(out)
        out = self.fc2(out)
        
        # 残差結合
        return out + residual


class VoteQNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, mask):
        # x: [batch_size, state_dim]
        # mask: [batch_size, action_dim] (1=合法, 0=不合法)
        
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.res1(x)
        x = self.final_ln(x)

        value = self.value(x)          # [batch_size, 1]
        advantage = self.advantage(x)  # [batch_size, action_dim]

        masked_advantage = advantage * mask
        legal_counts = torch.clamp(mask.sum(dim=-1, keepdim=True), min=1.0)
        
        adv_mean = masked_advantage.sum(dim=-1, keepdim=True) / legal_counts
        
        q = value + (masked_advantage - adv_mean)
        q = torch.where(mask == 1, q, torch.tensor(-1e9, device=q.device))

        return q
