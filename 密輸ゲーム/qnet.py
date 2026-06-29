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
        # maskが121次元で入ってくるため、本来のアクションサイズ(6)へ
        mask = mask[:, :6]
        
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

class SmuggleQNet(nn.Module):
    def __init__(self, state_dim, hidden_dim,action_dim=121):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        # 状態価値Vは共通
        self.value = nn.Linear(hidden_dim, 1)
        
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, mask):
        # x: [batch_size, state_dim]
        # mask: [batch_size, 22] (前半11個が実際の額、後半11個が申告額)
        
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.res1(x)
        x = F.relu(x)
        x = self.res2(x)
        x = self.final_ln(x)

        value = self.value(x)  # [batch_size, 1]
        adv = self.advantage(x)
        
        masked_adv = adv * mask
        counts = torch.clamp(mask.sum(dim=-1, keepdim=True), min=1.0)
        mean = masked_adv.sum(dim=-1, keepdim=True) / counts
        q_values = value + (masked_adv - mean)
        
        q_values = torch.where(mask == 1, q_values, torch.tensor(-1e9, device=q_values.device))
        return q_values

class InspectQNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, mask):
        mask = mask[:, :12]
        
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.res1(x)
        x = F.relu(x)
        x = self.res2(x)
        x = self.final_ln(x)

        value = self.value(x)          
        advantage = self.advantage(x)  

        masked_advantage = advantage * mask
        legal_counts = torch.clamp(mask.sum(dim=-1, keepdim=True), min=1.0)
        
        adv_mean = masked_advantage.sum(dim=-1, keepdim=True) / legal_counts
        
        q = value + (masked_advantage - adv_mean)
        q = torch.where(mask == 1, q, torch.tensor(-1e9, device=q.device))

        return q
