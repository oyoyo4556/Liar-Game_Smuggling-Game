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

class SmuggleQNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim)
        self.res2 = ResidualBlock(hidden_dim)
        self.final_ln = nn.LayerNorm(hidden_dim)
        
        # 状態価値Vは共通
        self.value = nn.Linear(hidden_dim, 1)
        
        # 【修正】アドバンテージのヘッドを2つに分ける（各11択）
        self.advantage_actual = nn.Linear(hidden_dim, 11)
        self.advantage_declared = nn.Linear(hidden_dim, 11)

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
        
        mask_actual = mask[:, :11]
        mask_declared = mask[:, 11:]

        # --- ① 実際額（Actual）のQ値計算 ---
        adv_act = self.advantage_actual(x)
        masked_adv_act = adv_act * mask_actual
        counts_act = torch.clamp(mask_actual.sum(dim=-1, keepdim=True), min=1.0)
        mean_act = masked_adv_act.sum(dim=-1, keepdim=True) / counts_act
        q_actual = value + (masked_adv_act - mean_act)
        q_actual = torch.where(mask_actual == 1, q_actual, torch.tensor(-1e9, device=q_actual.device))

        # --- ② 申告額（Declared）のQ値計算 ---
        adv_dec = self.advantage_declared(x)
        masked_adv_dec = adv_dec * mask_declared
        counts_dec = torch.clamp(mask_declared.sum(dim=-1, keepdim=True), min=1.0)
        mean_dec = masked_adv_dec.sum(dim=-1, keepdim=True) / counts_dec
        q_declared = value + (masked_adv_dec - mean_dec)
        q_declared = torch.where(mask_declared == 1, q_declared, torch.tensor(-1e9, device=q_declared.device))

        # 2つのQ値をタプルで返す
        return q_actual, q_declared

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
