import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .random_agent import Agent
from ..qnet import VoteQNet,SmuggleQNet,InspectQNet
from ..processor import Processor
from ..phase import Phase

class DQNAgent(Agent):
    def __init__(
      self,
      state_dim,
      device,
      lr=1e-4,
      gamma=0.99,
      epsilon_start=1.0,
      epsilon_end=0.05,
      epsilon_decay=50000,
      target_update_interval=1000,
    ):

      self.device=device
      self.gamma=gamma

      #ε関連
      self.epsilon=epsilon_start
      self.epsilon_start=epsilon_start
      self.epsilon_end=epsilon_end
      self.epsilon_decay=epsilon_decay
      self.epsilon_steps = 0
      self.total_steps = 0

      self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

      self.target_update_interval = target_update_interval

      self.processor = Processor()

      self.vote_net = VoteQNet(state_dim=state_dim, hidden_dim=64, action_dim=6).to(device)
      self.smuggle_net = SmuggleQNet(state_dim=state_dim, hidden_dim=64).to(device)
      self.inspect_net = InspectQNet(state_dim=state_dim, hidden_dim=64, action_dim=12).to(device)

    def select_action(self,state,player_id,mask):
      
      state_tensor = self.processor.observation_tensor(state, player_id).unsqueeze(0).to(self.device)
      mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)

      phase = state.phase

      with torch.no_grad():
        if phase == Phase.VOTE:
            q_values = self.vote_net(state_tensor, mask_tensor)
            # argmaxをそのままNumpyの1次元スカラー配列（shape: ()）として取り出す
            action_idx = q_values.argmax(dim=-1).cpu().numpy()
            return action_idx  # 例: array(3, dtype=int64)

        elif phase == Phase.SMUGGLE:
            q_actual, q_declared = self.smuggle_net(state_tensor, mask_tensor)
                
            # それぞれの最大Q値のインデックスをTensorのまま取得
            act_idx = q_actual.argmax(dim=-1)
            dec_idx = q_declared.argmax(dim=-1)
                
            # 【Numpyに統一】2つのインデックスを結合し、サイズ2の1次元配列（shape: (2,)）にする
            action_array = torch.stack([act_idx, dec_idx]).cpu().numpy()
            return action_array  

        elif phase == Phase.INSPECT:
            q_values = self.inspect_net(state_tensor, mask_tensor)
            action_idx = q_values.argmax(dim=-1).cpu().numpy()
            return action_idx  # 例: array(0, dtype=int64)
            
        else:
            return None