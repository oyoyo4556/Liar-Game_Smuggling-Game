import random
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from agent.random_agent import Agent
from qnet import VoteQNet,SmuggleQNet,InspectQNet
from processor import Processor
from phase import Phase
from buffer import ReplayBuffer

import random
import numpy as np
import torch
import torch.nn as nn
from agent.random_agent import Agent
from qnet import VoteQNet, SmuggleQNet, InspectQNet
from processor import Processor
from phase import Phase
from buffer import ReplayBuffer

class DQNAgent(Agent):
    def __init__(
      self,
      state_dim,
      device,
      lr=1e-4,
      gamma=0.99,
      epsilon_start=1.0,
      epsilon_end=0.05,
      epsilon_decay=25000,
      target_update_interval=1000,
    ):
        self.device = device
        self.gamma = gamma

        # ε関連
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_steps = 0
        self.total_steps = 0

        self.processor = Processor()
        self.loss_fn = nn.SmoothL1Loss()
        self.target_update_interval = target_update_interval
        self.buffer = ReplayBuffer(buffer_size=100000)

        
        self.vote_net = VoteQNet(state_dim=state_dim, hidden_dim=64, action_dim=6).to(device)
        self.smuggle_net = SmuggleQNet(state_dim=state_dim, hidden_dim=128, action_dim=121).to(device)
        self.inspect_net = InspectQNet(state_dim=state_dim, hidden_dim=64, action_dim=12).to(device)
        
        self.target_vote_net = VoteQNet(state_dim=state_dim, hidden_dim=64, action_dim=6).to(device)
        self.target_smuggle_net = SmuggleQNet(state_dim=state_dim, hidden_dim=128, action_dim=121).to(device)
        self.target_inspect_net = InspectQNet(state_dim=state_dim, hidden_dim=64, action_dim=12).to(device)

        self.vote_optimizer = torch.optim.Adam(self.vote_net.parameters(), lr=lr)
        self.smuggle_optimizer = torch.optim.Adam(self.smuggle_net.parameters(), lr=lr)
        self.inspect_optimizer = torch.optim.Adam(self.inspect_net.parameters(), lr=lr)

    def select_action(self, state, player_id, mask,train_player_id):
        current_epsilon = self.epsilon
        if train_player_id is not None and player_id != train_player_id:
            current_epsilon = 0.0
        # ランダム行動（探索）
        if random.random() < current_epsilon:
            valid_actions = np.where(mask == 1)[0]
            action_idx = int(random.choice(valid_actions))
            
            return action_idx
      
        # モデルによる行動選択
        state_tensor = Processor.observation_tensor(state, player_id).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)

        phase = state.phase

        with torch.no_grad():
            if phase == Phase.VOTE:
                q_values = self.vote_net(state_tensor, mask_tensor)
                action_idx = q_values.argmax(dim=-1).cpu().numpy().item() # item()でPythonの標準整数にする
                return action_idx

            elif phase == Phase.SMUGGLE:
                q_values = self.smuggle_net(state_tensor, mask_tensor)
                action_idx = q_values.argmax(dim=-1).cpu().numpy().item() # 【修正】0〜120の整数値をそのまま返す
                return action_idx 

            elif phase == Phase.INSPECT:
                q_values = self.inspect_net(state_tensor, mask_tensor)
                action_idx = q_values.argmax(dim=-1).cpu().numpy().item()
                return action_idx
                
            else:
                return None
    
    def update(self, batch_size,phase):
        
        if self.buffer.get_len(phase) < batch_size:
            return 0
      
        batch = self.buffer.sample_batch(phase, batch_size, device=self.device)
        states, actions, rewards, next_states, dones, masks, next_masks = batch

        if phase == Phase.VOTE:
            net = self.vote_net
            next_net = self.smuggle_net          
            target_net = self.target_smuggle_net  
            optimizer = self.vote_optimizer
        elif phase == Phase.SMUGGLE:
            net = self.smuggle_net
            next_net = self.inspect_net
            target_net = self.target_inspect_net
            optimizer = self.smuggle_optimizer
        elif phase == Phase.INSPECT:
            net = self.inspect_net
            next_net = self.vote_net
            target_net = self.target_vote_net
            optimizer = self.inspect_optimizer

        q_values = net(states, masks)
        q_a = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 次のフェーズのメインネットで行動選択
            next_q_online = next_net(next_states, next_masks)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # 次のフェーズのターゲットネットで価値評価
            next_q_target = target_net(next_states, next_masks)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)

            # 超重要：Loss爆発防止
            # 次のマスクに「1（合法アクション）」が1つも含まれていないバッチ行を特定
            is_legal_available = (next_masks == 1).sum(dim=1) > 0
            # 有効な行動がない状態の next_q は、-1e9 ではなく強制的に 0.0 にする
            next_q = torch.where(is_legal_available, next_q, torch.tensor(0.0, device=self.device))

            # ターゲットQ値の計算
            target = rewards + self.gamma * (1.0 - dones) * next_q
        
        loss = self.loss_fn(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.update_epsilon()
        self.total_steps += 1 # ターゲット更新用のステップカウント
            
        return loss.item()
        
    def update_epsilon(self):
        self.epsilon_steps += 1
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * np.exp(-1.0 * self.epsilon_steps / self.epsilon_decay)
    
    def copy_to_weight(self):
        self.target_vote_net.load_state_dict(self.vote_net.state_dict())
        self.target_smuggle_net.load_state_dict(self.smuggle_net.state_dict())
        self.target_inspect_net.load_state_dict(self.inspect_net.state_dict())

    def save(self, filepath):
        state = {
            'vote_net': self.vote_net.state_dict(),
            'smuggle_net': self.smuggle_net.state_dict(),
            'inspect_net': self.inspect_net.state_dict(),
            'vote_optimizer': self.vote_optimizer.state_dict(),
            'smuggle_optimizer': self.smuggle_optimizer.state_dict(),
            'inspect_optimizer': self.inspect_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }
        torch.save(state, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath):
        state = torch.load(filepath, map_location=self.device,weights_only=False)
        self.vote_net.load_state_dict(state['vote_net'])
        self.smuggle_net.load_state_dict(state['smuggle_net'])
        self.inspect_net.load_state_dict(state['inspect_net'])
        self.vote_optimizer.load_state_dict(state['vote_optimizer'])
        self.smuggle_optimizer.load_state_dict(state['smuggle_optimizer'])
        self.inspect_optimizer.load_state_dict(state['inspect_optimizer'])
        self.epsilon = state['epsilon']
        self.total_steps = state['total_steps']
        print(f"Agent loaded from {filepath}")
       