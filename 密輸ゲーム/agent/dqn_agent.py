import random
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from .random_agent import Agent
from ..qnet import VoteQNet,SmuggleQNet,InspectQNet
from ..processor import Processor
from ..phase import Phase
from ..buffer import ReplayBuffer

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

      self.vote_optimizer = torch.optim.Adam(self.vote_net.parameters(), lr=lr)
      self.smuggle_optimizer = torch.optim.Adam(self.smuggle_net.parameters(),lr=lr)
      self.inspect_optimizer = torch.optim.Adam(self.inspect_net.parameters(),lr=lr)

      self.target_update_interval = target_update_interval

      self.processor = Processor()

      self.vote_net = VoteQNet(state_dim=state_dim, hidden_dim=64, action_dim=6).to(device)
      self.smuggle_net = SmuggleQNet(state_dim=state_dim, hidden_dim=64,action_dim=121).to(device)
      self.inspect_net = InspectQNet(state_dim=state_dim, hidden_dim=64, action_dim=12).to(device)
      self.target_vote_net = VoteQNet(state_dim=state_dim, hidden_dim=64, action_dim=6).to(device)
      self.target_smuggle_net = SmuggleQNet(state_dim=state_dim, hidden_dim=64,action_dim=121).to(device)
      self.target_inspect_net = InspectQNet(state_dim=state_dim, hidden_dim=64, action_dim=12).to(device)

      self.buffer = ReplayBuffer()

      self.loss_fn = nn.SmoothL1Loss()

    def select_action(self,state,player_id,mask):

      if random.random() < self.epsilon:
            
            valid_actions = np.where(mask == 1)[0]
            action_idx = int(random.choice(valid_actions))
            
            # SMUGGLEフェーズの時だけ、env.pyが受け取れるように[actual, declared]の形にバラして返す
            if state.phase == Phase.SMUGGLE:
                act_idx = action_idx // 11
                dec_idx = action_idx % 11
                return np.array([act_idx, dec_idx])
            
            return action_idx
      
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
            q_values = self.smuggle_net(state_tensor, mask_tensor)
            action_idx = q_values.argmax(dim=-1).cpu().numpy()  # 0〜120の整数
            
            # env.pyに渡すために (actual, declared) の配列にデコードする
            act_idx = action_idx // 11
            dec_idx = action_idx % 11
            return np.array([act_idx, dec_idx]) 

        elif phase == Phase.INSPECT:
            q_values = self.inspect_net(state_tensor, mask_tensor)
            action_idx = q_values.argmax(dim=-1).cpu().numpy()
            return action_idx  # 例: array(0, dtype=int64)
            
        else:
            return None
    
    def update(self, batch_size):
        phase = random.choice([Phase.VOTE, Phase.SMUGGLE, Phase.INSPECT])
        if self.buffer.get_len(phase) < batch_size:
            return 0
      
        batch = self.buffer.sample_batch(phase, batch_size, device=self.device)
        states, actions, rewards, next_states, dones, masks, next_masks = batch

        if phase == Phase.VOTE:
            net = self.vote_net
            next_net = self.smuggle_net          # 次の行動選択用メインネット
            target_net = self.target_smuggle_net  # 次の価値評価用ターゲットネット
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
    
            next_q_online = next_net(next_states, next_masks)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = target_net(next_states, next_masks)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)

            # ターゲットQ値の計算
            target = rewards + self.gamma * (1.0 - dones) * next_q
        
        # 損失の計算とバックプロパゲーション
        loss = self.loss_fn(q_a, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.update_epsilon()
            
        return loss.item()
        
    
    def update_epsilon(self):
        
        self.epsilon_steps += 1
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        )*np.exp(-1.0*self.epsilon_steps/self.epsilon_decay)

    
    def copy_to_weight(self):
        """
        Target Networkへウェイトを同期する処理
        """
        self.target_vote_net.load_state_dict(self.vote_net.state_dict())
        self.target_smuggle_net.load_state_dict(self.smuggle_net.state_dict())
        self.target_inspect_net.load_state_dict(self.inspect_net.state_dict())

    def save(self, filepath):
        """
        3つのネットワークとオプティマイザの状態を一括で保存
        """
        state = {
            'vote_net': self.vote_net.state_dict(),
            'smuggle_net': self.smuggle_net.state_dict(),
            'inspect_net': self.inspect_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps
        }
        torch.save(state, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath):
        """
        保存された状態を一括で読み込み（学習の再開や評価用）
        """
        state = torch.load(filepath, map_location=self.device)
        self.vote_net.load_state_dict(state['vote_net'])
        self.smuggle_net.load_state_dict(state['smuggle_net'])
        self.inspect_net.load_state_dict(state['inspect_net'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epsilon = state['epsilon']
        self.total_steps = state['total_steps']
        print(f"Agent loaded from {filepath}")



       