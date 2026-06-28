import random
import numpy as np
import torch
from .phase import Phase

import random
import numpy as np
import torch
from phase import Phase

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = {
            Phase.VOTE: [],
            Phase.SMUGGLE: [],
            Phase.INSPECT: []
        }
        self.pos = {Phase.VOTE: 0, Phase.SMUGGLE: 0, Phase.INSPECT: 0}

    def add(self, state_np, action_np, reward, next_state_np, done, mask_np, next_mask_np, phase):
        """
        すべてのデータを純粋な Numpy 配列、または数値として受け取って保存する。
        """
        # 自動フェーズなどは弾く
        if phase not in self.buffer:
            return

        # すべて Numpy なので、安全のために .copy() をとって完全に独立したデータとして保存
        data = (
            state_np.copy(),
            action_np.copy() if hasattr(action_np, 'copy') else action_np,
            float(reward),
            next_state_np.copy(),
            bool(done),
            mask_np.copy(),
            next_mask_np.copy()
        )

        if len(self.buffer[phase]) < self.buffer_size:
            self.buffer[phase].append(data)
        else:
            self.buffer[phase][self.pos[phase]] = data
        
        self.pos[phase] = (self.pos[phase] + 1) % self.buffer_size

    def sample_batch(self, phase, batch_size, device="cpu"):
        """
        溜まっていた Numpy データを指定サイズ分ランダム抽出し、
        学習の直前で【初めて】PyTorch の Tensor に変換して返す。
        """
        if len(self.buffer[phase]) < batch_size:
            return None

        batch = random.sample(self.buffer[phase], batch_size)
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)

        # 学習に回すため、ここで一気に Tensor に一括変換（高速です）
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32, device=device)
        masks_tensor = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
        next_masks_tensor = torch.tensor(np.array(next_masks), dtype=torch.float32, device=device)

        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, masks_tensor, next_masks_tensor)

    def get_len(self, phase):
        return len(self.buffer[phase])