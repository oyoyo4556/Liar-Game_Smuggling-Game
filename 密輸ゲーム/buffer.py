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
        if phase not in self.buffer:
            return

        # 後々の不揃いエラーを防ぐため、保存する時点でマスクを最大サイズ「121」にゼロパディングして固定長にする
        def pad_mask(m):
            padded = np.zeros(121, dtype=np.float32)
            if m is not None and len(m) > 0:
                padded[:len(m)] = m
            return padded

        data = (
            state_np.copy(),
            action_np.copy() if hasattr(action_np, 'copy') else action_np,
            float(reward),
            next_state_np.copy(),
            bool(done),
            pad_mask(mask_np),       # 固定長(121)に統一
            pad_mask(next_mask_np)   # 固定長(121)に統一
        )

        if len(self.buffer[phase]) < self.buffer_size:
            self.buffer[phase].append(data)
        else:
            self.buffer[phase][self.pos[phase]] = data
        
        self.pos[phase] = (self.pos[phase] + 1) % self.buffer_size

    def sample_batch(self, phase, batch_size, device="cpu"):
        """
        溜まっていた Numpy データを指定サイズ分ランダム抽出し、
        学習の直前で PyTorch の Tensor に変換して返す。
        """
        if len(self.buffer[phase]) < batch_size:
            return None

        batch = random.sample(self.buffer[phase], batch_size)
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.long, device=device)
        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32, device=device)
        
        # すでに固定長になっているため、安全に一括変換可能
        masks_tensor = torch.tensor(np.array(masks), dtype=torch.float32, device=device)
        next_masks_tensor = torch.tensor(np.array(next_masks), dtype=torch.float32, device=device)

        return (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, masks_tensor, next_masks_tensor)

    def get_len(self, phase):
        return len(self.buffer[phase])