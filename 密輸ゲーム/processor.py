import torch
import torch.nn.functional as F
import numpy as np

class Processor:
    def __init__(self):
        pass

    @staticmethod
    def observation_numpy(game_state, player_id: int) -> np.ndarray:
        """
        GameState を強化学習用の共通データ形式である np.ndarray (float32) に変換する。
        バッファへの保存にはこのメソッドの出力をそのまま使用します。
        """
        game = game_state
        obs_components = []

        # 1. フェーズ情報 (One-hot エンコーディング: 6次元)
        # Numpyで1軸のアイデンティティ行列（単位行列）を使ってone-hot化
        phase_idx = int(game.phase)
        phase_one_hot = np.eye(6, dtype=np.float32)[phase_idx]
        obs_components.append(phase_one_hot)

        # 2. ラウンド・手番情報 (2次元)
        obs_components.append(np.array([game.round / game.max_rounds], dtype=np.float32))
        obs_components.append(np.array([float(game.turn)], dtype=np.float32))

        # 3. 国外口座残高 (2次元)
        obs_components.append(np.array([game.foreign1_account / 1000.0], dtype=np.float32))
        obs_components.append(np.array([game.foreign2_account / 1000.0], dtype=np.float32))

        # 4. 現在のラウンド進行状況 (1次元)
        declared_normalized = game.current.declared_amount / 100.0
        obs_components.append(np.array([declared_normalized], dtype=np.float32))
    
        # 自分がこのラウンドの代表者かどうかのフラグ (1次元)
        is_representative = 1.0 if (player_id == game.smuggler.representative or player_id == game.inspector.representative) else 0.0
        obs_components.append(np.array([is_representative], dtype=np.float32))

        # 5. プレイヤー全員の個人口座残高 (6次元)
        accounts = [p.personal_account / 500.0 for p in game.players]
        obs_components.append(np.array(accounts, dtype=np.float32))

        # 6. 各チームの役職 (6次元)
        team_map = [0.0] * len(game.players)
        for p_id in game.smuggler.members:
            team_map[p_id] = 1.0
        obs_components.append(np.array(team_map, dtype=np.float32))

        # すべての配列を横一列に結合して1本のフラットなベクトル（23次元）にする
        return np.concatenate(obs_components, axis=0)

    @staticmethod
    def observation_tensor(game_state, player_id: int) -> torch.Tensor:
        obs_np = Processor.observation_numpy(game_state, player_id)
        return torch.from_numpy(obs_np)
