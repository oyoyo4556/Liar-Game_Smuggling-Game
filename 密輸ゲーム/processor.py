import torch
import torch.nn.functional as F

class Processor:
    def __init__(self):
        pass

    def observation_tensor(game_state, player_id: int) -> torch.Tensor:
        """
        GameState を強化学習用の torch.Tensor (float32) に変換する。
        """
        game = game_state
        obs_components = []

        # 1. フェーズ情報 (One-hot エンコーディング: 6次元)
        # Phaseの列挙型 (0〜5) を 6次元のOne-hotに変換
        phase_tensor = F.one_hot(torch.tensor(int(game.phase)), num_classes=6).float()
        obs_components.append(phase_tensor)

        # 2. ラウンド・手番情報 (2次元)
        # 30ラウンドが上限なので、max_roundsで割って 0〜1 に正規化
        obs_components.append(torch.tensor([game.round / game.max_rounds], dtype=torch.float32))
        obs_components.append(torch.tensor([float(game.turn)], dtype=torch.float32))

        # 3. 国外口座残高 (2次元)
        # 初期値（例：3人×300=900）を基準に正規化
        obs_components.append(torch.tensor([game.foreign1_account / 1000.0], dtype=torch.float32))
        obs_components.append(torch.tensor([game.foreign2_account / 1000.0], dtype=torch.float32))

        # 4. 現在のラウンド進行状況 (1次元)
        declared_normalized = game.current.declared_amount / 100.0
        obs_components.append(torch.tensor([declared_normalized], dtype=torch.float32))
    
        # 自分がこのラウンドの代表者（密輸側、または検査側）かどうかのフラグ
        is_representative = 1.0 if (player_id == game.smuggler.representative or player_id == game.inspector.representative) else 0.0
        obs_components.append(torch.tensor([is_representative], dtype=torch.float32))

        # 5. プレイヤー全員の個人口座残高 (6次元)
        # 3対3を想定し、全6人分の残高を並べる。初期値100、パスポート代-400などを考慮し、500.0等で割って正規化
        accounts = [p.personal_account / 500.0 for p in game.players]
        obs_components.append(torch.tensor(accounts, dtype=torch.float32))

        # 6. 各チームの役職（誰が密輸側チームで、誰が検査側チームか）(6次元)
        # プレイヤーID 0〜5 が 密輸チーム(1.0) か 検査チーム(0.0) かを並べる
        team_map = [0.0] * len(game.players)
        for p_id in game.smugler.members:
            team_map[p_id] = 1.0
        obs_components.append(torch.tensor(team_map, dtype=torch.float32))


        # すべての要素を横一列に結合して、1本のフラットなベクトルにする
        # 形状: [合計次元数] (例: 6 + 1 + 1 + 1 + 2 + 6 + 6 = 23次元)
        return torch.cat(obs_components, dim=0)
