from phase import Phase
from state import GameState,PlayerState,TeamState
from state import RoundInfo
import random
import numpy as np

class BatchsmugglingGame:
    def __init__(self, num_players=6):
        self.game = GameState()
        self.game.players = [PlayerState(id=i, team=-1) for i in range(num_players)]

    def reset(self):
        # 1. 基本パラメータの初期化
        self.game.phase = Phase.VOTE
        self.game.round = 1
        self.game.turn = 0
        self.game.current = RoundInfo()  # ラウンド情報もクリア

        num_players = len(self.game.players)
        half = num_players // 2

        # 2. 全プレイヤーの個人口座リセットと、チーム決めのためのシャッフル
        player_ids = list(range(num_players))
        random.shuffle(player_ids)  

        # 前半をチーム0、後半をチーム1にする
        team0_members = player_ids[:half]
        team1_members = player_ids[half:]

        for p_id in team0_members:
            self.game.players[p_id].team = 0
            self.game.players[p_id].personal_account = 100

        for p_id in team1_members:
            self.game.players[p_id].team = 1
            self.game.players[p_id].personal_account = 100

        # 3. チーム状態の初期化
        # 最初（Turn 0）はチーム0が密輸側、チーム1が検査側に固定
        self.game.smuggler = TeamState(team=0, members=team0_members, representative=-1)
        self.game.inspector = TeamState(team=1, members=team1_members, representative=-1)

        # 4. 国外口座の初期化
        self.game.foreign1_account = half * 300
        self.game.foreign2_account = half * 300

        return self.get_state()

    def get_state(self):
        #obsは外側で作る
        return self.game
    
    def step(self,actions):
        
        if self.game.phase == Phase.VOTE:
            self.step_vote(actions)
        elif self.game.phase == Phase.SMUGGLE:
            self.step_smuggle(actions)
        elif self.game.phase == Phase.INSPECT:
            self.step_inspect(actions)
        
        while self.game.phase in [Phase.DISTRIBUTE, Phase.PUBLIC_UPDATE]:
            if self.game.phase == Phase.DISTRIBUTE:
                self.step_distribute()
            elif self.game.phase == Phase.PUBLIC_UPDATE:
                self.step_public_update()
        
        # ----------------------------------------
        # 共通の return 処理
        # ----------------------------------------
        
        # 終了判定
        terminated = (self.game.phase == Phase.TERMINATED)

        # 報酬（Reward）の計算
        rewards = {p.id: 0 for p in self.game.players}
        if terminated:
            for p in self.game.players:
                # 最終的な個人口座の残高をそのまま報酬（あるいはスコア）とする
                rewards[p.id] = p.personal_account / 100

        next_masks = {}
        if not terminated:
            for p in self.game.players:
                next_masks[p.id] = self.legal_action_mask(p.id)

        # 状態、報酬、終了フラグ、マスクを返す
        return self.get_state(), rewards, terminated, {"masks": next_masks}
    
    def step_vote(self,actions):
        '''
        集計 -> 代表者決定 -> phase=SMUGGLE 
        -> legal_action_mask更新してstateを出す
        actions: { player_id: voted_player_id }
        '''
        # 1. 各プレイヤーが誰に何票入れたかを集計するカウンターを用意
        # ※ リストのインデックスをplayer_idとして票数をカウント
        vote_counts = [0] * len(self.game.players)
        
        for _, voted_id in actions.items():
            # 【簡易バリデーション】投票先が有効なプレイヤーかチェック
            if 0 <= voted_id < len(self.game.players):
                
                vote_counts[voted_id] += 1

        # 2. 密輸側（Smugglers）の代表者を決定
        smugglers = self.game.smuggler.members
        max_smuggle_votes = max(vote_counts[p_id] for p_id in smugglers)
        # 最大得票数を持っているメンバーをすべてリストアップ（同票対策）
        smuggle_candidates = [p_id for p_id in smugglers if vote_counts[p_id] == max_smuggle_votes]
        # その中からランダムに1人を選択
        self.game.smuggler.representative = random.choice(smuggle_candidates)

        # 3. 検査官側（Inspectors）の代表者を決定
        inspectors = self.game.inspector.members
        max_inspect_votes = max(vote_counts[p_id] for p_id in inspectors)
        inspect_candidates = [p_id for p_id in inspectors if vote_counts[p_id] == max_inspect_votes]
        self.game.inspector.representative = random.choice(inspect_candidates)



        # 4. ラウンド情報の初期化
        self.game.current = RoundInfo()

        # 5. フェーズを「密輸（SMUGGLE）」に移行
        self.game.phase = Phase.SMUGGLE


    def step_smuggle(self,actions):
        """
        密輸フェーズのアクション処理
        actions: { player_id: (actual_idx, declared_idx) }
        ※ actual_idx: 0〜10, declared_idx: 0〜10
        """
        rep_id = self.game.smuggler.representative

        if rep_id in actions:
            
            act_idx = actions[rep_id][0]
            dec_idx = actions[rep_id][1]
        else:
            act_idx, dec_idx = 0, 0

        # 金額への変換（Numpyの整数型のまま10倍して代入可能）
        actual_amount = act_idx * 10
        declared_amount = dec_idx * 10

        # 状態更新
        if self.game.smuggler.team == 0:
            self.game.foreign1_account -= actual_amount
        else:
            self.game.foreign2_account -= actual_amount

        self.game.current.smuggled_amount = actual_amount
        self.game.current.declared_amount = declared_amount
        self.game.phase = Phase.INSPECT


    def step_inspect(self, actions: dict[int, int]):
        """
        検査フェーズのアクション処理
        actions: { player_id: action_idx }
        ※ action_idx: 0 = PASS, 1〜11 = 0〜100万ダウト
        """
        rep_id = self.game.inspector.representative

        # 1. 代表者のアクション（インデックス）を取得
        action_idx = actions[rep_id] if rep_id in actions else 0

        # 2. インデックスから「パスかダウトか」「ダウト額」を判定・デコード
        if action_idx == 0:
            # --- PASS (不検査) の場合 ---
            is_doubt = 0
            doubt_amount = 0
        else:
            # --- DOUBT (検査) の場合 ---
            is_doubt = 1
            # インデックス 1->0万, 2->10万, ..., 11->100万 にデコード
            doubt_amount = (action_idx - 1) * 10

        # 3. 判定ロジック
        actual_amount = self.game.current.smuggled_amount
        declared_amount = self.game.current.declared_amount

        if is_doubt == 0:
            self.game.current.inspected = False
            self.game.current.doubt_amount = 0
            self.game.current.success = True
            
            # 密輸側チームの個人口座に直接入るのではなく、gain_amountにプール
            self.game.current.gain_amount = actual_amount
            self.game.current.stolen_amount = 0
        else:
            self.game.current.inspected = True
            self.game.current.doubt_amount = doubt_amount
            
            compensation = doubt_amount // 2

            if actual_amount <= doubt_amount and actual_amount != 0:
                # 【検査官の勝ち】
                self.game.current.success = False
                self.game.current.gain_amount = actual_amount 
                self.game.current.stolen_amount = 0
            else:
                # 【密輸側の勝ち】
                self.game.current.success = True
                self.game.current.gain_amount = actual_amount + compensation
                self.game.players[rep_id].personal_account -= compensation
                self.game.current.stolen_amount = -compensation

        # 分配（DISTRIBUTE）フェーズへ移行
        self.game.phase = Phase.DISTRIBUTE


    def step_distribute(self):
        """
        分配フェーズのアクション処理（自動山分けルール）。
        ※自動処理のため、actionsの中身は無視(またはNoneでOK)します。
        """
        # 1. どちらのチームが勝った（利益を得た）かによって分配先を分ける
        if self.game.current.success:
            # 密輸側が成功（パスされた、またはダウトを上回った）
            winning_team = self.game.smuggler
        else:
            # 検査官側が成功（ダウトで看破した）
            winning_team = self.game.inspector

        # 2. 勝ったチームへの利益分配（割り切れない端数は一旦考慮せず、単純に人数で割る）
        win_share = self.game.current.gain_amount // len(winning_team.members)
        for p_id in winning_team.members:
            self.game.players[p_id].personal_account += win_share

        # 3. フェーズを「公開情報更新（PUBLIC_UPDATE）」に移行
        self.game.phase = Phase.PUBLIC_UPDATE

    
    def step_public_update(self):
        '''
        履歴更新 -> round += 1 -> phase = VOTE
        -> legal_action_mask更新してstateを出す

        round_maxでdone=trueにしてreward計算
        '''
        # 代表者IDは次のVOTEフェーズで再決定されるため、一旦リセット
        self.game.smuggler.representative = -1
        self.game.inspector.representative = -1

        # 交代用の関数、または直接入れ替え
        def swap_roles():
            # members と 担当する team ID をごそっと入れ替える
            s_members, s_team = self.game.smuggler.members, self.game.smuggler.team
            i_members, i_team = self.game.inspector.members, self.game.inspector.team
            
            self.game.smuggler.members, self.game.smuggler.team = i_members, i_team
            self.game.inspector.members, self.game.inspector.team = s_members, s_team

        if self.game.turn == 0:
            # --- 往路終了（片方のチームの密輸が終わり） ---
            swap_roles()
            self.game.turn = 1
            self.game.phase = Phase.VOTE
        else:
            # --- 復路終了（両チームが1回ずつ密輸を終えた = 1ラウンド完了） ---
            if self.game.round < self.game.max_rounds:
                self.game.round += 1
                self.game.turn = 0  # ターンは最初（0）に戻る
                swap_roles()        # 元の攻守関係に戻す
                self.game.phase = Phase.VOTE
            else:
                self.game.phase = Phase.TERMINATED
                self.exact_calculate()

    def legal_action_mask(self,player_id:int):
        phase = self.game.phase

        # ----------------------------------------
        # 1. VOTE（投票）フェーズ: マスクサイズ = 6 (全プレイヤー数)
        # ----------------------------------------
        if phase == Phase.VOTE:
            mask = np.zeros(len(self.game.players), dtype=np.int32)
            voter_team = self.game.players[player_id].team
            
            # 原作準拠: 自チームのメンバー（自分を含む3人）にしか投票できない
            for p in self.game.players:
                if p.team == voter_team:
                    mask[p.id] = 1
            return mask

        # ----------------------------------------
        # 2. SMUGGLE（密輸）フェーズ: マスクサイズ = 121 (実際11択 × 申告11択)
        # ----------------------------------------
        elif phase == Phase.SMUGGLE:
            if player_id != self.game.smuggler.representative:
                return np.zeros(121, dtype=np.int32) # 手番外はすべて不可

            # 11×11 = 121次元のマスクを初期化（すべて0 = 選択不可）
            mask = np.zeros(121, dtype=np.int32)
            
            # 密輸側の国の残高を取得
            current_foreign_account = (
                self.game.foreign1_account if self.game.smuggler.team == 0 
                else self.game.foreign2_account
            )

            # 実際額(act_idx)と申告額(dec_idx)のすべての組み合わせをチェック
            for act_idx in range(11):
                amount = act_idx * 10 
                
                # 国外口座の残高を超えない額なら、その実際額は合法
                if amount <= current_foreign_account:
                    # 実際額が合法なら、申告額（0〜10）は何を選んでも自由（ルール上制限なし）
                    for dec_idx in range(11):
                        # 2次元のインデックスを1次元(0〜120)にフラット化
                        action_idx = act_idx * 11 + dec_idx
                        mask[action_idx] = 1

            return mask

        # ----------------------------------------
        # 3. INSPECT（検査）フェーズ: マスクサイズ = 12 (パス1択 + ダウト額11択)
        # ----------------------------------------
        elif phase == Phase.INSPECT:
            if player_id != self.game.inspector.representative:
                return np.zeros(12, dtype=np.int32)

            mask = np.zeros(12, dtype=np.int32)
            
            # インデックス 0: PASS（不検査）は常に選択可能
            mask[0] = 1

            # インデックス 1〜11: ダウト額（10〜100万円）のマスク
            # 「払える金額だけしかダウトできない」縛りを実装
            # ミスした時の慰謝料は「ダウト額 // 2」なので、これが自分の口座残高以下である必要がある
            my_account = self.game.players[player_id].personal_account

            for doubt_idx in range(11):
                doubt_amount = doubt_idx * 10
                compensation = doubt_amount // 2
                
                # 慰謝料を自分の個人口座から支払えるなら合法
                if compensation <= my_account:
                    # アクション配列の 1〜11 にマッピング（doubt_idx + 1）
                    mask[doubt_idx + 1] = 1
            return mask

        # ----------------------------------------
        # 4. その他の自動フェーズ (DISTRIBUTE, PUBLIC_UPDATEなど)
        # ----------------------------------------
        else:
            # 意思決定がないフェーズは空のマスクを返す
            return np.array([], dtype=np.int32)

    def exact_calculate(self):
        """
        ゲーム終了時の最終清算ロジック
        """
        # 1. 国外口座の残高を相手チームに山分け
        # チーム0の国外口座の残り（foreign1_account）は、チーム1のメンバーで山分け
        team0_members = [p.id for p in self.game.players if p.team == 0]
        team1_members = [p.id for p in self.game.players if p.team == 1]

        if len(team1_members) > 0:
            share_from_foreign1 = self.game.foreign1_account // len(team1_members)
            for p_id in team1_members:
                self.game.players[p_id].personal_account += share_from_foreign1
        
        # チーム1の国外口座の残り（foreign2_account）は、チーム0のメンバーで山分け
        if len(team0_members) > 0:
            share_from_foreign2 = self.game.foreign2_account // len(team0_members)
            for p_id in team0_members:
                self.game.players[p_id].personal_account += share_from_foreign2

        # 2. 口座残高を0にクリア（任意：状態の記録用）
        self.game.foreign1_account = 0
        self.game.foreign2_account = 0

        # 3. 借金の返済
        for i in range(len(self.game.players)):
            self.game.players[i].personal_account -= 400
    