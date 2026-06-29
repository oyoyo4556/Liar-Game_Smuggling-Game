import numpy as np
import random
from collections import defaultdict, deque
from phase import Phase
from processor import Processor

class SmugglingTrainer:
    def __init__(self, env, agent, batch_size=64, train_start=1000):
        """
        env: BatchsmugglingGame のインスタンス
        agent: DQNAgent のインスタンス（内部にバッファ self.buffer を持っている前提）
        """
        self.env = env
        self.agent = agent
        self.buffer = agent.buffer  # エージェント内のバッファを参照
        
        self.batch_size = batch_size
        self.train_start = train_start
        
        self.env_steps = 0
        self.loss_history = defaultdict(lambda: deque(maxlen=1000))
        self.reward_history = deque(maxlen=1000)

    def run_episode(self,train_player_id=0):
        """
        1エピソードを実行し、手番プレイヤーの有効な行動だけをバッファに保存する
        """
        self.env.reset()
        state = self.env.game
        
        # 全プレイヤーのマスクを取得
        masks = {p.id: self.env.legal_action_mask(p.id) for p in state.players}
        
        episode_rewards = defaultdict(float)
        terminated = False

        while not terminated:
            phase_at_step = state.phase
            actions = {}
            
            # このステップで本当に意思決定を行うプレイヤー（代表者たち）を格納するリスト
            active_players = []

            # 1. 意思決定が必要なプレイヤー（マスクに1が1つ以上あるプレイヤー）を厳密に抽出
            for p in state.players:
                mask_np = masks[p.id]
                # マスクが空、またはすべて0（手番ではない）なら絶対にスキップ
                if mask_np is None or len(mask_np) == 0 or np.sum(mask_np) == 0:
                    continue
                
                active_players.append(p.id)
                
                # エージェントに行動を決定させる
                action = self.agent.select_action(state, p.id, mask_np,train_player_id=train_player_id)
                actions[p.id] = action

            # もし誰も行動しない特殊な自動フェーズ等の場合は、空の行動辞書で環境を1ステップ進める
            next_state, rewards, terminated, info = self.env.step(actions)
            next_masks = info["masks"]

            # 2. 実際に行動を起こしたプレイヤーの「その瞬間の経験」だけを個別にバッファへ保存
            for p_id in active_players:
                # 行動直前の状態をNumpy化
                state_np = Processor.observation_numpy(state, p_id)
                
                # 行動後の状態をNumpy化
                next_state_np = Processor.observation_numpy(next_state, p_id)
                # 次のマスク（もしなければゼロ配列）
                next_mask_np = next_masks.get(p_id, np.zeros_like(masks[p_id]))
                
                # バッファに追加
                self.buffer.add(
                    state_np=state_np,
                    action_np=actions[p_id],
                    reward=rewards[p_id],
                    next_state_np=next_state_np,
                    done=terminated,
                    mask_np=masks[p_id],
                    next_mask_np=next_mask_np,
                    phase=phase_at_step
                )
                
                episode_rewards[p_id] += rewards[p_id]
                self.env_steps += 1

            # 状態とマスクを更新して次のステップへ
            state = next_state
            masks = next_masks

        if self.env_steps >= self.train_start :
            update_phase = random.choice([Phase.VOTE, Phase.SMUGGLE, Phase.INSPECT])
            loss = self.agent.update(self.batch_size,update_phase)
            if loss > 0:
                self.loss_history[update_phase].append(loss)
                
        # ターゲットネットワークの同期
        if self.agent.total_steps > 0 and self.agent.total_steps % self.agent.target_update_interval == 0:
            self.agent.copy_to_weight()

        avg_reward = np.mean(list(episode_rewards.values())) if episode_rewards else 0.0
        self.reward_history.append(avg_reward)
        
        return episode_rewards
    
    def evaluate_and_print_episode(self):
        print("\n" + "="*50)
        print(" 🚨 TEST RUN: 1 EPISODE VISUALIZATION 🚨")
        print("="*50)
    
        # 1. 環境のリセット
        self.env.reset()
        state = self.env.game
        terminated = False
    
        # 2. 全プレイヤーの初期マスク取得
        masks = {p.id: self.env.legal_action_mask(p.id) for p in state.players}
    
        round_count = 1

        while not terminated:
            phase_at_step = state.phase
            actions = {}
            active_players = []

            # 意思決定が必要なプレイヤー（手番プレイヤー）を探索
            for p in state.players:
                mask_np = masks[p.id]
                if mask_np is None or len(mask_np) == 0 or np.sum(mask_np) == 0:
                     continue
            
                active_players.append(p.id)
            
                original_epsilon = self.agent.epsilon
                self.agent.epsilon = 0.0  
            
                action = self.agent.select_action(state, p.id, mask_np,train_player_id=None)
                actions[p.id] = action
            
                self.agent.epsilon = original_epsilon  # epsilon を元に戻す

            if active_players:
                print(f"\n--- [Round {state.round+1}] Phase: {phase_at_step.name} ---")
                for p_id in active_players:
                    act = actions[p_id]
                
                    # フェーズごとに、数字（インデックス）が何を意味するか翻訳して表示
                    if phase_at_step == Phase.VOTE:
                        # 投票先（例: 0〜5のプレイヤーID）
                        print(f" 👤 Player {p_id} ➔ 🗳️ Vote Target: Player {act}")
                    
                    elif phase_at_step == Phase.SMUGGLE:
                        # 121次元のアクションIDを「実際額」と「申告額」にデコード
                        actual_amount = (act // 11) * 10
                        declared_amount = (act % 11) * 10
                        print(f" 👤 Player {p_id} (密輸代表) ➔ 💼 実際額: ${actual_amount} | 📢 申告額: ${declared_amount} (ActionID: {act})")
                    
                    elif phase_at_step == Phase.INSPECT:
                        if act == 0:
                            print(f" 👤 Player {p_id} (検査代表) ➔ 🛡️ 行動: PASS (不検査)")
                        else:
                            doubt_amount = (act - 1) * 10
                            print(f" 👤 Player {p_id} (検査代表) ➔ 🔍 行動: DOUBT (${doubt_amount}ダウト) (ActionID: {act})")
            # 4. 環境を1ステップ進める
            next_state, rewards, terminated, info = self.env.step(actions)
            next_masks = info["masks"]

            if phase_at_step == Phase.INSPECT and self.env.game.current:
                c = self.env.game.current
                print(f"   💥 [ダウト結果判明] 実際額: ${c.smuggled_amount} vs ダウト額: ${c.doubt_amount}")
                if not c.inspected:
                    print(f"   ➔ 検査官がPASSしたため、密輸側が ${c.gain_amount} をプールに獲得！")
                else:
                    if not c.success:
                        print(f"   ➔ 【検査官の勝ち】実際額がダウト額以下！密輸額 ${c.gain_amount} は税関に没収（プール）！")
                    else:
                        compensation = c.doubt_amount // 2
                        print(f"   ➔ 【密輸側の勝ち】実際額の方が大きい、または実際額が0万！")
                        print(f"      密輸側は元金と賠償金を合わせて ${c.gain_amount} をプールに獲得！")
                        print(f"      検査官は自腹で賠償金 ${compensation} を支払った。")

            # 状態とマスクを更新
            state = next_state
            masks = next_masks

        print("\n" + "="*50)
        print(" 🎉 GAME OVER (エピソード終了) 🎉")
        # ゲーム終了時の最終スコア（各プレイヤーの個人口座残高など）を表示するとさらに面白いです
        print("Final Accounts:")
        for p in state.players:
            print(f"  Player {p.id}: ${p.personal_account} (Team: {p.team})")
        print("="*50 + "\n")