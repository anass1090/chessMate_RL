import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import utils.board as b
from environment.v3.kqk_env import KQKEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    """MLP: 768-dim board observation → logits over 4096 actions."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(b.OBS_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4096),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReinforceAgentV3:
    """
    REINFORCE agent — v3.

    Changes over v2:
    - Uses v3 environment with shaped reward (black king distance from center)
      and curriculum start (fraction of episodes begin from mate-in-1 positions).
    - curriculum_ratio controls the mix: 1.0 = all mate-in-1, 0.0 = all random.
      A value of 0.5 means half the episodes start from mate-in-1, giving the
      agent frequent exposure to the +1.0 reward signal while still training on
      general positions.
    """

    MODEL_DIR = "models"

    def __init__(self, lr: float = 1e-3, curriculum_ratio: float = 0.5):
        self.policy          = Policy().to(DEVICE)
        self.opt             = optim.Adam(self.policy.parameters(), lr=lr)
        self.curriculum_ratio = curriculum_ratio
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        print(f"Using device: {DEVICE}")
        self.env = KQKEnv(curriculum_ratio=curriculum_ratio)
        self.mate_pool = self.env.mate_pool  # shared across all envs

    def sample_action(self, logits: torch.Tensor, legal_actions: list[int]) -> tuple[int, int]:
        probs = torch.softmax(logits[legal_actions], dim=0)
        idx   = int(torch.multinomial(probs, 1).item())
        return legal_actions[idx], idx

    def select_action(self, obs: np.ndarray) -> tuple[int, torch.Tensor]:
        obs_t         = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        logits        = self.policy(obs_t).squeeze(0)
        legal_actions = [b.move_to_action(m) for m in self.env.board.legal_moves]
        action, idx   = self.sample_action(logits, legal_actions)
        probs         = torch.softmax(logits[legal_actions], dim=0)
        log_prob      = torch.log(probs[idx])
        return action, log_prob

    def gradient_update(self, all_obs: list, all_actions: list, all_returns: list) -> None:
        obs_batch = torch.from_numpy(np.stack(all_obs)).to(DEVICE)
        ret_batch = torch.tensor(all_returns, dtype=torch.float32).to(DEVICE)
        ret_batch = (ret_batch - ret_batch.mean()) / (ret_batch.std() + 1e-8)

        all_logits = self.policy(obs_batch)  # (N, 4096)

        log_probs = []
        for i, (legal, chosen_idx) in enumerate(all_actions):
            probs = torch.softmax(all_logits[i][legal], dim=0)
            log_probs.append(torch.log(probs[chosen_idx]))

        loss = -(torch.stack(log_probs) * ret_batch).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self, n_episodes: int = 10_000, gamma: float = 0.99,
              log_every: int = 500, n_envs: int = 8, update_freq: int = 128):
        envs     = [KQKEnv(curriculum_ratio=self.curriculum_ratio, mate_pool=self.mate_pool) for _ in range(n_envs)]
        obs_list = np.stack([env.reset()[0] for env in envs])
        ep_bufs  = [{"obs": [], "act": [], "rew": []} for _ in range(n_envs)]

        all_obs:     list = []
        all_actions: list = []
        all_returns: list = []
        ep_count = 0

        while ep_count < n_episodes:
            with torch.no_grad():
                logits_batch = self.policy(
                    torch.from_numpy(obs_list).to(DEVICE)
                ).cpu()

            for i, env in enumerate(envs):
                if ep_count >= n_episodes:
                    break

                legal       = [b.move_to_action(m) for m in env.board.legal_moves]
                action, idx = self.sample_action(logits_batch[i], legal)

                ep_bufs[i]["obs"].append(obs_list[i].copy())
                ep_bufs[i]["act"].append((legal, idx))

                obs, reward, terminated, truncated, _ = env.step(action)
                ep_bufs[i]["rew"].append(reward)
                obs_list[i] = obs

                if terminated or truncated:
                    G, rets = 0.0, []
                    for r in reversed(ep_bufs[i]["rew"]):
                        G = r + gamma * G
                        rets.insert(0, G)

                    all_obs.extend(ep_bufs[i]["obs"])
                    all_actions.extend(ep_bufs[i]["act"])
                    all_returns.extend(rets)

                    obs, _      = env.reset()
                    obs_list[i] = obs
                    ep_bufs[i]  = {"obs": [], "act": [], "rew": []}
                    ep_count   += 1

                    if ep_count % log_every == 0:
                        print(f"Episode {ep_count}/{n_episodes}")

            if len(all_obs) >= update_freq:
                self.gradient_update(all_obs, all_actions, all_returns)
                all_obs, all_actions, all_returns = [], [], []

        if all_obs:
            self.gradient_update(all_obs, all_actions, all_returns)

    def save(self, name: str = "kqk_reinforce_v3"):
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        torch.save(self.policy.state_dict(), path)
        print(f"Saved → {path}")

    def load(self, name: str = "kqk_reinforce_v3"):
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        self.policy.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded ← {path}")

    def evaluate_mate_in_one(self, n_episodes: int = 500) -> float:
        """Evaluate specifically from mate-in-1 positions. Returns checkmate rate."""
        if not self.mate_pool:
            raise ValueError("No mate pool available.")
        correct = 0
        import chess
        for _ in range(n_episodes):
            board = chess.Board(random.choice(self.mate_pool))
            obs   = b.board_to_obs(board)
            with torch.no_grad():
                legal_actions = [b.move_to_action(m) for m in board.legal_moves]
                logits        = self.policy(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)).squeeze(0)
                action, _     = self.sample_action(logits.cpu(), legal_actions)
            board.push(b.action_to_move(action))
            if board.is_checkmate():
                correct += 1
        rate = correct / n_episodes
        print(f"Mate-in-1 accuracy: {correct}/{n_episodes} = {rate*100:.1f}%")
        return rate

    def evaluate(self, n_episodes: int = 50) -> dict:
        """Run greedy evaluation from random positions (no curriculum)."""
        counts: dict[str, int] = {}
        steps  = []

        # Evaluate from general positions regardless of curriculum_ratio
        eval_env = KQKEnv(curriculum_ratio=0.0)

        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done   = False
            info   = {}

            while not done:
                with torch.no_grad():
                    legal_actions = [b.move_to_action(m) for m in eval_env.board.legal_moves]
                    logits        = self.policy(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)).squeeze(0)
                    action, _     = self.sample_action(logits.cpu(), legal_actions)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated

            reason = info.get("reason", "timeout")
            counts[reason] = counts.get(reason, 0) + 1
            steps.append(info.get("step", eval_env.step_count))

        results = {
            **counts,
            "mean_steps":     float(np.mean(steps)),
            "checkmate_rate": counts.get("checkmate", 0) / n_episodes,
        }
        print(f"\n--- Eval ({n_episodes} eps) ---")
        for reason, count in sorted(counts.items()):
            print(f"  {reason:30s}: {count}")
        print(f"  {'mean_steps':30s}: {results['mean_steps']:.1f}")
        print(f"  {'checkmate_rate':30s}: {results['checkmate_rate']*100:.1f}%")
        return results
