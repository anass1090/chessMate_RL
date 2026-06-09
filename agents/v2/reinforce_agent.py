import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import utils.board as b
from environment.v2.kqk_env import KQKEnv

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


class ReinforceAgentV2:
    """
    REINFORCE agent — v2.

    Fix over v1: returns are normalized across the entire update batch
    rather than per-episode. Per-episode normalization inverts the signal
    for short episodes where all rewards are negative (e.g. a 2-step episode
    ending in a draw makes the first action look good relative to the second,
    teaching the agent to sacrifice the queen on purpose).
    """

    MODEL_DIR = "models"

    def __init__(self, lr: float = 1e-3):
        self.env    = KQKEnv()
        self.policy = Policy().to(DEVICE)
        self.opt    = optim.Adam(self.policy.parameters(), lr=lr)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        print(f"Using device: {DEVICE}")

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
        obs_batch  = torch.FloatTensor(np.stack(all_obs)).to(DEVICE)
        ret_batch  = torch.tensor(all_returns, dtype=torch.float32).to(DEVICE)

        # Normalize across the full batch so the scale of returns is consistent
        # regardless of episode length. This is the key fix vs v1.
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
              log_every: int = 500, n_envs: int = 128, update_freq: int = 4096):
        envs      = [KQKEnv() for _ in range(n_envs)]
        obs_list  = np.stack([env.reset()[0] for env in envs])
        ep_bufs   = [{"obs": [], "act": [], "rew": []} for _ in range(n_envs)]

        all_obs:     list = []
        all_actions: list = []
        all_returns: list = []
        ep_count = 0

        while ep_count < n_episodes:
            with torch.no_grad():
                logits_batch = self.policy(
                    torch.from_numpy(obs_list).to(DEVICE)
                ).cpu()  # one transfer per outer iteration instead of one per env step

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

    def save(self, name: str = "kqk_reinforce_v2"):
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        torch.save(self.policy.state_dict(), path)
        print(f"Saved → {path}")

    def load(self, name: str = "kqk_reinforce_v2"):
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        self.policy.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded ← {path}")

    def evaluate(self, n_episodes: int = 50) -> dict:
        """Run greedy evaluation and print checkmate / draw / timeout rates."""
        counts: dict[str, int] = {}
        steps  = []

        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done   = False
            info   = {}

            while not done:
                with torch.no_grad():
                    action, _ = self.select_action(obs)
                obs, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

            reason = info.get("reason", "timeout")
            counts[reason] = counts.get(reason, 0) + 1
            steps.append(info.get("step", self.env.step_count))

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
