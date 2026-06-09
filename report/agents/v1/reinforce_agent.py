import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import utils.board as b
from environment.v1.kqk_env import KQKEnv

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


class ReinforceAgent:
    """Vanilla policy gradient (REINFORCE) agent for the KQK environment."""

    MODEL_DIR = "models"

    def __init__(self, lr: float = 1e-3):
        self.env    = KQKEnv()
        self.policy = Policy().to(DEVICE)
        self.opt    = optim.Adam(self.policy.parameters(), lr=lr)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        print(f"Using device: {DEVICE}")

    def sample_action(self, logits: torch.Tensor, legal_actions: list[int]) -> tuple[int, int]:
        """
        Sample a legal action from pre-computed logits.

        Takes an explicit legal_actions list so it works with any environment
        instance, not just self.env. Applies softmax only over legal move
        indices so illegal moves are never sampled, then draws one action
        proportional to those probabilities.
        Returns (action, chosen_idx) where chosen_idx is the position within
        legal_actions (not the raw action integer).
        """
        probs = torch.softmax(logits[legal_actions], dim=0)
        idx   = int(torch.multinomial(probs, 1).item())
        return legal_actions[idx], idx

    def select_action(self, obs: np.ndarray) -> tuple[int, torch.Tensor]:
        """
        Full obs → action pipeline used during evaluation.

        Runs the observation through the network to get logits, delegates
        action sampling to sample_action, then recomputes log_prob for the
        chosen action (needed by the caller).
        """
        obs_t         = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        logits        = self.policy(obs_t).squeeze(0)
        legal_actions = [b.move_to_action(m) for m in self.env.board.legal_moves]
        action, idx   = self.sample_action(logits, legal_actions)
        probs         = torch.softmax(logits[legal_actions], dim=0)
        log_prob      = torch.log(probs[idx])
        return action, log_prob

    def gradient_update(self, all_obs: list, all_actions: list, all_returns: list) -> None:
        """
        One gradient update over a collected batch of experience.

        Does a single forward pass over all observations, re-computes
        log-probs for the chosen actions, and applies the REINFORCE loss.
        """
        obs_batch  = torch.FloatTensor(np.stack(all_obs)).to(DEVICE)
        ret_batch  = torch.tensor(all_returns, dtype=torch.float32).to(DEVICE)
        all_logits = self.policy(obs_batch)                              # (N, 4096)

        # Re-compute log-probs so gradients flow back through the network.
        # Each step has a different legal move set so this loop can't be vectorised.
        log_probs = []
        for i, (legal, chosen_idx) in enumerate(all_actions):
            probs = torch.softmax(all_logits[i][legal], dim=0)
            log_probs.append(torch.log(probs[chosen_idx]))

        # REINFORCE loss: maximise E[log π(a|s) · G_t], written as a minimisation.
        # Positive return → increase log-prob of that action.
        # Negative return → decrease it.
        loss = -(torch.stack(log_probs) * ret_batch).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self, n_episodes: int = 10_000, gamma: float = 0.99,
              log_every: int = 500, n_envs: int = 128, update_freq: int = 4096):
        """
        Train using REINFORCE with vectorized environments.

        Runs n_envs games in lockstep: at each timestep all environments'
        observations are stacked into one forward pass to select actions,
        replacing n_envs sequential single-sample calls with one batched call.
        A gradient update happens every update_freq steps collected.
        """
        envs      = [KQKEnv() for _ in range(n_envs)]
        obs_list  = np.stack([env.reset()[0] for env in envs])
        ep_bufs   = [{"obs": [], "act": [], "rew": []} for _ in range(n_envs)]

        all_obs:     list = []
        all_actions: list = []
        all_returns: list = []
        ep_count = 0

        while ep_count < n_episodes:
            # One forward pass for all active environments at this timestep.
            with torch.no_grad():
                logits_batch = self.policy(
                    torch.FloatTensor(obs_list).to(DEVICE)
                )

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
                    # Discounted return for each step: G_t = r_t + γ·G_{t+1}
                    # Normalised per episode to keep gradients stable across
                    # episodes of different lengths.
                    G, rets = 0.0, []
                    for r in reversed(ep_bufs[i]["rew"]):
                        G = r + gamma * G
                        rets.insert(0, G)
                    ret_t = torch.tensor(rets, dtype=torch.float32)
                    ret_t = (ret_t - ret_t.mean()) / (ret_t.std(correction=0) + 1e-8)

                    all_obs.extend(ep_bufs[i]["obs"])
                    all_actions.extend(ep_bufs[i]["act"])
                    all_returns.extend(ret_t.tolist())

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

    def save(self, name: str = "kqk_reinforce"):
        """Save weights to models/<name>.pt."""
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        torch.save(self.policy.state_dict(), path)
        print(f"Saved → {path}")

    def load(self, name: str = "kqk_reinforce"):
        """Load weights from models/<name>.pt."""
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