import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import chess
import utils.board as b
from environment.v2.kqk_env import KQKEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    """
    Shared backbone → actor head (policy logits) + critic head (state value).

    Sharing the backbone means both heads learn a common board representation,
    which is more parameter-efficient than two separate networks. The actor and
    critic heads are cheap linear layers on top.
    """

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(b.OBS_SIZE, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, 4096)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent for the KQK environment.

    Key difference over plain REINFORCE:
    - A critic network estimates V(s) — the expected return from each position.
    - The actor updates on advantages A_t = G_t - V(s_t) instead of raw returns.
    - Advantage asks "was this action better or worse than expected from here?"
      rather than "was the whole episode good or bad?" — much lower variance.
    - A baseline-free REINFORCE must credit/blame all moves in an episode equally;
      A2C can identify which specific moves were above or below expectation.
    """

    MODEL_DIR = "models"

    def __init__(self, lr: float = 1e-3, curriculum_ratio: float = 0.5,
                 critic_coef: float = 0.5, entropy_coef: float = 0.01):
        self.critic_coef  = critic_coef
        self.entropy_coef = entropy_coef
        self.net          = ActorCritic().to(DEVICE)
        self.opt          = optim.Adam(self.net.parameters(), lr=lr)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        print(f"Using device: {DEVICE}")
        self.env          = KQKEnv(curriculum_ratio=curriculum_ratio)
        self.mate_pool    = self.env.mate_pool
        self.curriculum_ratio = curriculum_ratio

    def sample_action(self, logits: torch.Tensor, legal_actions: list[int]) -> tuple[int, int]:
        probs = torch.softmax(logits[legal_actions], dim=0)
        idx   = int(torch.multinomial(probs, 1).item())
        return legal_actions[idx], idx

    def gradient_update(self, all_obs: list, all_actions: list, all_returns: list) -> None:
        obs_batch = torch.from_numpy(np.stack(all_obs)).to(DEVICE)
        ret_batch = torch.tensor(all_returns, dtype=torch.float32).to(DEVICE)

        logits_batch, values = self.net(obs_batch)

        # Advantage: how much better was this action than the critic expected?
        advantages = ret_batch - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs = []
        entropies = []
        for i, (legal, chosen_idx) in enumerate(all_actions):
            probs = torch.softmax(logits_batch[i][legal], dim=0)
            log_probs.append(torch.log(probs[chosen_idx] + 1e-8))
            entropies.append(-(probs * torch.log(probs + 1e-8)).sum())

        log_probs_t = torch.stack(log_probs)
        entropy_t   = torch.stack(entropies).mean()

        actor_loss  = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values, ret_batch)
        # Entropy bonus encourages exploration — prevents policy collapsing too early.
        loss        = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_t

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.opt.step()

    def train(self, n_episodes: int = 20_000, gamma: float = 0.99,
              log_every: int = 2000, n_envs: int = 8, update_freq: int = 128):
        envs     = [KQKEnv(curriculum_ratio=self.curriculum_ratio, mate_pool=self.mate_pool) for _ in range(n_envs)]
        obs_list = np.stack([env.reset()[0] for env in envs])
        ep_bufs  = [{"obs": [], "act": [], "rew": []} for _ in range(n_envs)]

        all_obs:     list = []
        all_actions: list = []
        all_returns: list = []
        ep_count = 0

        while ep_count < n_episodes:
            with torch.no_grad():
                logits_batch, _ = self.net(torch.from_numpy(obs_list).to(DEVICE))
                logits_batch    = logits_batch.cpu()

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

    def save(self, name: str = "kqk_a2c_v4"):
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        torch.save(self.net.state_dict(), path)
        print(f"Saved → {path}")

    def load(self, name: str = "kqk_a2c_v4"):
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded ← {path}")

    def evaluate_mate_in_one(self, n_episodes: int = 500, greedy: bool = False) -> float:
        """
        Evaluate specifically from mate-in-1 positions.
        greedy=False: sample from policy distribution (reflects true probability)
        greedy=True:  always pick highest-probability legal move (reflects learned preference)
        """
        if not self.mate_pool:
            raise ValueError("No mate pool available.")
        correct = 0
        for _ in range(n_episodes):
            board = chess.Board(random.choice(self.mate_pool))
            obs   = b.board_to_obs(board)
            with torch.no_grad():
                legal_actions = [b.move_to_action(m) for m in board.legal_moves]
                logits, _     = self.net(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
                legal_logits  = logits.squeeze(0).cpu()[legal_actions]
                if greedy:
                    idx    = int(legal_logits.argmax().item())
                    action = legal_actions[idx]
                else:
                    action, _ = self.sample_action(logits.squeeze(0).cpu(), legal_actions)
            board.push(b.action_to_move(action))
            if board.is_checkmate():
                correct += 1
        mode = "greedy" if greedy else "stochastic"
        rate = correct / n_episodes
        print(f"Mate-in-1 accuracy ({mode}): {correct}/{n_episodes} = {rate*100:.1f}%")
        return rate

    def evaluate(self, n_episodes: int = 50) -> dict:
        """Run evaluation from random positions (no curriculum)."""
        counts: dict[str, int] = {}
        steps  = []
        eval_env = KQKEnv(curriculum_ratio=0.0)

        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done   = False
            info   = {}

            while not done:
                with torch.no_grad():
                    legal_actions = [b.move_to_action(m) for m in eval_env.board.legal_moves]
                    logits, _     = self.net(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
                    action, _     = self.sample_action(logits.squeeze(0).cpu(), legal_actions)
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
