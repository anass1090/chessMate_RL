import os
import random
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import chess
import utils.board as b
from environment.v4.kqk_opponent_env import KQKOpponentEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_AGENT_DIR            = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_HISTORY_PATH = os.path.join(_AGENT_DIR, "opponent_train_history.json")


class ActorCritic(nn.Module):
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


class PPOOpponent:
    """
    PPO agent for the black king in KQK — tries to survive and draw.

    train(white_agent) snapshots the white agent's weights at the start of
    the run and uses them as a frozen white policy for the duration. Call
    train() again with a refreshed white_agent to iterate.
    """

    MODEL_DIR = "models"

    def __init__(self, lr: float = 1e-3,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 n_workers: int = 31, episodes_per_worker: int = 16):
        self.entropy_coef        = entropy_coef
        self.value_coef          = value_coef
        self.n_workers           = n_workers
        self.episodes_per_worker = episodes_per_worker
        self.net                 = ActorCritic().to(DEVICE)
        self.opt                 = optim.Adam(self.net.parameters(), lr=lr)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        print(f"Using device: {DEVICE}")

    def _state_dict_np(self) -> dict:
        return {k: v.cpu().numpy() for k, v in self.net.state_dict().items()}

    def gradient_update(self, all_obs: np.ndarray, all_actions: list,
                        all_returns: np.ndarray,
                        ppo_epochs: int = 4, clip_eps: float = 0.2) -> None:
        obs_t = torch.from_numpy(all_obs).to(DEVICE)
        ret_t = torch.from_numpy(all_returns).to(DEVICE)
        B     = len(all_actions)

        row_idx = torch.cat([torch.full((len(legal),), i, dtype=torch.long)
                              for i, (legal, _) in enumerate(all_actions)])
        col_idx = torch.cat([torch.tensor(legal, dtype=torch.long)
                              for legal, _ in all_actions])
        chosen  = torch.tensor([legal[idx] for legal, idx in all_actions],
                                dtype=torch.long, device=DEVICE)

        mask = torch.full((B, 4096), float('-inf'), device=DEVICE)
        mask[row_idx.to(DEVICE), col_idx.to(DEVICE)] = 0.0

        with torch.no_grad():
            old_logits, old_values = self.net(obs_t)
            old_log_probs = (F.log_softmax(old_logits + mask, dim=-1)
                             .gather(1, chosen.unsqueeze(1)).squeeze(1))
            advantages = ret_t - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(ppo_epochs):
            logits, values = self.net(obs_t)
            log_probs_all  = F.log_softmax(logits + mask, dim=-1)
            log_probs      = log_probs_all.gather(1, chosen.unsqueeze(1)).squeeze(1)

            ratio      = (log_probs - old_log_probs).exp()
            clipped    = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
            actor_loss = -torch.min(ratio * advantages, clipped).mean()
            value_loss = F.mse_loss(values, ret_t)
            entropy    = -(log_probs_all.exp() * log_probs_all.nan_to_num(neginf=0.0)).sum(-1).mean()

            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.opt.step()

    def train(self, white_agent, n_episodes: int = 100_000, gamma: float = 0.99,
              log_every: int | None = None,
              n_workers: int | None = None, episodes_per_worker: int | None = None,
              ppo_epochs: int = 4, clip_eps: float = 0.2,
              history_path: str | None = None) -> dict:
        """
        Train the opponent against a frozen snapshot of white_agent.

        white_agent must be a PPOAgent (or any object with _state_dict_np()).
        Its weights are snapshotted at the start and held fixed for this run.
        """
        if log_every is None:
            log_every = int(n_episodes / 10)

        n_workers           = n_workers           or self.n_workers
        episodes_per_worker = episodes_per_worker or self.episodes_per_worker

        white_weights = white_agent._state_dict_np()

        ctx       = mp.get_context("spawn")
        weight_qs = [ctx.Queue(maxsize=1) for _ in range(n_workers)]
        result_q  = ctx.Queue()

        workers = [
            ctx.Process(
                target=_worker_loop,
                args=(weight_qs[i], result_q, episodes_per_worker, gamma, white_weights),
                daemon=True,
            )
            for i in range(n_workers)
        ]
        for p in workers:
            p.start()

        ep_count      = 0
        sd_np         = self._state_dict_np()
        train_rewards: list[float] = []

        for q in weight_qs:
            q.put(sd_np)

        try:
            while ep_count < n_episodes:
                obs_parts, all_actions, ret_parts = [], [], []
                for _ in range(n_workers):
                    obs_arr, acts, rets_arr = result_q.get()
                    obs_parts.append(obs_arr)
                    all_actions.extend(acts)
                    ret_parts.append(rets_arr)

                all_rets = np.concatenate(ret_parts)
                train_rewards.append(float(all_rets.mean()))

                self.gradient_update(
                    np.concatenate(obs_parts),
                    all_actions,
                    all_rets,
                    ppo_epochs=ppo_epochs,
                    clip_eps=clip_eps,
                )
                ep_count += n_workers * episodes_per_worker

                if ep_count % log_every < n_workers * episodes_per_worker:
                    print(f"Episode {ep_count}/{n_episodes}  "
                          f"mean_return={train_rewards[-1]:.3f}")

                if ep_count < n_episodes:
                    sd_np = self._state_dict_np()
                    for q in weight_qs:
                        q.put(sd_np)

        finally:
            for q in weight_qs:
                try:
                    q.put(None)
                except Exception:
                    pass
            for p in workers:
                p.join(timeout=3)
                if p.is_alive():
                    p.terminate()

        import json
        history = {"train_rewards": train_rewards}
        save_path = history_path or _DEFAULT_HISTORY_PATH
        with open(save_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved → {save_path}")
        return history

    def evaluate(self, white_agent, n_episodes: int = 500) -> dict:
        """Evaluate the opponent against white_agent (stochastic black, greedy white)."""
        white_fn = _make_white_fn(white_agent._state_dict_np())

        envs     = [KQKOpponentEnv(white_fn=white_fn) for _ in range(n_episodes)]
        obs_list = [env.reset()[0] for env in envs]
        dones    = [False] * n_episodes
        infos    = [{}]    * n_episodes
        steps    = [0]     * n_episodes

        while not all(dones):
            active    = [i for i, d in enumerate(dones) if not d]
            obs_batch = torch.FloatTensor(
                np.stack([obs_list[i] for i in active])
            ).to(DEVICE)

            with torch.no_grad():
                logits_batch, _ = self.net(obs_batch)

            for j, i in enumerate(active):
                legal_actions             = [b.move_to_action(m) for m in envs[i].board.legal_moves]
                action, _                 = _sample_action(logits_batch[j].cpu(), legal_actions)
                obs, _, term, trunc, info = envs[i].step(action)
                obs_list[i]               = obs
                if term or trunc:
                    dones[i] = True
                    infos[i] = info if term else {"reason": "truncated"}
                    steps[i] = envs[i].step_count

        counts: dict[str, int] = {}
        for info in infos:
            reason = info.get("reason", "truncated")
            counts[reason] = counts.get(reason, 0) + 1

        draw_reasons = {"stalemate", "insufficient_material", "threefold_repetition", "fifty_moves"}
        n_draws = sum(counts.get(r, 0) for r in draw_reasons)

        results = {
            **counts,
            "mean_steps":     float(np.mean(steps)),
            "checkmate_rate": counts.get("checkmate", 0) / n_episodes,
            "draw_rate":      n_draws / n_episodes,
        }
        print(f"\n--- Opponent Eval ({n_episodes} eps) ---")
        for reason, count in sorted(counts.items()):
            print(f"  {reason:30s}: {count}")
        print(f"  {'mean_steps':30s}: {results['mean_steps']:.1f}")
        print(f"  {'checkmate_rate':30s}: {results['checkmate_rate']*100:.1f}%")
        print(f"  {'draw_rate':30s}: {results['draw_rate']*100:.1f}%")

    def save(self, name: str = "kqk_opponent_v1") -> None:
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        torch.save(self.net.state_dict(), path)
        print(f"Saved → {path}")

    def load(self, name: str = "kqk_opponent_v1") -> None:
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded ← {path}")


def _make_white_fn(weights: dict):
    """Build a callable white policy from a state-dict snapshot."""
    net = ActorCritic()
    net.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    net.eval()

    def white_fn(obs: np.ndarray, legal: list[int]) -> int:
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(obs).unsqueeze(0))
        idx    = int(logits.squeeze(0)[legal].argmax().item())
        return legal[idx]

    return white_fn


def _sample_action(logits: torch.Tensor, legal: list[int]) -> tuple[int, int]:
    raw   = torch.nan_to_num(logits[legal].float(), nan=0.0, posinf=1e4, neginf=-1e4)
    probs = torch.softmax(raw, dim=0)
    idx   = int(torch.multinomial(probs, 1).item())
    return legal[idx], idx


def _worker_loop(weight_q: mp.Queue, result_q: mp.Queue,
                 eps_per_batch: int, gamma: float,
                 white_weights: dict) -> None:
    import sys
    sys.path.insert(0, ".")

    torch.set_num_threads(1)

    # Black network — updated each batch
    net = ActorCritic()
    net.eval()

    # White network — frozen for the whole run
    white_net = ActorCritic()
    white_net.load_state_dict({k: torch.from_numpy(v) for k, v in white_weights.items()})
    white_net.eval()

    def white_fn(obs: np.ndarray, legal: list[int]) -> int:
        with torch.no_grad():
            logits, _ = white_net(torch.from_numpy(obs).unsqueeze(0))
        idx = int(logits.squeeze(0)[legal].argmax().item())
        return legal[idx]

    env = KQKOpponentEnv(white_fn=white_fn)

    while True:
        msg = weight_q.get()
        if msg is None:
            break

        net.load_state_dict({k: torch.from_numpy(v) for k, v in msg.items()})
        net.eval()

        all_obs, all_actions, all_returns = [], [], []

        for _ in range(eps_per_batch):
            obs, _ = env.reset()
            ep_obs, ep_acts, ep_rews = [], [], []
            done = False

            while not done:
                legal = [b.move_to_action(m) for m in env.board.legal_moves]
                with torch.no_grad():
                    logits, _ = net(torch.from_numpy(obs).unsqueeze(0))
                    logits    = logits.squeeze(0)
                action, idx = _sample_action(logits, legal)

                ep_obs.append(obs.copy())
                ep_acts.append((legal, idx))

                obs, reward, terminated, truncated, _ = env.step(action)
                ep_rews.append(reward)
                done = terminated or truncated

            G, rets = 0.0, []
            for r in reversed(ep_rews):
                G = r + gamma * G
                rets.insert(0, G)

            all_obs.extend(ep_obs)
            all_actions.extend(ep_acts)
            all_returns.extend(rets)

        result_q.put((
            np.stack(all_obs),
            all_actions,
            np.array(all_returns, dtype=np.float32),
        ))
