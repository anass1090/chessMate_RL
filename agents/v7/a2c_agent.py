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
from environment.v6.kqk_env import KQKEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class A2CAgent:
    """
    A2C agent for the KQK environment — v7.

    Adds to v6: persistent worker processes for parallel episode collection
    (CPU inference) and a vectorized gradient update (masked softmax on GPU).
    """

    MODEL_DIR = "models"

    def __init__(self, lr: float = 1e-3, curriculum_ratio: float = 0.5,
                 critic_coef: float = 0.5, entropy_coef: float = 0.01,
                 n_workers: int = 31, episodes_per_worker: int = 8):
        self.critic_coef        = critic_coef
        self.entropy_coef       = entropy_coef
        self.curriculum_ratio   = curriculum_ratio
        self.n_workers          = n_workers
        self.episodes_per_worker = episodes_per_worker
        self.net = ActorCritic().to(DEVICE)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        print(f"Using device: {DEVICE}")

        _env           = KQKEnv(curriculum_ratio=curriculum_ratio)
        self.mate_pool = _env.mate_pool
        del _env

    def _state_dict_np(self) -> dict:
        return {k: v.cpu().numpy() for k, v in self.net.state_dict().items()}

    def gradient_update(self, all_obs: np.ndarray, all_actions: list,
                        all_returns: np.ndarray) -> None:
        """Vectorized A2C update using masked softmax over legal moves."""
        obs_batch = torch.from_numpy(all_obs).to(DEVICE)
        ret_batch = torch.from_numpy(all_returns).to(DEVICE)

        logits_batch, values = self.net(obs_batch)

        advantages = ret_batch - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        B = len(all_actions)
        mask   = torch.full((B, 4096), float('-inf'), device=DEVICE)
        chosen = torch.zeros(B, dtype=torch.long, device=DEVICE)
        for i, (legal, chosen_idx) in enumerate(all_actions):
            mask[i, legal] = 0.0
            chosen[i]      = legal[chosen_idx]

        log_probs_all = F.log_softmax(logits_batch + mask, dim=-1)
        log_probs_t   = log_probs_all.gather(1, chosen.unsqueeze(1)).squeeze(1)
        # neginf=0 prevents 0*-inf=nan in backward (exp(-inf)=0, log(0)=0 by convention)
        entropy_t     = -(log_probs_all.exp() * log_probs_all.nan_to_num(neginf=0.0)).sum(-1).mean()

        actor_loss  = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values, ret_batch)
        loss        = actor_loss + self.critic_coef * critic_loss - self.entropy_coef * entropy_t

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
        self.opt.step()

    def train(self, n_episodes: int = 20_000, gamma: float = 0.99,
              log_every: int = 2000, n_workers: int | None = None,
              episodes_per_worker: int | None = None) -> None:
        """Spawn persistent workers once, then alternate: collect → update → send weights."""
        n_workers          = n_workers          or self.n_workers
        episodes_per_worker = episodes_per_worker or self.episodes_per_worker

        ctx       = mp.get_context("spawn")
        weight_qs = [ctx.Queue(maxsize=1) for _ in range(n_workers)]
        result_q  = ctx.Queue()

        workers = [
            ctx.Process(
                target=_worker_loop,
                args=(weight_qs[i], result_q, self.mate_pool,
                      self.curriculum_ratio, episodes_per_worker, gamma),
                daemon=True,
            )
            for i in range(n_workers)
        ]
        for p in workers:
            p.start()

        ep_count = 0
        sd_np    = self._state_dict_np()
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

                self.gradient_update(
                    np.concatenate(obs_parts),
                    all_actions,
                    np.concatenate(ret_parts),
                )
                ep_count += n_workers * episodes_per_worker

                if ep_count % log_every < n_workers * episodes_per_worker:
                    print(f"Episode {ep_count}/{n_episodes}")

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

    def _sample_action(self, logits: torch.Tensor, legal_actions: list[int]) -> tuple[int, int]:
        return _sample_action(logits, legal_actions)

    def save(self, name: str = "kqk_a2c_v7") -> None:
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        torch.save(self.net.state_dict(), path)
        print(f"Saved → {path}")

    def load(self, name: str = "kqk_a2c_v7") -> None:
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded ← {path}")

    def evaluate_mate_in_one(self, n_episodes: int = 500, greedy: bool = False) -> float:
        """One batched forward pass for all episodes instead of one per episode."""
        if not self.mate_pool:
            raise ValueError("No mate pool available.")

        boards    = [chess.Board(random.choice(self.mate_pool)) for _ in range(n_episodes)]
        obs_batch = torch.FloatTensor(
            np.stack([b.board_to_obs(board) for board in boards])
        ).to(DEVICE)

        with torch.no_grad():
            logits_batch, _ = self.net(obs_batch)

        correct = 0
        for i, board in enumerate(boards):
            legal_actions = [b.move_to_action(m) for m in board.legal_moves]
            logits        = logits_batch[i].cpu()
            if greedy:
                idx    = int(logits[legal_actions].argmax().item())
                action = legal_actions[idx]
            else:
                action, _ = _sample_action(logits, legal_actions)
            board.push(b.action_to_move(action))
            if board.is_checkmate():
                correct += 1

        mode = "greedy" if greedy else "stochastic"
        rate = correct / n_episodes
        print(f"Mate-in-1 accuracy ({mode}): {correct}/{n_episodes} = {rate*100:.1f}%")
        return rate

    def evaluate(self, n_episodes: int = 50) -> dict:
        """Run all episodes in parallel with batched inference per step."""
        envs     = [KQKEnv(curriculum_ratio=0.0) for _ in range(n_episodes)]
        obs_list = [env.reset()[0] for env in envs]
        dones    = [False] * n_episodes
        infos    = [{}] * n_episodes
        steps    = [0] * n_episodes

        while not all(dones):
            active = [i for i, d in enumerate(dones) if not d]
            obs_batch = torch.FloatTensor(
                np.stack([obs_list[i] for i in active])
            ).to(DEVICE)

            with torch.no_grad():
                logits_batch, _ = self.net(obs_batch)

            for j, i in enumerate(active):
                legal_actions       = [b.move_to_action(m) for m in envs[i].board.legal_moves]
                action, _           = _sample_action(logits_batch[j].cpu(), legal_actions)
                obs, _, term, trunc, info = envs[i].step(action)
                obs_list[i]         = obs
                if term or trunc:
                    dones[i] = True
                    infos[i] = info
                    steps[i] = envs[i].step_count

        counts: dict[str, int] = {}
        for info in infos:
            reason = info.get("reason", "timeout")
            counts[reason] = counts.get(reason, 0) + 1

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


def _sample_action(logits: torch.Tensor, legal: list[int]) -> tuple[int, int]:
    raw   = torch.nan_to_num(logits[legal].float(), nan=0.0, posinf=1e4, neginf=-1e4)
    probs = torch.softmax(raw, dim=0)
    idx   = int(torch.multinomial(probs, 1).item())
    return legal[idx], idx


def _worker_loop(weight_q: mp.Queue, result_q: mp.Queue,
                 mate_pool: list, curriculum_ratio: float,
                 eps_per_batch: int, gamma: float) -> None:
    """Persistent worker: receives weights, collects episodes, sends results back."""
    import sys
    sys.path.insert(0, ".")

    torch.set_num_threads(1)  # prevent OMP contention when many workers run in parallel

    net = ActorCritic()
    net.eval()
    env = KQKEnv(curriculum_ratio=curriculum_ratio, mate_pool=mate_pool)

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
                action, idx = _sample_action(logits.squeeze(0), legal)

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
