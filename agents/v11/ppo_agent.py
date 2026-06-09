import json
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
from environment.v10.kqk_env import KQKEnv

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EVAL_SET_PATH = os.path.join(_AGENT_DIR, "eval_set.json")
_DEFAULT_HISTORY_PATH  = os.path.join(_AGENT_DIR, "train_history.json")


def generate_eval_set(n: int = 1000, seed: int = 42,
                      path: str = None) -> list[str]:
    """Fixed held-out KQK FENs; saved on first call and reloaded after."""
    path = path or _DEFAULT_EVAL_SET_PATH
    if os.path.exists(path):
        with open(path) as f:
            fens = json.load(f)
        print(f"Loaded eval set ({len(fens)} positions) ← {path}")
        return fens

    rng = random.Random(seed)
    fens = []
    while len(fens) < n:
        fens.append(b.random_kqk_position().fen())
    rng.shuffle(fens)

    with open(path, "w") as f:
        json.dump(fens, f)
    print(f"Generated eval set ({len(fens)} positions) → {path}")
    return fens


def load_train_history(path: str = None) -> dict:
    """Load a training history dict saved by train()."""
    path = path or _DEFAULT_HISTORY_PATH
    with open(path) as f:
        return json.load(f)


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


class PPOAgent:
    """
    PPO agent for the KQK environment — v11.

    Like v10 but opponent movement is controlled by movement_ratio (float,
    0.0 = always centrum, 1.0 = always random) instead of a fixed string.
    movement_ratio is stored on the agent and used as the default for train().
    """

    MODEL_DIR = "models"

    def __init__(self, lr: float = 1e-3, curriculum_ratio: float = 0.5,
                 movement_ratio: float = 0.5,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 n_workers: int = 31, episodes_per_worker: int = 16):
        self.entropy_coef        = entropy_coef
        self.value_coef          = value_coef
        self.curriculum_ratio    = curriculum_ratio
        self.movement_ratio      = movement_ratio
        self.n_workers           = n_workers
        self.episodes_per_worker = episodes_per_worker
        self.net                 = ActorCritic().to(DEVICE)
        self.opt                 = optim.Adam(self.net.parameters(), lr=lr)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        print(f"Using device: {DEVICE}")

        _env           = KQKEnv(curriculum_ratio=curriculum_ratio)
        self.mate_pool = _env.mate_pool
        del _env

    def _state_dict_np(self) -> dict:
        return {k: v.cpu().numpy() for k, v in self.net.state_dict().items()}

    def gradient_update(self, all_obs: np.ndarray, all_actions: list,
                        all_returns: np.ndarray,
                        ppo_epochs: int = 4, clip_eps: float = 0.2) -> None:
        obs_t = torch.from_numpy(all_obs).to(DEVICE)
        ret_t = torch.from_numpy(all_returns).to(DEVICE)
        B     = len(all_actions)

        # Vectorized mask + chosen — single GPU index_put_, no per-row loop
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

    def train(self, n_episodes: int = 100_000, gamma: float = 0.99,
              movement_ratio: float | None = None, log_every: int | None = None,
              n_workers: int | None = None, episodes_per_worker: int | None = None,
              ppo_epochs: int = 4, clip_eps: float = 0.2,
              eval_every: int = 500, eval_fens: list[str] | None = None,
              history_path: str | None = None) -> dict:
        """
        Returns a dict with keys:
          train_rewards : list of mean batch returns (one per gradient update)
          eval_history  : list of {episode, checkmate_rate, mean_steps} dicts
          eval_set_path : path used for the eval set (or None)
        The dict is also saved to disk as train_history.json.

        movement_ratio defaults to self.movement_ratio if not passed explicitly.
        """
        if movement_ratio is None:
            movement_ratio = self.movement_ratio
        if log_every is None:
            log_every = int(n_episodes / 10)

        n_workers           = n_workers           or self.n_workers
        episodes_per_worker = episodes_per_worker or self.episodes_per_worker

        ctx       = mp.get_context("spawn")
        weight_qs = [ctx.Queue(maxsize=1) for _ in range(n_workers)]
        result_q  = ctx.Queue()

        workers = [
            ctx.Process(
                target=_worker_loop,
                args=(weight_qs[i], result_q, self.mate_pool,
                      self.curriculum_ratio, episodes_per_worker, gamma, movement_ratio),
                daemon=True,
            )
            for i in range(n_workers)
        ]
        for p in workers:
            p.start()

        ep_count      = 0
        sd_np         = self._state_dict_np()
        train_rewards: list[float] = []
        eval_history:  list[dict]  = []

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

                if eval_fens is not None and ep_count % eval_every < n_workers * episodes_per_worker:
                    stats = self.evaluate_on_fens(eval_fens, movement_ratio=movement_ratio)
                    eval_history.append({
                        "episode":        ep_count,
                        "checkmate_rate": stats["checkmate_rate"],
                        "mean_steps":     stats["mean_steps"],
                    })
                    print(f"  [eval] ep={ep_count}  "
                          f"checkmate={stats['checkmate_rate']*100:.1f}%  "
                          f"mean_steps={stats['mean_steps']:.1f}")

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

        history = {
            "train_rewards": train_rewards,
            "eval_history":  eval_history,
            "eval_set_path": str(_DEFAULT_EVAL_SET_PATH) if eval_fens is not None else None,
        }
        save_path = history_path or _DEFAULT_HISTORY_PATH
        with open(save_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved → {save_path}")
        return history

    def render_game(self, fen: str | None = None, greedy: bool = True,
                    movement_ratio: float = 0.0) -> tuple[list, list]:
        env = KQKEnv(curriculum_ratio=0.0, movement_ratio=movement_ratio)
        if fen:
            env.board = chess.Board(fen)
            env.step_count = 0
        else:
            env.reset()

        frames = [env.board.copy()]
        labels = ["start"]
        done   = False

        while not done:
            legal_actions = [b.move_to_action(m) for m in env.board.legal_moves]

            with torch.no_grad():
                obs       = b.board_to_obs(env.board)
                logits, _ = self.net(torch.FloatTensor(obs).unsqueeze(0).to(DEVICE))
                logits    = logits.squeeze(0).cpu()

            if greedy:
                idx    = int(logits[legal_actions].argmax().item())
                action = legal_actions[idx]
            else:
                action, _ = _sample_action(logits, legal_actions)

            move = b.action_to_move(action)
            san  = env.board.san(move)

            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            frames.append(env.board.copy())
            reason = info.get("reason", "")
            labels.append(f"{san}  {'— ' + reason if reason else ''}")

        return frames, labels

    def animate_game(self, fen: str | None = None, greedy: bool = True,
                     movement_ratio: float = 0.0, size: int = 400,
                     interval: int = 600) -> None:
        import chess.svg
        import ipywidgets as widgets
        from IPython.display import display, SVG

        frames, labels = self.render_game(fen=fen, greedy=greedy, movement_ratio=movement_ratio)
        svgs = [chess.svg.board(board, size=size) for board in frames]

        play   = widgets.Play(min=0, max=len(svgs) - 1, step=1,
                              interval=interval, description="Play")
        slider = widgets.IntSlider(min=0, max=len(svgs) - 1, step=1,
                                   layout=widgets.Layout(width="400px"))
        label  = widgets.Label(value=labels[0])
        output = widgets.Output()
        widgets.jslink((play, "value"), (slider, "value"))

        with output:
            display(SVG(svgs[0]))

        def on_change(change):
            i = change["new"]
            with output:
                output.clear_output(wait=True)
                display(SVG(svgs[i]))
            label.value = f"Move {i}: {labels[i]}"

        slider.observe(on_change, names="value")
        display(widgets.VBox([widgets.HBox([play, slider]), label, output]))

    def save(self, name: str = "kqk_ppo_v11") -> None:
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        torch.save(self.net.state_dict(), path)
        print(f"Saved → {path}")

    def load(self, name: str = "kqk_ppo_v11") -> None:
        path = os.path.join(self.MODEL_DIR, f"{name}.pt")
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"Loaded ← {path}")

    def evaluate_mate_in_one(self, n_episodes: int = 500, greedy: bool = False) -> float:
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

    def evaluate(self, n_episodes: int = 50, movement_ratio: float = 0.0) -> dict:
        envs     = [KQKEnv(curriculum_ratio=0.0, movement_ratio=movement_ratio)
                    for _ in range(n_episodes)]
        obs_list = [env.reset()[0] for env in envs]
        dones    = [False] * n_episodes
        infos    = [{}] * n_episodes
        steps    = [0] * n_episodes

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
        print(f"\n--- Eval ({n_episodes} eps, movement_ratio={movement_ratio}) ---")
        for reason, count in sorted(counts.items()):
            print(f"  {reason:30s}: {count}")
        print(f"  {'mean_steps':30s}: {results['mean_steps']:.1f}")
        print(f"  {'checkmate_rate':30s}: {results['checkmate_rate']*100:.1f}%")
        return results

    def evaluate_on_fens(self, fens: list[str], movement_ratio: float = 0.0) -> dict:
        """Greedy episode per FEN; fixed positions make results comparable across checkpoints."""
        envs     = [KQKEnv(curriculum_ratio=0.0, movement_ratio=movement_ratio) for _ in fens]
        obs_list = []
        for env, fen in zip(envs, fens):
            env.reset()
            env.board      = chess.Board(fen)
            env.step_count = 0
            obs_list.append(b.board_to_obs(env.board))

        dones = [False] * len(fens)
        infos = [{}]    * len(fens)
        steps = [0]     * len(fens)

        while not all(dones):
            active    = [i for i, d in enumerate(dones) if not d]
            obs_batch = torch.FloatTensor(
                np.stack([obs_list[i] for i in active])
            ).to(DEVICE)

            with torch.no_grad():
                logits_batch, _ = self.net(obs_batch)

            for j, i in enumerate(active):
                legal_actions             = [b.move_to_action(m) for m in envs[i].board.legal_moves]
                logits                    = logits_batch[j].cpu()
                idx                       = int(logits[legal_actions].argmax().item())
                action                    = legal_actions[idx]
                obs, _, term, trunc, info = envs[i].step(action)
                obs_list[i]               = obs
                if term or trunc:
                    dones[i] = True
                    infos[i] = info
                    steps[i] = envs[i].step_count

        counts: dict[str, int] = {}
        for info in infos:
            reason = info.get("reason", "timeout")
            counts[reason] = counts.get(reason, 0) + 1

        n = len(fens)
        return {
            **counts,
            "mean_steps":     float(np.mean(steps)),
            "checkmate_rate": counts.get("checkmate", 0) / n,
        }


def _sample_action(logits: torch.Tensor, legal: list[int]) -> tuple[int, int]:
    raw   = torch.nan_to_num(logits[legal].float(), nan=0.0, posinf=1e4, neginf=-1e4)
    probs = torch.softmax(raw, dim=0)
    idx   = int(torch.multinomial(probs, 1).item())
    return legal[idx], idx


def _worker_loop(weight_q: mp.Queue, result_q: mp.Queue,
                 mate_pool: list, curriculum_ratio: float,
                 eps_per_batch: int, gamma: float,
                 movement_ratio: float) -> None:
    import sys
    sys.path.insert(0, ".")

    torch.set_num_threads(1)

    net = ActorCritic()
    net.eval()
    env = KQKEnv(curriculum_ratio=curriculum_ratio, mate_pool=mate_pool,
                 movement_ratio=movement_ratio)

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
