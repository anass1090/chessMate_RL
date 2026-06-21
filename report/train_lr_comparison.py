#!/usr/bin/env python3
"""
Section 7: LR comparison from scratch — logs checkmate fraction per batch.
Uses the v1 REINFORCE agent (no multiprocessing) so it runs alongside Section 3.
Run from report/ directory:  python3 train_lr_comparison.py
Overwrites: lr_comparison.json  (keys "0.01", "0.001", "0.0001" → list of per-update checkmate fractions)
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import utils.board as b
from environment.v3.kqk_env import KQKEnv

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPISODES = 30_000
N_ENVS     = 64
UPDATE_FREQ = 1024        # collect this many steps before a gradient update
GAMMA       = 0.99
CURRICULUM  = 0.5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lr_comparison.json")

print(f"Device: {DEVICE}")


class Policy(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(b.OBS_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 4096),
        )
    def forward(self, x): return self.net(x)


def train_lr(lr: float) -> list[float]:
    """Train REINFORCE for N_EPISODES, return checkmate_fraction per gradient update."""
    policy = Policy().to(DEVICE)
    opt    = optim.Adam(policy.parameters(), lr=lr)

    _env       = KQKEnv(curriculum_ratio=CURRICULUM)
    mate_pool  = _env.mate_pool
    envs       = [KQKEnv(curriculum_ratio=CURRICULUM, mate_pool=mate_pool) for _ in range(N_ENVS)]
    obs_list   = np.stack([env.reset()[0] for env in envs])
    ep_bufs    = [{"obs": [], "act": [], "rew": [], "checkmate": False} for _ in range(N_ENVS)]

    all_obs, all_acts, all_rets = [], [], []
    checkmate_fracs = []
    ep_count   = 0
    batch_checkmates = 0
    batch_eps  = 0

    while ep_count < N_EPISODES:
        with torch.no_grad():
            logits_batch = policy(torch.FloatTensor(obs_list).to(DEVICE))

        for i, env in enumerate(envs):
            if ep_count >= N_EPISODES:
                break

            legal  = [b.move_to_action(m) for m in env.board.legal_moves]
            probs  = torch.softmax(logits_batch[i][legal], dim=0)
            idx    = int(torch.multinomial(probs, 1).item())
            action = legal[idx]

            ep_bufs[i]["obs"].append(obs_list[i].copy())
            ep_bufs[i]["act"].append((legal, idx))

            obs, reward, terminated, truncated, info = env.step(action)
            ep_bufs[i]["rew"].append(reward)
            obs_list[i] = obs

            if terminated or truncated:
                if info.get("reason") == "checkmate":
                    ep_bufs[i]["checkmate"] = True

                G, rets = 0.0, []
                for r in reversed(ep_bufs[i]["rew"]):
                    G = r + GAMMA * G
                    rets.insert(0, G)
                ret_t = torch.tensor(rets, dtype=torch.float32)
                ret_t = (ret_t - ret_t.mean()) / (ret_t.std(correction=0) + 1e-8)

                all_obs.extend(ep_bufs[i]["obs"])
                all_acts.extend(ep_bufs[i]["act"])
                all_rets.extend(ret_t.tolist())

                batch_checkmates += int(ep_bufs[i]["checkmate"])
                batch_eps += 1
                ep_count  += 1

                obs_list[i] = env.reset()[0]
                ep_bufs[i]  = {"obs": [], "act": [], "rew": [], "checkmate": False}

        if len(all_obs) >= UPDATE_FREQ:
            obs_t  = torch.FloatTensor(np.stack(all_obs)).to(DEVICE)
            ret_t  = torch.tensor(all_rets, dtype=torch.float32).to(DEVICE)
            logits = policy(obs_t)

            B    = len(all_acts)
            mask = torch.full((B, 4096), float('-inf'), device=DEVICE)
            chosen = torch.zeros(B, dtype=torch.long, device=DEVICE)
            for i, (legal, ci) in enumerate(all_acts):
                mask[i, legal] = 0.0
                chosen[i] = legal[ci]
            log_probs = torch.log_softmax(logits + mask, dim=-1).gather(1, chosen.unsqueeze(1)).squeeze(1)

            loss = -(log_probs * ret_t).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            frac = batch_checkmates / max(batch_eps, 1)
            checkmate_fracs.append(round(frac, 4))

            all_obs, all_acts, all_rets = [], [], []
            batch_checkmates = 0
            batch_eps = 0

    return checkmate_fracs


lrs = {"0.01": 1e-2, "0.001": 1e-3, "0.0001": 1e-4}
results = {}

for key, lr in lrs.items():
    print(f"\n=== lr={lr} ===")
    t0 = time.time()
    fracs = train_lr(lr)
    elapsed = time.time() - t0
    results[key] = fracs
    final = np.convolve(fracs, np.ones(20)/20, "valid")[-1] if len(fracs) >= 20 else np.mean(fracs)
    print(f"  {len(fracs)} updates  final smoothed checkmate frac: {final*100:.1f}%  ({elapsed:.0f}s)")

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

print(f"\nDone. Saved {OUT}")
