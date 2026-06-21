#!/usr/bin/env python3
"""
Section 3: Proper from-scratch training curve showing the aha-moment.
Trains PPO with curriculum annealing and logs GREEDY general checkmate rate
every 5k episodes so the jump from ~0% to ~97%+ appears in the curve.

Run from report/ directory:  python3 train_section3.py
Saves: ppo_training_history.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import multiprocessing as mp

import utils.board as b
from agents.v6.ppo_agent import PPOAgent, _sample_action
from environment.v3.kqk_env import KQKEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_training_history.json")


def eval_greedy(agent: PPOAgent, n: int = 300) -> float:
    """Greedy eval on random positions vs centrum opponent. Returns checkmate fraction."""
    envs     = [KQKEnv(curriculum_ratio=0.0, movement="centrum") for _ in range(n)]
    obs_list = [env.reset()[0] for env in envs]
    dones    = [False] * n
    infos    = [{}]    * n

    while not all(dones):
        active    = [i for i, d in enumerate(dones) if not d]
        obs_batch = torch.FloatTensor(np.stack([obs_list[i] for i in active])).to(DEVICE)
        with torch.no_grad():
            logits_batch, _ = agent.net(obs_batch)
        for j, i in enumerate(active):
            legal = [b.move_to_action(m) for m in envs[i].board.legal_moves]
            # greedy: pick highest logit over legal moves
            idx    = int(logits_batch[j].cpu()[legal].argmax().item())
            action = legal[idx]
            obs, _, term, trunc, info = envs[i].step(action)
            obs_list[i] = obs
            if term or trunc:
                dones[i] = True
                infos[i] = info

    return sum(1 for info in infos if info.get("reason") == "checkmate") / n


def train_phase(agent, n_episodes, chunk_size, ep_offset, history):
    """Train for n_episodes in chunks, eval and save after each chunk."""
    for chunk_start in range(0, n_episodes, chunk_size):
        t0 = time.time()
        agent.train(n_episodes=chunk_size, movement="centrum", log_every=chunk_size * 100)
        ep = ep_offset + chunk_start + chunk_size

        rate = eval_greedy(agent, n=300)
        history.append({"episode": ep, "checkmate_rate": round(rate, 4)})
        elapsed = time.time() - t0
        print(f"  ep {ep:7,}  ratio={agent.curriculum_ratio:.1f}  checkmate: {rate*100:.1f}%  ({elapsed:.0f}s)")

        with open(OUT, "w") as f:
            json.dump(history, f, indent=2)

    return ep_offset + n_episodes


def main():
    # Curriculum schedule — mirrors the real nb10/nb11 training but scaled down
    # to ~700k total episodes so it runs in ~15 minutes.
    # The aha-moment (general checkmate jumps from ~0% to 90%+) happens when
    # the agent finally generalises from mate-in-1 patterns to full-game positions.
    schedule = [
        (1.0, 200_000),   # teach mate patterns; general checkmate stays ~0%
        (0.5, 150_000),   # mix in random starts; generalisation begins
        (0.2, 150_000),   # mostly random; aha-moment expected here
        (0.0, 150_000),   # full random only; locks in generalised play
    ]
    chunk_size = 5_000

    print(f"Device: {DEVICE}")
    total = sum(n for _, n in schedule)
    print(f"From-scratch PPO: {total:,} episodes, eval every {chunk_size:,}\n")

    agent = PPOAgent(curriculum_ratio=1.0)

    # Episode 0 baseline
    print("Episode      0  — baseline (random weights) ...")
    rate0 = eval_greedy(agent, n=300)
    history = [{"episode": 0, "checkmate_rate": round(rate0, 4)}]
    print(f"  checkmate: {rate0*100:.1f}%")
    with open(OUT, "w") as f:
        json.dump(history, f)

    ep_offset = 0
    for ratio, n_eps in schedule:
        print(f"\n--- Phase: ratio={ratio:.1f}  ({n_eps:,} episodes) ---")
        agent.curriculum_ratio = ratio
        ep_offset = train_phase(agent, n_eps, chunk_size, ep_offset, history)

    print(f"\nDone. {len(history)} eval points  →  {OUT}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
