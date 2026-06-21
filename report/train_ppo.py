#!/usr/bin/env python3
"""
Section 3: Train PPO from scratch and log eval checkmate rate.
Run from report/ directory:  python3 train_ppo.py
Saves: ppo_training_history.json  (list of {episode, checkmate_rate})
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


def eval_agent(agent: PPOAgent, n: int = 500) -> float:
    """Vectorised greedy eval on random positions. Returns checkmate fraction."""
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
            action, _ = _sample_action(logits_batch[j].cpu(), legal)
            obs, _, term, trunc, info = envs[i].step(action)
            obs_list[i] = obs
            if term or trunc:
                dones[i] = True
                infos[i] = info

    return sum(1 for info in infos if info.get("reason") == "checkmate") / n


def main():
    total_eps  = 300_000   # enough for PPO to converge to ~99%
    chunk_eps  = 6_000     # episodes per training chunk (~12 gradient steps)
    eval_n     = 500       # episodes per eval (vectorised, fast)

    print(f"Device: {DEVICE}")
    print(f"PPO from scratch: {total_eps:,} episodes, eval every {chunk_eps:,}\n")

    agent = PPOAgent(curriculum_ratio=0.5)

    # Evaluate at episode 0 (random weights)
    print("Episode      0  — evaluating random weights...")
    rate0 = eval_agent(agent, n=eval_n)
    history = [{"episode": 0, "checkmate_rate": round(rate0, 4)}]
    print(f"  checkmate_rate: {rate0*100:.1f}%")
    with open(OUT, "w") as f:
        json.dump(history, f)

    for ep_start in range(0, total_eps, chunk_eps):
        t0 = time.time()
        agent.train(n_episodes=chunk_eps, movement="centrum", log_every=chunk_eps * 10)
        ep_done = ep_start + chunk_eps

        rate = eval_agent(agent, n=eval_n)
        history.append({"episode": ep_done, "checkmate_rate": round(rate, 4)})
        elapsed = time.time() - t0
        print(f"Episode {ep_done:7,}  checkmate: {rate*100:.1f}%  ({elapsed:.0f}s)")

        with open(OUT, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nDone. {len(history)} eval points saved to {OUT}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
