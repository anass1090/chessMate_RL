#!/usr/bin/env python3
"""
Full PPO curriculum training from scratch.
Logs general checkmate rate AND mate-in-1 accuracy every EVAL_EVERY episodes.
Schedule mirrors the original exploration notebooks (nb10/nb11): 6 phases × 500k eps.

Output: report/ppo_full_history.json
  [{"episode": int, "ratio": float, "checkmate_rate": float, "mate_in_one": float}, ...]
"""
import sys, os, json, time, random, chess
# report/ must come FIRST so workers find report/environment/v3/kqk_env.py (with movement param)
_here   = os.path.dirname(os.path.abspath(__file__))
_root   = os.path.dirname(_here)
sys.path.insert(0, _root)   # chessMate/ — for utils.board
sys.path.insert(0, _here)   # report/    — for environment.v3.kqk_env (new version)

import torch
import numpy as np

from report.agents.v6.ppo_agent import PPOAgent
from report.environment.v3.kqk_env import KQKEnv
import utils.board as b

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EVAL_EVERY = 10_000   # evaluate both metrics every N training episodes
N_EVAL     = 300      # episodes per greedy eval
MOVEMENT   = "centrum"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_full_history.json")

SCHEDULE = [
    (1.0, 500_000),
    (0.8, 500_000),
    (0.6, 500_000),
    (0.4, 500_000),
    (0.2, 500_000),
    (0.0, 500_000),
]

print(f"Device: {DEVICE}")
print(f"Schedule: 6 phases × 500k = 3M total episodes")
print(f"Output: {OUT}\n")


def eval_general(agent, n=N_EVAL):
    """Greedy eval on random KQK positions."""
    envs = [KQKEnv(curriculum_ratio=0.0, movement=MOVEMENT) for _ in range(n)]
    obs  = [e.reset()[0] for e in envs]
    done = [False] * n
    info = [{}] * n
    net  = agent.net
    net.eval()
    while not all(done):
        active = [i for i, d in enumerate(done) if not d]
        ob = torch.FloatTensor(np.stack([obs[i] for i in active])).to(DEVICE)
        with torch.no_grad(): logits, _ = net(ob)
        for j, i in enumerate(active):
            legal = [b.move_to_action(m) for m in envs[i].board.legal_moves]
            act   = legal[int(logits[j].cpu()[legal].argmax())]
            obs[i], _, term, trunc, inf = envs[i].step(act)
            if term or trunc:
                done[i] = True
                info[i] = inf
    net.train()
    return sum(1 for x in info if x.get("reason") == "checkmate") / n


def eval_mate_in_one(agent, n=N_EVAL):
    """Greedy eval: does the agent find the immediate checkmate move?"""
    mate_pool = agent.mate_pool
    boards = [chess.Board(random.choice(mate_pool)) for _ in range(n)]
    obs_t  = torch.FloatTensor(np.stack([b.board_to_obs(bd) for bd in boards])).to(DEVICE)
    net    = agent.net
    net.eval()
    with torch.no_grad(): logits_batch, _ = net(obs_t)
    correct = 0
    for i, board in enumerate(boards):
        legal  = [b.move_to_action(m) for m in board.legal_moves]
        action = legal[int(logits_batch[i].cpu()[legal].argmax())]
        board.push(b.action_to_move(action))
        if board.is_checkmate():
            correct += 1
    net.train()
    return correct / n


if __name__ == "__main__":
    agent = PPOAgent(curriculum_ratio=1.0, n_workers=31, episodes_per_worker=16)
    history = []
    ep_total = 0

    for ratio, n_phase_eps in SCHEDULE:
        agent.curriculum_ratio = ratio
        remaining = n_phase_eps
        print(f"\n── Phase  ratio={ratio:.1f}  ({n_phase_eps:,} episodes) ──")

        while remaining > 0:
            chunk = min(EVAL_EVERY, remaining)
            t0 = time.time()
            agent.train(n_episodes=chunk, movement=MOVEMENT, log_every=chunk + 1)
            elapsed = time.time() - t0

            ep_total += chunk
            remaining -= chunk

            gen = eval_general(agent)
            m1  = eval_mate_in_one(agent)
            history.append({
                "episode":        ep_total,
                "ratio":          ratio,
                "checkmate_rate": round(gen, 4),
                "mate_in_one":    round(m1, 4),
            })
            with open(OUT, "w") as f:
                json.dump(history, f)

            print(f"  ep {ep_total:>8,}  ratio={ratio:.1f}  "
                  f"checkmate={gen*100:.1f}%  mate-in-1={m1*100:.1f}%  ({elapsed:.0f}s)")

    print(f"\nDone. {OUT}")
