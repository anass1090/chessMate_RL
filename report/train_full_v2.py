#!/usr/bin/env python3
"""
Full PPO curriculum training v2 — one continuous call per phase (no mid-phase restarts).
6 phases × 500k + extended Phase 6 until convergence.
Saves model after each phase. Overwrites ppo_full_history.json.
"""
import sys, os, json, time, random, chess
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
sys.path.insert(0, _here)

import torch
import numpy as np

from report.agents.v6.ppo_agent import PPOAgent, ActorCritic, _sample_action
from report.environment.v3.kqk_env import KQKEnv
import utils.board as b

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EVAL    = 300
MOVEMENT  = "centrum"
HIST_OUT  = os.path.join(_here, "ppo_full_history.json")
MODEL_DIR = os.path.join(_here, "phase_checkpoints")
os.makedirs(MODEL_DIR, exist_ok=True)

SCHEDULE = [
    (1.0, 500_000),
    (0.8, 500_000),
    (0.6, 500_000),
    (0.4, 500_000),
    (0.2, 500_000),
    (0.0, 1_000_000),   # extended: needs more time to converge
]

print(f"Device: {DEVICE}")
total = sum(n for _, n in SCHEDULE)
print(f"Schedule: {[(r, n//1000) for r,n in SCHEDULE]} (k eps)")
print(f"Total: {total/1_000_000:.1f}M episodes\n")


def eval_general(net, n=N_EVAL):
    envs = [KQKEnv(curriculum_ratio=0.0, movement=MOVEMENT) for _ in range(n)]
    obs  = [e.reset()[0] for e in envs]
    done = [False]*n; info = [{}]*n
    net.eval()
    while not all(done):
        active = [i for i,d in enumerate(done) if not d]
        ob = torch.FloatTensor(np.stack([obs[i] for i in active])).to(DEVICE)
        with torch.no_grad(): logits, _ = net(ob)
        for j,i in enumerate(active):
            legal = [b.move_to_action(m) for m in envs[i].board.legal_moves]
            act   = legal[int(logits[j].cpu()[legal].argmax())]
            obs[i], _, term, trunc, inf = envs[i].step(act)
            if term or trunc: done[i]=True; info[i]=inf
    net.train()
    return sum(1 for x in info if x.get("reason")=="checkmate") / n


def eval_mate_in_one(net, mate_pool, n=N_EVAL):
    boards = [chess.Board(random.choice(mate_pool)) for _ in range(n)]
    obs_t  = torch.FloatTensor(np.stack([b.board_to_obs(bd) for bd in boards])).to(DEVICE)
    net.eval()
    with torch.no_grad(): logits_batch, _ = net(obs_t)
    correct = 0
    for i, board in enumerate(boards):
        legal  = [b.move_to_action(m) for m in board.legal_moves]
        action = legal[int(logits_batch[i].cpu()[legal].argmax())]
        board.push(b.action_to_move(action))
        if board.is_checkmate(): correct += 1
    net.train()
    return correct / n


if __name__ == "__main__":
    agent     = PPOAgent(curriculum_ratio=1.0, n_workers=31, episodes_per_worker=16)
    mate_pool = agent.mate_pool
    history   = []
    ep_total  = 0
    EVAL_EVERY = 25_000   # evaluate every 25k eps to balance logging vs speed

    for phase_idx, (ratio, n_phase_eps) in enumerate(SCHEDULE, 1):
        agent.curriculum_ratio = ratio
        print(f"\n── Phase {phase_idx}  ratio={ratio:.1f}  ({n_phase_eps:,} eps) ──")

        # Train the full phase in chunks of EVAL_EVERY
        remaining = n_phase_eps
        while remaining > 0:
            chunk = min(EVAL_EVERY, remaining)
            t0 = time.time()
            agent.train(n_episodes=chunk, movement=MOVEMENT, log_every=chunk+1)
            elapsed = time.time() - t0

            ep_total  += chunk
            remaining -= chunk

            gen = eval_general(agent.net)
            m1  = eval_mate_in_one(agent.net, mate_pool)
            history.append({
                "episode":        ep_total,
                "ratio":          ratio,
                "checkmate_rate": round(gen, 4),
                "mate_in_one":    round(m1, 4),
            })
            with open(HIST_OUT, "w") as f:
                json.dump(history, f)
            print(f"  ep {ep_total:>8,}  ratio={ratio:.1f}  "
                  f"checkmate={gen*100:.1f}%  mate-in-1={m1*100:.1f}%  ({elapsed:.0f}s)")

            # Early stop if converged (Phase 6 only)
            if ratio == 0.0 and gen >= 0.95:
                print(f"  Converged at {gen*100:.1f}% — stopping Phase 6 early.")
                remaining = 0

        # Save model after each phase
        ckpt = os.path.join(MODEL_DIR, f"phase{phase_idx}_ratio{ratio:.1f}.pt")
        torch.save(agent.net.state_dict(), ckpt)
        print(f"  Saved: {ckpt}")

    print(f"\nDone. History saved to {HIST_OUT}")
