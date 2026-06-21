#!/usr/bin/env python3
"""PPO learning rate comparison: 3 runs × 100k episodes, mate-in-1 accuracy every 10k."""
import sys, os, json, time, random, chess
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_here))
sys.path.insert(0, _here)

import torch, numpy as np
from report.agents.v6.ppo_agent import PPOAgent
from report.environment.v3.kqk_env import KQKEnv
import utils.board as b

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPS     = 100_000
EVAL_EVERY = 10_000
N_EVAL    = 200
OUT = os.path.join(_here, "lr_comparison_ppo.json")
print(f"Device: {DEVICE}\n")

_env      = KQKEnv(curriculum_ratio=1.0)
MATE_POOL = _env.mate_pool

def eval_m1(net, n=N_EVAL):
    boards = [chess.Board(random.choice(MATE_POOL)) for _ in range(n)]
    obs_t  = torch.FloatTensor(np.stack([b.board_to_obs(bd) for bd in boards])).to(DEVICE)
    net.eval()
    with torch.no_grad(): logits_batch, _ = net(obs_t)
    correct = sum(
        1 for i, board in enumerate(boards)
        if (lambda legal, action: board.push(b.action_to_move(action)) or board.is_checkmate())(
            legal := [b.move_to_action(m) for m in board.legal_moves],
            legal[int(logits_batch[i].cpu()[legal].argmax())]
        )
    )
    net.train()
    return correct / n

def eval_m1_clean(net, n=N_EVAL):
    boards = [chess.Board(random.choice(MATE_POOL)) for _ in range(n)]
    obs_t  = torch.FloatTensor(np.stack([b.board_to_obs(bd) for bd in boards])).to(DEVICE)
    net.eval()
    with torch.no_grad(): logits_batch, _ = net(obs_t)
    net.train()
    correct = 0
    for i, board in enumerate(boards):
        legal  = [b.move_to_action(m) for m in board.legal_moves]
        action = legal[int(logits_batch[i].cpu()[legal].argmax())]
        board.push(b.action_to_move(action))
        if board.is_checkmate(): correct += 1
    return correct / n

if __name__ == "__main__":
    results = {}
    for lr in [1e-2, 1e-3, 1e-4]:
        key = str(lr)
        print(f"── lr={lr} ──")
        agent = PPOAgent(lr=lr, curriculum_ratio=1.0, n_workers=29, episodes_per_worker=16)
        curve = []
        remaining = N_EPS
        ep_done   = 0
        while remaining > 0:
            chunk = min(EVAL_EVERY, remaining)
            t0    = time.time()
            agent.train(n_episodes=chunk, movement="centrum", log_every=chunk+1)
            ep_done   += chunk
            remaining -= chunk
            acc = eval_m1_clean(agent.net)
            curve.append(round(acc, 4))
            print(f"  ep {ep_done:>7,}  mate-in-1={acc*100:.1f}%  ({time.time()-t0:.0f}s)")
        results[key] = curve
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved {OUT}")
