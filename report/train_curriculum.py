#!/usr/bin/env python3
"""
Section 6: Curriculum annealing experiment.
Train PPO at ratio=1.0 until mate-in-1 accuracy stabilises, then anneal ratio down.
Run from report/ directory:  python3 train_curriculum.py
Saves: curriculum_history.json  (list of {ratio, mate_acc})
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multiprocessing as mp
from agents.v6.ppo_agent import PPOAgent

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curriculum_history.json")


def main():
    # Episodes per phase — Phase 1 is longer to bootstrap from scratch
    schedule = [
        (1.0, 120_000),   # train until mate-in-1 is solid (~80-90%)
        (0.8,  80_000),
        (0.6,  80_000),
        (0.4,  80_000),
        (0.2,  80_000),
        (0.0,  80_000),
    ]

    print(f"Curriculum annealing: {sum(n for _, n in schedule):,} total episodes\n")

    agent   = PPOAgent(curriculum_ratio=1.0)
    history = []

    for ratio, n_eps in schedule:
        agent.curriculum_ratio = ratio
        print(f"--- ratio={ratio:.1f}  training {n_eps:,} episodes ---")
        t0 = time.time()
        agent.train(n_episodes=n_eps, movement="centrum", log_every=n_eps)
        elapsed = time.time() - t0

        acc = agent.evaluate_mate_in_one(n_episodes=500, greedy=False)
        history.append({"ratio": ratio, "mate_acc": round(acc * 100, 1)})
        print(f"  mate-in-1: {acc*100:.1f}%  ({elapsed:.0f}s)\n")

        with open(OUT, "w") as f:
            json.dump(history, f, indent=2)

    print(f"Done. Saved {OUT}")
    for h in history:
        print(f"  ratio={h['ratio']:.1f}  mate-in-1={h['mate_acc']:.1f}%")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
