"""
Collect 256-dim trunk activations from the trained PolicyNet on random KQK positions.
Trunk = net[0:4] (Linear→ReLU→Linear→ReLU), hooked after the second ReLU.
Output: explainability/data/activations.npz
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utils.board as b
from agents.v7.reinforce_agent import PolicyNet

DEFAULT_MODEL_NAME = "kqk_reinforce_v7_stage_2.pt"
OUT_PATH = os.path.join(os.path.dirname(__file__), "data", "activations.npz")


def _model_path(model_name: str = DEFAULT_MODEL_NAME) -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "notebooks", "exploration", "models", model_name
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_POSITIONS = 5000


def _load_net(model_name: str = DEFAULT_MODEL_NAME) -> PolicyNet:
    net = PolicyNet()
    net.load_state_dict(torch.load(_model_path(model_name), map_location=DEVICE))
    net.to(DEVICE)
    net.eval()
    return net


def collect(n_positions: int = N_POSITIONS, seed: int = 42, model_name: str = DEFAULT_MODEL_NAME) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = _load_net(model_name)
    trunk: nn.Sequential = net.net[0:4]  # Linear→ReLU→Linear→ReLU

    boards = [b.random_kqk_position() for _ in range(n_positions)]
    obs_np = np.stack([b.board_to_obs(board) for board in boards])  # (N, 768)

    obs_t = torch.from_numpy(obs_np).to(DEVICE)

    with torch.no_grad():
        activations = trunk(obs_t).cpu().numpy()  # (N, 256)
        logits_all = net.net(obs_t).cpu().numpy()  # (N, 4096)

    fens = [board.fen() for board in boards]

    # Chosen action = argmax over all 4096 logits (no legality mask here — for analysis only)
    chosen_actions = logits_all.argmax(axis=1)
    chosen_probs = torch.softmax(torch.from_numpy(logits_all), dim=-1).numpy()
    chosen_prob_vals = chosen_probs[np.arange(n_positions), chosen_actions]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        activations=activations.astype(np.float32),
        fens=np.array(fens),
        chosen_actions=chosen_actions.astype(np.int32),
        chosen_probs=chosen_prob_vals.astype(np.float32),
    )
    print(f"Saved {n_positions} activations → {OUT_PATH}")
    return dict(activations=activations, fens=fens,
                chosen_actions=chosen_actions, chosen_probs=chosen_prob_vals)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model filename inside notebooks/exploration/models/")
    args = parser.parse_args()
    collect(model_name=args.model)
