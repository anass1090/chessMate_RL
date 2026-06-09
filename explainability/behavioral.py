"""
Behavioral trajectory analysis: run the greedy agent from N random
positions and record checkmate patterns.
Black moves are made by the trained opponent model (greedy) when one is
provided, otherwise randomly.

Statistics produced:
  - 8×8 heatmap of the square where the black king is mated
  - Game-length (plies) distribution
  - Mean move entropy across training stages — drops as play becomes
    more decisive and systematic

Usage
-----
    from explainability.behavioral import collect_trajectories, plot_mate_heatmap
    stats = collect_trajectories(net, n_episodes=2000)
    plot_mate_heatmap(stats["mate_squares"])
    plot_length_distribution(stats["lengths"])
"""

import os
import sys

import chess
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utils.board as b

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_STAGE_MODELS = [
    "kqk_ppo_v2.pt",
    "kqk_ppo_v2_stage_2.pt",
    "kqk_ppo_v2_stage_3.pt",
    "kqk_ppo_v2_stage_4.pt",
]

STAGE_LABELS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

MAX_PLIES = 200


def _model_path(model_name: str) -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "notebooks", "exploration", "models", model_name
    )


def _load_net(model_name: str) -> nn.Module:
    from agents.v9.ppo_agent import ActorCritic
    net = ActorCritic()
    net.load_state_dict(torch.load(_model_path(model_name), map_location=DEVICE))
    return net.to(DEVICE).eval()


def _load_opponent_net(model_name: str) -> nn.Module:
    from agents.v12.ppo_opponent import ActorCritic as OpponentActorCritic
    net = OpponentActorCritic()
    net.load_state_dict(torch.load(_model_path(model_name), map_location=DEVICE))
    return net.to(DEVICE).eval()


DEFAULT_OPPONENT_MODEL = "kqk_opponent_v2_stage_2.pt"


def _legal_entropy(net: nn.Module, board: chess.Board,
                   device: torch.device) -> tuple[chess.Move, float]:
    """Return (greedy best move, entropy over legal moves)."""
    legal   = list(board.legal_moves)
    actions = [b.move_to_action(m) for m in legal]
    obs_t   = torch.from_numpy(b.board_to_obs(board)).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = net(obs_t)

    legal_logits = torch.tensor([logits[0, a] for a in actions], dtype=torch.float32)
    probs        = torch.softmax(legal_logits, dim=0).numpy()
    entropy      = -float(np.sum(probs * np.log(probs + 1e-10)))
    best         = legal[int(legal_logits.argmax())]
    return best, entropy


def collect_trajectories(net: nn.Module,
                          n_episodes: int = 2000,
                          seed: int = 42,
                          device: torch.device = None,
                          opponent_net: nn.Module = None) -> dict:
    """
    Play n_episodes from random KQK positions.
    White moves greedily via `net`. Black moves greedily via `opponent_net`
    when provided, otherwise randomly.

    Returns
    -------
    {
      "mate_squares" : int array  — black-king square at each checkmate
      "lengths"      : int array  — game length in plies per episode
      "entropies"    : float array — mean white-move entropy per episode
      "outcomes"     : list[str]  — "checkmate" | "draw" per episode
      "mate_zones"   : list[str]  — "corner" | "edge" | "center" per mate
    }
    """
    if device is None:
        device = next(net.parameters()).device

    opp_device = next(opponent_net.parameters()).device if opponent_net is not None else device
    rng        = np.random.default_rng(seed)

    mate_squares: list[int]   = []
    lengths:      list[int]   = []
    entropies:    list[float] = []
    outcomes:     list[str]   = []
    mate_zones:   list[str]   = []

    for _ in range(n_episodes):
        board        = b.random_kqk_position()
        game_entropy = []
        ply          = 0

        while not board.is_game_over() and ply < MAX_PLIES:
            if board.turn == chess.WHITE:
                move, ent = _legal_entropy(net, board, device)
                game_entropy.append(ent)
            elif opponent_net is not None:
                move, _ = _legal_entropy(opponent_net, board, opp_device)
            else:
                legal = list(board.legal_moves)
                move  = legal[int(rng.integers(len(legal)))]

            board.push(move)
            ply += 1

        lengths.append(ply)
        entropies.append(float(np.mean(game_entropy)) if game_entropy else float("nan"))

        if board.is_checkmate():
            sq = board.king(chess.BLACK)
            r, f = divmod(sq, 8)
            is_corner = sq in (0, 7, 56, 63)
            is_edge   = r in (0, 7) or f in (0, 7)
            mate_squares.append(sq)
            mate_zones.append("corner" if is_corner else "edge" if is_edge else "center")
            outcomes.append("checkmate")
        else:
            outcomes.append("draw")

    return {
        "mate_squares": np.array(mate_squares, dtype=np.int32),
        "lengths":      np.array(lengths,      dtype=np.int32),
        "entropies":    np.array(entropies,    dtype=np.float32),
        "outcomes":     outcomes,
        "mate_zones":   mate_zones,
    }


def plot_mate_heatmap(mate_squares: np.ndarray,
                      ax: plt.Axes = None,
                      title: str = "Checkmate square heatmap") -> plt.Figure:
    """8×8 heatmap showing where the black king gets mated."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4.5, 4))
    else:
        fig = ax.get_figure()

    grid = np.zeros((8, 8), dtype=np.float32)
    for sq in mate_squares:
        r, f = divmod(int(sq), 8)
        grid[r, f] += 1.0
    grid /= max(grid.max(), 1.0)

    im = ax.imshow(np.flipud(grid), cmap="YlOrRd", vmin=0.0, vmax=1.0,
                   aspect="equal", interpolation="nearest")
    ax.set_xticks(range(8))
    ax.set_xticklabels(list("abcdefgh"), fontsize=8)
    ax.set_yticks(range(8))
    ax.set_yticklabels([str(r) for r in range(8, 0, -1)], fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if standalone:
        plt.tight_layout()
    return fig


def plot_length_distribution(lengths: np.ndarray,
                              ax: plt.Axes = None,
                              title: str = "Game length distribution") -> plt.Figure:
    """Histogram of game lengths in plies."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 3))
    else:
        fig = ax.get_figure()

    ax.hist(lengths, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(float(np.median(lengths)), color="orange", linestyle="--",
               linewidth=1.2, label=f"median = {np.median(lengths):.0f} plies")
    ax.set_xlabel("Game length (plies)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        plt.tight_layout()
    return fig


def plot_entropy_across_stages(model_names: list[str] = DEFAULT_STAGE_MODELS,
                                stage_labels: list[str] = STAGE_LABELS,
                                n_episodes: int = 500,
                                seed: int = 42,
                                opponent_net: nn.Module = None,
                                ax: plt.Axes = None) -> plt.Figure:
    """
    Bar chart of mean legal-move entropy per training stage.
    Lower entropy = more decisive, systematic play.
    Pass `opponent_net` to use the trained black agent instead of random moves.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.get_figure()

    mean_ents, std_ents = [], []
    for name in model_names:
        print(f"  collecting for {name}…")
        net   = _load_net(name)
        stats = collect_trajectories(net, n_episodes=n_episodes, seed=seed,
                                     opponent_net=opponent_net)
        ents  = stats["entropies"]
        valid = ents[~np.isnan(ents)]
        mean_ents.append(float(np.mean(valid)))
        std_ents.append(float(np.std(valid)))

    lbls = stage_labels[:len(model_names)]
    ax.bar(range(len(model_names)), mean_ents, yerr=std_ents,
           color="steelblue", capsize=5, width=0.5)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel("Mean entropy over legal moves")
    ax.set_title("Policy entropy across training stages\n(lower = more decisive)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="kqk_ppo_v2_stage_4.pt")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--entropy",  action="store_true",
                        help="Plot entropy across all default stages instead")
    args = parser.parse_args()

    if args.entropy:
        fig = plot_entropy_across_stages(n_episodes=args.episodes)
        plt.show()
    else:
        net   = _load_net(args.model)
        stats = collect_trajectories(net, n_episodes=args.episodes)

        n_mate = sum(o == "checkmate" for o in stats["outcomes"])
        print(f"Checkmate  : {n_mate}/{args.episodes} ({100*n_mate/args.episodes:.1f}%)")
        print(f"Median len : {np.median(stats['lengths']):.0f} plies")
        print(f"Mean H(π)  : {np.nanmean(stats['entropies']):.3f}")

        if stats["mate_zones"]:
            from collections import Counter
            for zone, cnt in Counter(stats["mate_zones"]).most_common():
                print(f"  {zone:<8s}: {cnt} ({100*cnt/n_mate:.1f}%)")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        plot_mate_heatmap(stats["mate_squares"], ax=axes[0])
        plot_length_distribution(stats["lengths"], ax=axes[1])
        plt.tight_layout()
        plt.show()