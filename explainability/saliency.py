"""
Perturbation-based input saliency for a given board position and move.

For each of the 64 squares, zero out its 12 piece-feature channels
(6 piece types × 2 colours) in the 768-dim input, re-run the policy,
and measure the drop in probability for the chosen action.

Returns an 8×8 importance map: high value = that square most influenced
the decision. This proves whether the agent focuses on, e.g., the black
king's escape squares when playing a cutting move with the queen.

Usage
-----
    from explainability.saliency import square_saliency, plot_saliency
    smap = square_saliency(fen, move_uci, net)
    fig  = plot_saliency(fen, move_uci, smap)
"""

import io
import os
import sys

import chess
import chess.svg
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utils.board as b

# The 768-dim obs encodes square sq as:
#   channel = piece_type_idx * 128 + colour_idx * 64 + sq
# → 12 channels per square (6 types × 2 colours)
def _square_channels(sq: int) -> list[int]:
    return [pt * 128 + c * 64 + sq for pt in range(6) for c in range(2)]


def square_saliency(fen: str,
                    move_uci: str,
                    net: nn.Module,
                    device: torch.device = None) -> np.ndarray:
    """
    Compute 8×8 saliency map for `move_uci` in position `fen`.

    Each cell = drop in policy probability for the move when that
    square's 12 input channels are zeroed out.

    Returns
    -------
    saliency : np.ndarray shape (8, 8), float32  (a1 = [0,0])
    """
    if device is None:
        device = next(net.parameters()).device

    board  = chess.Board(fen)
    action = b.move_to_action(chess.Move.from_uci(move_uci))

    obs    = b.board_to_obs(board)                                  # (768,)
    obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits_base, _ = net(obs_t)
        p_base = float(torch.softmax(logits_base, dim=-1)[0, action].cpu())

    # Build 64 perturbed observations in one batch
    obs_batch = np.tile(obs, (64, 1))                               # (64, 768)
    for sq in range(64):
        for ch in _square_channels(sq):
            obs_batch[sq, ch] = 0.0

    obs_bt = torch.from_numpy(obs_batch).float().to(device)
    with torch.no_grad():
        logits_all, _ = net(obs_bt)                                 # (64, 4096)
        probs_all = torch.softmax(logits_all, dim=-1).cpu().numpy()

    saliency = np.array(
        [max(0.0, p_base - float(probs_all[sq, action])) for sq in range(64)],
        dtype=np.float32,
    )
    return saliency.reshape(8, 8)


def plot_saliency(fen: str,
                  move_uci: str,
                  saliency: np.ndarray,
                  ax_board: plt.Axes = None,
                  ax_heat: plt.Axes = None) -> plt.Figure:
    """Board diagram (with move arrow) next to 8×8 saliency heatmap."""
    try:
        import cairosvg
        _has_cairo = True
    except ImportError:
        _has_cairo = False

    standalone = ax_board is None
    if standalone:
        fig, (ax_board, ax_heat) = plt.subplots(1, 2, figsize=(10, 4.5))
    else:
        fig = ax_board.get_figure()

    # --- board diagram ---
    board = chess.Board(fen)
    move  = chess.Move.from_uci(move_uci)
    svg_str = chess.svg.board(
        board,
        arrows=[chess.svg.Arrow(move.from_square, move.to_square, color="#cc0000")],
        size=300,
    )
    if _has_cairo:
        png_bytes = cairosvg.svg2png(bytestring=svg_str.encode())
        ax_board.imshow(plt.imread(io.BytesIO(png_bytes)))
    else:
        ax_board.text(0.5, 0.5, f"{fen}\n→ {move_uci}",
                      ha="center", va="center",
                      transform=ax_board.transAxes, fontsize=7)
    ax_board.axis("off")
    ax_board.set_title(f"Move: {move_uci}", fontsize=10)

    # --- saliency heatmap ---
    vmax = max(float(saliency.max()), 1e-6)
    im = ax_heat.imshow(np.flipud(saliency), cmap="hot", vmin=0.0, vmax=vmax,
                        aspect="equal", interpolation="nearest")
    ax_heat.set_xticks(range(8))
    ax_heat.set_xticklabels(list("abcdefgh"), fontsize=8)
    ax_heat.set_yticks(range(8))
    ax_heat.set_yticklabels([str(r) for r in range(8, 0, -1)], fontsize=8)
    ax_heat.set_title("Square saliency\n(prob drop on occlusion)", fontsize=9)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    if standalone:
        plt.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    from agents.v9.ppo_agent import ActorCritic

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--fen",   default="3k4/8/8/8/4Q3/8/8/4K3 w - - 0 1")
    parser.add_argument("--move",  default=None,
                        help="UCI move. Omit to use the greedy best move.")
    parser.add_argument("--model", default="kqk_ppo_v2_stage_4.pt")
    args = parser.parse_args()

    model_path = os.path.join(
        os.path.dirname(__file__), "..", "notebooks", "exploration", "models", args.model
    )
    net = ActorCritic()
    net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    net.to(DEVICE).eval()

    board = chess.Board(args.fen)

    if args.move is None:
        obs_t  = torch.from_numpy(b.board_to_obs(board)).float().unsqueeze(0).to(DEVICE)
        legal  = list(board.legal_moves)
        acts   = [b.move_to_action(m) for m in legal]
        with torch.no_grad():
            logits, _ = net(obs_t)
        best = legal[int(torch.tensor([logits[0, a] for a in acts]).argmax())]
        args.move = best.uci()
        print(f"Greedy move: {args.move}")

    smap = square_saliency(args.fen, args.move, net, DEVICE)
    fig  = plot_saliency(args.fen, args.move, smap)
    plt.show()