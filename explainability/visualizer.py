"""
Visualization utilities for the concept discovery pipeline.

plot_umap(coords, labels)                        — scatter plot colored by cluster
concept_heatmap(fens)                            — 8×8 piece-presence heatmap
build_concept_labels(ae_model, activations_path) — auto-label concepts via feature correlation
build_concept_labels_tree(ae_model, ...)         — nonlinear decision-tree labels per concept
explain_move(fen, move_uci, ae, trunk, ...)      — board + active-concept bar chart
"""

import os
import sys

import chess
import chess.svg
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utils.board as b


# ---------------------------------------------------------------------------
# 1. UMAP scatter
# ---------------------------------------------------------------------------

def plot_umap(umap_coords: np.ndarray, cluster_labels: np.ndarray,
              ax: plt.Axes = None, title: str = "UMAP of trunk activations") -> plt.Figure:
    """Scatter plot of 2-D UMAP coordinates coloured by HDBSCAN cluster."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 7))
    else:
        fig = ax.get_figure()

    unique = sorted(set(cluster_labels.tolist()))
    cmap   = plt.get_cmap("tab20", max(len(unique), 1))

    n_real = max(1, sum(1 for l in unique if l != -1))
    cmap   = plt.get_cmap("tab20" if n_real <= 20 else "turbo", n_real)

    noise_mask = cluster_labels == -1
    if noise_mask.any():
        ax.scatter(umap_coords[noise_mask, 0], umap_coords[noise_mask, 1],
                   c="lightgrey", s=3, alpha=0.3, rasterized=True, label="noise")

    real_labels = [l for l in unique if l != -1]
    for i, label in enumerate(real_labels):
        mask = cluster_labels == label
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=[cmap(i)], s=4, alpha=0.6, rasterized=True)
        cx = umap_coords[mask, 0].mean()
        cy = umap_coords[mask, 1].mean()
        ax.text(cx, cy, str(label), fontsize=7, ha="center", va="center",
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc=cmap(i), ec="none", alpha=0.8))

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if noise_mask.any():
        ax.legend(markerscale=3, loc="best", fontsize=7)
    if standalone:
        plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Concept heatmap
# ---------------------------------------------------------------------------

def _fen_to_presence(fens: list[str]) -> np.ndarray:
    """Return (8, 8) float array: mean piece presence across positions."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for fen in fens:
        board = chess.Board(fen)
        for sq, _ in board.piece_map().items():
            r, c = divmod(sq, 8)
            grid[r, c] += 1.0
    return grid / max(len(fens), 1)


def concept_heatmap(top_fen_strings: list[str],
                    ax: plt.Axes = None,
                    title: str = "Concept heatmap") -> plt.Figure:
    """Average piece-presence heatmap for a list of FEN positions."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()

    grid = _fen_to_presence(top_fen_strings)
    grid_display = np.flipud(grid)

    im = ax.imshow(grid_display, cmap="YlOrRd", vmin=0.0, vmax=1.0,
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


# ---------------------------------------------------------------------------
# 3. Feature definitions
# ---------------------------------------------------------------------------

_ATOMIC_FEATURE_NAMES = [
    # --- distances (Chebyshev) ---
    "Q→BK dist",           # 0
    "WK→BK dist",          # 1
    "Q→BK rank gap",       # 2
    "Q→BK file gap",       # 3
    "WK→BK rank gap",      # 4
    "WK→BK file gap",      # 5
    # --- BK edge membership ---
    "BK on rank 1",        # 6
    "BK on rank 8",        # 7
    "BK on file a",        # 8
    "BK on file h",        # 9
    "BK on edge",          # 10
    "BK in corner",        # 11
    "BK edge dist",        # 12
    # --- BK half-board ---
    "BK in top half",      # 13
    "BK in right half",    # 14
    # --- mobility / control ---
    "Q mobility",          # 15
    "BK mobility",         # 16
    "BK mobility=1",       # 17
    "BK mobility=0",       # 18
    # --- centrality ---
    "BK centrality",       # 19
    "WK centrality",       # 20
    # --- alignment ---
    "Q-BK same rank",      # 21
    "Q-BK same file",      # 22
    "Q-BK same diagonal",  # 23
    "WK closer than Q",    # 24
    "WK-Q-BK collinear",   # 25
    # --- absolute position (continuous) ---
    "BK rank",             # 26  0=rank1 … 7=rank8
    "BK file",             # 27  0=file a … 7=file h
    "Q rank",              # 28
    "Q file",              # 29
    # --- signed king offsets ---
    "BK-WK rank signed",   # 30  bk_r - wk_r
    "BK-WK file signed",   # 31  bk_f - wk_f
    # --- piece edge membership ---
    "Q on edge",           # 32
    "WK on edge",          # 33
    # --- confinement ---
    "BK corner zone",      # 34  min(r,7-r,f,7-f) <= 1
    "Q→BK adj",            # 35  Chebyshev dist == 1
    "Q cuts off BK",       # 36  q on adjacent rank or file (barrier)
]

_CONJUNCTION_FEATURE_NAMES = [
    "BK edge & Q near",    # 37  bk_edge * (q_bk_dist <= 2)
    "BK edge & mob≤1",     # 38  bk_edge * bk_mob1
    "BK corner & Q near",  # 39  bk_corner * (q_bk_dist <= 2)
    "BK mob=1 & WK close", # 40  bk_mob1 * (wk_bk_dist <= 2)
    "Q aligned & near",    # 41  (same_rank | same_file) * (q_bk_dist <= 3)
    "BK rank1 & Q→rank",   # 42  bk_on_rank1 * same_rank
    "BK filea & Q→file",   # 43  bk_on_filea * same_file
]

_FEATURE_NAMES = _ATOMIC_FEATURE_NAMES + _CONJUNCTION_FEATURE_NAMES

FEATURE_NAMES = _FEATURE_NAMES  # public alias

BINARY_FEATURES = {
    "BK on rank 1", "BK on rank 8", "BK on file a", "BK on file h",
    "BK on edge", "BK in corner", "BK in top half", "BK in right half",
    "BK mobility=1", "BK mobility=0",
    "Q-BK same rank", "Q-BK same file", "Q-BK same diagonal",
    "WK closer than Q",
    "Q on edge", "WK on edge", "BK corner zone", "Q→BK adj",
    "BK edge & Q near", "BK edge & mob≤1", "BK corner & Q near",
    "BK mob=1 & WK close", "Q aligned & near",
    "BK rank1 & Q→rank", "BK filea & Q→file",
    "Q cuts off BK",
}


def _chebyshev(sq1: int, sq2: int) -> int:
    r1, f1 = divmod(sq1, 8)
    r2, f2 = divmod(sq2, 8)
    return max(abs(r1 - r2), abs(f1 - f2))


def _compute_board_features(fens: list[str]) -> np.ndarray:
    """Return (N, 43) float32 array of hand-crafted + conjunction board features."""
    rows = []
    for fen in fens:
        board = chess.Board(fen)
        bk_sq = board.king(chess.BLACK)
        wk_sq = board.king(chess.WHITE)
        q_bb  = board.pieces(chess.QUEEN, chess.WHITE)
        q_sq  = next(iter(q_bb)) if q_bb else None

        bk_r, bk_f = divmod(bk_sq, 8)
        wk_r, wk_f = divmod(wk_sq, 8)

        q_bk_dist  = _chebyshev(q_sq, bk_sq) if q_sq is not None else 7
        wk_bk_dist = _chebyshev(wk_sq, bk_sq)

        if q_sq is not None:
            q_r, q_f  = divmod(q_sq, 8)
            q_bk_rank = float(abs(q_r - bk_r))
            q_bk_file = float(abs(q_f - bk_f))
        else:
            q_r = q_f = 0
            q_bk_rank = q_bk_file = 7.0

        wk_bk_rank = float(abs(wk_r - bk_r))
        wk_bk_file = float(abs(wk_f - bk_f))

        bk_on_rank1 = float(bk_r == 0)
        bk_on_rank8 = float(bk_r == 7)
        bk_on_filea = float(bk_f == 0)
        bk_on_fileh = float(bk_f == 7)
        bk_edge     = float(bk_r in (0, 7) or bk_f in (0, 7))
        bk_corner   = float(bk_sq in (0, 7, 56, 63))
        bk_edge_dist = float(min(bk_r, 7 - bk_r, bk_f, 7 - bk_f))

        q_mobility = float(len(board.attacks(q_sq))) if q_sq is not None else 0.0

        board_copy = board.copy()
        board_copy.turn = chess.BLACK
        bk_mob = sum(1 for _ in board_copy.legal_moves)
        bk_mobility = float(bk_mob)
        bk_mob1     = float(bk_mob == 1)
        bk_mob0     = float(bk_mob == 0)

        bk_centrality = float(3.5 - max(abs(bk_r - 3.5), abs(bk_f - 3.5)))
        wk_centrality = float(3.5 - max(abs(wk_r - 3.5), abs(wk_f - 3.5)))

        if q_sq is not None:
            same_rank = float(q_r == bk_r)
            same_file = float(q_f == bk_f)
            same_diag = float(abs(q_r - bk_r) == abs(q_f - bk_f))
            wk_closer = float(wk_bk_dist < q_bk_dist)
            area      = abs(wk_r*(q_f-bk_f) + q_r*(bk_f-wk_f) + bk_r*(wk_f-q_f)) / 2.0
            collinear = float(max(0.0, 1.0 - area / 16.0))
        else:
            same_rank = same_file = same_diag = wk_closer = collinear = 0.0

        # --- new atomic features ---
        q_on_edge    = float(q_r in (0, 7) or q_f in (0, 7)) if q_sq is not None else 0.0
        wk_on_edge   = float(wk_r in (0, 7) or wk_f in (0, 7))
        bk_corn_zone = float(min(bk_r, 7 - bk_r, bk_f, 7 - bk_f) <= 1)
        q_bk_adj     = float(q_bk_dist == 1) if q_sq is not None else 0.0
        q_cuts_off   = float(abs(q_r - bk_r) == 1 or abs(q_f - bk_f) == 1) if q_sq is not None else 0.0

        # --- conjunction features ---
        conj_edge_q_near    = bk_edge * float(q_bk_dist <= 2)
        conj_edge_mob1      = bk_edge * bk_mob1
        conj_corner_q_near  = bk_corner * float(q_bk_dist <= 2)
        conj_mob1_wk_close  = bk_mob1 * float(wk_bk_dist <= 2)
        conj_aligned_near   = float(same_rank or same_file) * float(q_bk_dist <= 3)
        conj_rank1_qrank    = bk_on_rank1 * same_rank
        conj_filea_qfile    = bk_on_filea * same_file

        rows.append([
            # atomic (26 original)
            float(q_bk_dist), float(wk_bk_dist),
            q_bk_rank, q_bk_file, wk_bk_rank, wk_bk_file,
            bk_on_rank1, bk_on_rank8, bk_on_filea, bk_on_fileh,
            bk_edge, bk_corner, bk_edge_dist,
            float(bk_r >= 4), float(bk_f >= 4),
            q_mobility, bk_mobility, bk_mob1, bk_mob0,
            bk_centrality, wk_centrality,
            same_rank, same_file, same_diag,
            wk_closer, collinear,
            # new atomic (10)
            float(bk_r), float(bk_f),
            float(q_r) if q_sq is not None else 0.0,
            float(q_f) if q_sq is not None else 0.0,
            float(bk_r - wk_r), float(bk_f - wk_f),
            q_on_edge, wk_on_edge, bk_corn_zone, q_bk_adj, q_cuts_off,
            # conjunction (7)
            conj_edge_q_near, conj_edge_mob1, conj_corner_q_near,
            conj_mob1_wk_close, conj_aligned_near,
            conj_rank1_qrank, conj_filea_qfile,
        ])
    return np.array(rows, dtype=np.float32)


def compute_board_features(fens: list[str]) -> np.ndarray:
    """Public wrapper around _compute_board_features."""
    return _compute_board_features(fens)


# ---------------------------------------------------------------------------
# 4. Concept auto-labelling — linear (correlation + composite)
# ---------------------------------------------------------------------------

def build_concept_labels(ae_model: nn.Module,
                         activations_path: str = None,
                         top_k_features: int = 3,
                         min_r_secondary: float = 0.15,
                         ) -> tuple[dict[int, str], np.ndarray]:
    """
    Correlate each AE encoder dimension with hand-crafted board features
    (including conjunction features) and build composite labels from the
    top-K correlated features.

    Returns
    -------
    labels : dict {concept_id: label_string}
    table  : structured array (concept_id i4, r f4, feature U120, direction U2)
             sorted by |r| descending
    """
    if activations_path is None:
        activations_path = os.path.join(os.path.dirname(__file__), "data", "activations.npz")

    data        = np.load(activations_path, allow_pickle=True)
    activations = torch.from_numpy(data["activations"]).float()
    fens        = list(data["fens"])

    ae_device = next(ae_model.parameters()).device
    with torch.no_grad():
        _, encoded = ae_model(activations.to(ae_device))
        encoded_np = encoded.cpu().numpy()  # (N, 512)

    features = _compute_board_features(fens)  # (N, 43)

    def _zscore(x):
        std = x.std(axis=0)
        std[std < 1e-8] = 1.0
        return (x - x.mean(axis=0)) / std

    corr = (_zscore(encoded_np).T @ _zscore(features)) / len(fens)  # (512, 43)

    alive = encoded_np.max(axis=0) > 0.01

    labels: dict[int, str] = {}
    rows = []
    for c in range(corr.shape[0]):
        if not alive[c]:
            continue

        order     = np.argsort(np.abs(corr[c]))[::-1]
        best_feat = int(order[0])
        r_val     = float(corr[c, best_feat])
        sign      = "↑" if r_val > 0 else "↓"

        parts = [f"{_FEATURE_NAMES[best_feat]} {sign}"]
        for idx in order[1:top_k_features]:
            r_sec = float(corr[c, idx])
            if abs(r_sec) < min_r_secondary:
                break
            parts.append(f"{_FEATURE_NAMES[idx]} {'↑' if r_sec > 0 else '↓'}")

        composite = " | ".join(parts)
        labels[c] = f"{composite}  (r={r_val:.2f})"
        rows.append((c, abs(r_val), composite, sign))

    dtype = np.dtype([("concept_id", "i4"), ("r", "f4"),
                      ("feature", "U120"), ("direction", "U2")])
    table = np.array(rows, dtype=dtype)
    table.sort(order="r")
    table = table[::-1]

    print(f"{len(table)}/{encoded_np.shape[1]} alive concepts")
    return labels, table


# ---------------------------------------------------------------------------
# 5. Concept auto-labelling — nonlinear (decision tree per concept)
# ---------------------------------------------------------------------------

def _tree_to_rule(tree, feature_names: list[str],
                  binary_features: set[str], max_conditions: int = 3) -> str:
    """
    Extract the path to the highest-mean-activation leaf as a human-readable rule.
    Follows the branch with higher leaf mean at each split.
    """
    from sklearn.tree import _tree as _sk_tree

    t = tree.tree_

    def best_leaf(node: int) -> tuple[float, list[str]]:
        if t.feature[node] == _sk_tree.TREE_UNDEFINED:
            return float(t.value[node][0][0]), []

        fname  = feature_names[t.feature[node]]
        thresh = t.threshold[node]

        lv, lpath = best_leaf(t.children_left[node])
        rv, rpath = best_leaf(t.children_right[node])

        if rv >= lv:
            # right branch: feature > threshold  →  binary means "True"
            cond = fname if fname in binary_features else f"{fname} > {thresh:.1f}"
            return rv, [cond] + rpath
        else:
            # left branch: feature ≤ threshold  →  binary means "False"
            cond = f"¬{fname}" if fname in binary_features else f"{fname} ≤ {thresh:.1f}"
            return lv, [cond] + lpath

    _, conditions = best_leaf(0)
    conditions = conditions[:max_conditions]
    return " ∧ ".join(conditions) if conditions else "always"


def build_concept_labels_tree(ae_model: nn.Module,
                               activations_path: str = None,
                               max_depth: int = 3,
                               ) -> tuple[dict[int, str], np.ndarray]:
    """
    Fit a shallow decision tree per alive concept to discover what combination
    of board features predicts its activation.  Returns composite rule labels
    like ``"BK on edge ∧ Q→BK adj ∧ BK mob=1 & WK close"``.

    Returns
    -------
    labels : dict {concept_id: rule_string}
    table  : structured array (concept_id i4, r2 f4, feature U200, direction U2)
             sorted by R² descending  (r2 field = tree's R² on training data)
    """
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score

    if activations_path is None:
        activations_path = os.path.join(os.path.dirname(__file__), "data", "activations.npz")

    data        = np.load(activations_path, allow_pickle=True)
    activations = torch.from_numpy(data["activations"]).float()
    fens        = list(data["fens"])

    ae_device = next(ae_model.parameters()).device
    with torch.no_grad():
        _, encoded = ae_model(activations.to(ae_device))
        encoded_np = encoded.cpu().numpy()  # (N, 512)

    features = _compute_board_features(fens)  # (N, 43)

    alive = encoded_np.max(axis=0) > 0.01

    labels: dict[int, str] = {}
    rows = []
    for c in range(encoded_np.shape[1]):
        if not alive[c]:
            continue

        y    = encoded_np[:, c]
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        tree.fit(features, y)

        r2   = float(r2_score(y, tree.predict(features)))
        rule = _tree_to_rule(tree, _FEATURE_NAMES, BINARY_FEATURES)

        labels[c] = f"{rule}  (R²={r2:.2f})"
        rows.append((c, max(r2, 0.0), rule, ""))

    dtype = np.dtype([("concept_id", "i4"), ("r2", "f4"),
                      ("feature", "U200"), ("direction", "U2")])
    table = np.array(rows, dtype=dtype)
    table.sort(order="r2")
    table = table[::-1]

    print(f"{len(table)}/{encoded_np.shape[1]} alive concepts (tree, max_depth={max_depth})")
    return labels, table


# ---------------------------------------------------------------------------
# 6. Move explanation
# ---------------------------------------------------------------------------

def explain_move(fen: str, move_uci: str,
                 ae_model: nn.Module, trunk: nn.Sequential,
                 concept_labels: dict[int, str],
                 top_k: int = 5) -> plt.Figure:
    """
    Board diagram + horizontal bar chart of the top-k active concepts.
    Works with labels from either build_concept_labels or build_concept_labels_tree.
    """
    import io
    import cairosvg

    trunk_device = next(trunk.parameters()).device
    ae_device    = next(ae_model.parameters()).device

    board = chess.Board(fen)
    obs   = torch.from_numpy(b.board_to_obs(board)).float().unsqueeze(0).to(trunk_device)

    with torch.no_grad():
        activation = trunk(obs).squeeze(0)
        _, encoded = ae_model(activation.unsqueeze(0).to(ae_device))
        encoded    = encoded.squeeze(0).cpu().numpy()

    active_mask = encoded > 0.01
    if active_mask.sum() == 0:
        active_mask = encoded > 0
    active_idx = np.where(active_mask)[0]
    top_idx    = active_idx[np.argsort(encoded[active_idx])[-top_k:][::-1]]
    strengths  = encoded[top_idx]
    bar_labels = [concept_labels.get(int(i), f"#{i}") for i in top_idx]

    move    = chess.Move.from_uci(move_uci)
    svg_str = chess.svg.board(
        board,
        arrows=[chess.svg.Arrow(move.from_square, move.to_square, color="#cc0000")],
        size=300,
    )
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode())
    board_img = plt.imread(io.BytesIO(png_bytes))

    fig, (ax_board, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 4.5),
        gridspec_kw={"width_ratios": [1, 2]},
    )

    ax_board.imshow(board_img)
    ax_board.axis("off")
    ax_board.set_title(f"Move: {move_uci}", fontsize=10, pad=6)

    colors = plt.get_cmap("Blues")(
        0.4 + 0.5 * (strengths / (strengths.max() + 1e-8))
    )
    bars = ax_bar.barh(range(top_k), strengths[::-1], color=colors[::-1])
    ax_bar.set_yticks(range(top_k))
    ax_bar.set_yticklabels(bar_labels[::-1], fontsize=8)
    ax_bar.set_xlabel("Concept activation strength", fontsize=9)
    ax_bar.set_title("Top active concepts", fontsize=10)
    ax_bar.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, strengths[::-1]):
        ax_bar.text(val + strengths.max() * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=8)

    fig.suptitle(fen, fontsize=7, y=0.02, color="grey")
    plt.tight_layout()
    return fig