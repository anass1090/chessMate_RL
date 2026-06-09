"""
"What–When–Where" analysis.

  WHAT  — a chess concept  (e.g. "BK on edge", "BK mob=1 & WK close")
  WHEN  — training stage   (checkpoint name)
  WHERE — layer depth      (early = after 1st ReLU, late = after 2nd ReLU)

For each (stage × layer × concept) triple a linear probe AUC is computed.
The result is two heatmaps side-by-side — one per layer — with rows = stages
and columns = concepts.

Usage
-----
    from explainability.what_when_where import run, plot
    results = run()
    plot(results)
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utils.board as b
from explainability.visualizer import compute_board_features, FEATURE_NAMES, BINARY_FEATURES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_STAGE_MODELS = [
    "kqk_ppo_v2.pt",
    "kqk_ppo_v2_stage_2.pt",
    "kqk_ppo_v2_stage_3.pt",
    "kqk_ppo_v2_stage_4.pt",
]

STAGE_LABELS = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

DEFAULT_CONCEPTS = [
    # Black king position / confinement
    "BK on edge",
    "BK in corner",
    "BK corner zone",
    "BK mobility=1",
    "BK mobility=0",
    "BK edge & mob≤1",
    "BK mob=1 & WK close",
    "BK corner & Q near",
    # Queen alignment with black king
    "Q-BK same rank",
    "Q-BK same file",
    "Q-BK same diagonal",
    "Q→BK adj",
    "Q aligned & near",
    "Q cuts off BK",
    # Queen quality / position
    "Q mobility",
    "Q on edge",
    # King-queen coordination
    "WK closer than Q",
    "WK-Q-BK collinear",
    # Specific confinement patterns
    "BK rank1 & Q→rank",
    "BK filea & Q→file",
]

N_POSITIONS = 5_000


def _model_path(model_name: str) -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "notebooks", "exploration", "models", model_name
    )


def _load_net(model_name: str):
    from agents.v9.ppo_agent import ActorCritic
    net = ActorCritic()
    net.load_state_dict(torch.load(_model_path(model_name), map_location=DEVICE))
    return net.to(DEVICE).eval()


def _collect_layer_acts(net, obs_t: torch.Tensor,
                        chunk: int = 1024) -> dict[str, np.ndarray]:
    """
    Return {"early": (N,256), "late": (N,256)} by hooking:
      backbone[1] — after the 1st ReLU  (early)
      backbone[3] — after the 2nd ReLU  (late / trunk output)
    """
    early_bufs, late_bufs = [], []

    h0 = net.backbone[1].register_forward_hook(
        lambda m, i, o: early_bufs.append(o.detach().cpu())
    )
    h1 = net.backbone[3].register_forward_hook(
        lambda m, i, o: late_bufs.append(o.detach().cpu())
    )

    with torch.no_grad():
        for start in range(0, len(obs_t), chunk):
            net(obs_t[start:start + chunk].to(DEVICE))

    h0.remove()
    h1.remove()

    return {
        "early": torch.cat(early_bufs).numpy().astype(np.float32),
        "late":  torch.cat(late_bufs).numpy().astype(np.float32),
    }


def _probe_auc(X: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return float("nan")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))


def run(model_names: list[str] = DEFAULT_STAGE_MODELS,
        concepts: list[str] = DEFAULT_CONCEPTS,
        n_positions: int = N_POSITIONS,
        seed: int = 42) -> dict:
    """
    Compute probe AUC for every (stage, layer, concept) triple.

    Returns
    -------
    {
      "early"        : np.ndarray (n_stages, n_concepts),
      "late"         : np.ndarray (n_stages, n_concepts),
      "stage_labels" : list[str],
      "concepts"     : list[str],
    }
    """
    np.random.seed(seed)

    # Generate positions once — same positions evaluated across all stages
    boards  = [b.random_kqk_position() for _ in range(n_positions)]
    obs_np  = np.stack([b.board_to_obs(board) for board in boards])
    obs_t   = torch.from_numpy(obs_np).float()
    fens    = [board.fen() for board in boards]
    feats   = compute_board_features(fens)  # (N, 43)

    feat_idx = {name: FEATURE_NAMES.index(name)
                for name in concepts if name in FEATURE_NAMES}

    n_s = len(model_names)
    n_c = len(concepts)
    auc_early = np.full((n_s, n_c), float("nan"))
    auc_late  = np.full((n_s, n_c), float("nan"))

    for s, model_name in enumerate(model_names):
        print(f"[{s + 1}/{n_s}] {model_name}…")
        net  = _load_net(model_name)
        acts = _collect_layer_acts(net, obs_t)

        for c, concept in enumerate(concepts):
            if concept not in feat_idx:
                continue
            y = feats[:, feat_idx[concept]].astype(int)
            auc_early[s, c] = _probe_auc(acts["early"], y, seed)
            auc_late[s, c]  = _probe_auc(acts["late"],  y, seed)

    n_used = min(n_s, len(STAGE_LABELS))
    stage_lbls = STAGE_LABELS[:n_used] if n_s <= len(STAGE_LABELS) else [
        f"Stage {i + 1}" for i in range(n_s)
    ]

    return {"early": auc_early, "late": auc_late,
            "stage_labels": stage_lbls, "concepts": concepts}


def plot(results: dict,
         ax_early: plt.Axes = None,
         ax_late: plt.Axes = None) -> plt.Figure:
    """Two heatmaps (early / late layer): rows = stages, columns = concepts."""
    standalone = ax_early is None
    if standalone:
        n_rows = len(results["stage_labels"])
        fig, (ax_early, ax_late) = plt.subplots(
            1, 2, figsize=(15, max(3, n_rows * 1.1))
        )
    else:
        fig = ax_early.get_figure()

    for ax, data, title in [
        (ax_early, results["early"], "Early layer (after 1st ReLU)"),
        (ax_late,  results["late"],  "Late layer  (after 2nd ReLU)"),
    ]:
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(results["concepts"])))
        ax.set_xticklabels(results["concepts"], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(results["stage_labels"])))
        ax.set_yticklabels(results["stage_labels"], fontsize=8)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUC")

        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                v = data[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="white" if v > 0.85 else "black")

    fig.suptitle("What–When–Where: linear probe AUC per training stage × layer",
                 fontsize=10)
    if standalone:
        plt.tight_layout()
    return fig


if __name__ == "__main__":
    results = run()
    fig = plot(results)
    plt.show()