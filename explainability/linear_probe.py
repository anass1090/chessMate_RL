"""
Formal linear probing: for each binary board concept, train a logistic
regression on the 256-dim trunk activations and report accuracy + AUC.

This answers: "Did the network learn concept X in its internal
representation?" High AUC means the concept is linearly decodable from
the trunk — the agent tracks it even without explicit supervision.

Output: dict {concept_name: {accuracy, balanced_accuracy, auc}}
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "activations.npz")

from explainability.visualizer import compute_board_features, FEATURE_NAMES, BINARY_FEATURES


def probe_all(activations_path: str = DATA_PATH,
              test_size: float = 0.2,
              seed: int = 42) -> dict[str, dict]:
    """
    Probe every binary concept in BINARY_FEATURES against trunk activations.

    Returns
    -------
    results : dict {concept_name: {accuracy, balanced_accuracy, auc}}
              sorted by AUC descending
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    data        = np.load(activations_path, allow_pickle=True)
    activations = data["activations"].astype(np.float32)  # (N, 256)
    fens        = list(data["fens"])

    features = compute_board_features(fens)  # (N, 43)

    X_tr, X_te, f_tr, f_te = train_test_split(
        activations, features, test_size=test_size, random_state=seed
    )

    results: dict[str, dict] = {}
    for i, name in enumerate(FEATURE_NAMES):
        if name not in BINARY_FEATURES:
            continue

        y_tr = f_tr[:, i].astype(int)
        y_te = f_te[:, i].astype(int)

        if y_tr.sum() < 10 or (len(y_tr) - y_tr.sum()) < 10:
            continue

        clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        y_pred  = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]

        results[name] = {
            "accuracy":          float(accuracy_score(y_te, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred)),
            "auc":               float(roc_auc_score(y_te, y_proba)),
        }

    return dict(sorted(results.items(), key=lambda kv: kv[1]["auc"], reverse=True))


def plot_probe_results(results: dict[str, dict],
                       metric: str = "auc",
                       ax: plt.Axes = None) -> plt.Figure:
    """Horizontal bar chart of probe AUC (or other metric) per concept."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, max(4, len(results) * 0.38)))
    else:
        fig = ax.get_figure()

    names  = list(results.keys())
    values = [results[n][metric] for n in names]
    colors = ["#1565C0" if v >= 0.75 else "#90CAF9" for v in values]

    ax.barh(range(len(names)), values, color=colors)
    ax.axvline(0.5,  color="grey",   linestyle="--", linewidth=0.8, label="chance (0.5)")
    ax.axvline(0.75, color="orange", linestyle="--", linewidth=0.8, label="strong (0.75)")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title("Linear probe results — trunk activations")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        plt.tight_layout()
    return fig


if __name__ == "__main__":
    results = probe_all()
    print(f"{'Concept':<30s}  {'AUC':>5s}  {'Bal-Acc':>7s}")
    print("-" * 48)
    for name, m in results.items():
        print(f"  {name:<28s}  {m['auc']:.3f}  {m['balanced_accuracy']:.3f}")
    plot_probe_results(results)
    plt.show()