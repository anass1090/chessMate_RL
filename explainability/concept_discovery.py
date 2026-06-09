"""
UMAP + HDBSCAN clustering over trunk activations.
Output: explainability/data/umap_results.npz
  - umap_coords: (N, 2) float32
  - cluster_labels: (N,) int32  (-1 = noise)
  - cluster_mean_activations: (n_clusters, 256) float32
  - cluster_top_fens: object array, shape (n_clusters, 10)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "activations.npz")
OUT_PATH    = os.path.join(os.path.dirname(__file__), "data", "umap_results.npz")

UMAP_NEIGHBORS = 15
UMAP_MIN_DIST  = 0.1
HDBSCAN_MIN    = 550
TOP_FENS_K     = 10


def run(seed: int = 42) -> dict:
    import umap
    import hdbscan

    data        = np.load(DATA_PATH, allow_pickle=True)
    activations = data["activations"].astype(np.float32)  # (N, 256)
    fens        = data["fens"]

    print("Running UMAP…")
    reducer = umap.UMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                        n_components=2, init="random", random_state=seed, n_jobs=1)
    coords  = reducer.fit_transform(activations).astype(np.float32)  # (N, 2)
    print(f"UMAP done, shape={coords.shape}")

    print("Running HDBSCAN…")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN)
    labels    = clusterer.fit_predict(coords).astype(np.int32)
    n_clusters = int(labels.max()) + 1
    print(f"HDBSCAN: {n_clusters} clusters, {(labels == -1).sum()} noise points")

    # Per-cluster mean activations and representative FENs
    mean_acts  = np.zeros((n_clusters, activations.shape[1]), dtype=np.float32)
    top_fens_list = []
    for c in range(n_clusters):
        mask = labels == c
        mean_acts[c] = activations[mask].mean(axis=0)

        # Representative = closest to cluster centroid in 2D
        cluster_coords = coords[mask]
        centroid       = cluster_coords.mean(axis=0)
        dists          = np.linalg.norm(cluster_coords - centroid, axis=1)
        top_local_idx  = np.argsort(dists)[:TOP_FENS_K]
        global_idx     = np.where(mask)[0][top_local_idx]
        top_fens_list.append(fens[global_idx])

    top_fens_arr = np.empty(n_clusters, dtype=object)
    for i, f in enumerate(top_fens_list):
        top_fens_arr[i] = f

    # Per-cluster feature descriptions
    from explainability.visualizer import compute_board_features, FEATURE_NAMES, BINARY_FEATURES
    all_features = compute_board_features(list(fens))          # (N, 28)
    global_mean  = all_features.mean(axis=0)
    global_std   = all_features.std(axis=0) + 1e-8

    cluster_descriptions = {}
    for c in range(n_clusters):
        mask         = labels == c
        cluster_feat = all_features[mask].mean(axis=0)         # (28,)
        z_scores     = (cluster_feat - global_mean) / global_std
        top3_idx     = np.argsort(np.abs(z_scores))[-3:][::-1]
        parts = []
        for idx in top3_idx:
            val      = cluster_feat[idx]
            z        = z_scores[idx]
            name     = FEATURE_NAMES[idx]
            if name in BINARY_FEATURES:
                pct   = int(round(val * 100))
                truth = "True" if val >= 0.5 else "False"
                parts.append(f"{name}: {truth} ({pct}%)")
            else:
                g_avg = global_mean[idx]
                parts.append(f"{name}: {val:.2f} (avg {g_avg:.2f})")
        cluster_descriptions[c] = " | ".join(parts)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        umap_coords=coords,
        cluster_labels=labels,
        cluster_mean_activations=mean_acts,
        cluster_top_fens=top_fens_arr,
    )
    print(f"Saved UMAP results → {OUT_PATH}")

    return dict(umap_coords=coords, cluster_labels=labels,
                cluster_mean_activations=mean_acts, cluster_top_fens=top_fens_arr,
                cluster_descriptions=cluster_descriptions)


if __name__ == "__main__":
    run()
