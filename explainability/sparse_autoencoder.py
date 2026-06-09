"""
Sparse autoencoder over the 256-dim trunk activations.
Architecture: encoder Linear(256,512)+ReLU, decoder Linear(512,256).
Loss: MSE reconstruction + L1 penalty (lambda=0.01) on encoder activations.
Output: explainability/data/sparse_ae.pt + top-20 FENs per concept.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "activations.npz")
AE_PATH   = os.path.join(os.path.dirname(__file__), "data", "sparse_ae.pt")
TOP_FENS_PATH = os.path.join(os.path.dirname(__file__), "data", "concept_top_fens.npz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM  = 256
HIDDEN_DIM = 512
L1_LAMBDA  = 0.01
BATCH_SIZE = 256
LR         = 1e-3
TOP_K      = 20
PATIENCE   = 20
MIN_DELTA  = 1e-6


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded


def train(batch_size: int = BATCH_SIZE, lr: float = LR,
          l1_lambda: float = L1_LAMBDA, patience: int = PATIENCE,
          min_delta: float = MIN_DELTA,
          seed: int = 42) -> tuple["SparseAutoencoder", list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(DATA_PATH, allow_pickle=True)
    activations = torch.from_numpy(data["activations"]).float()
    fens = data["fens"]

    dataset = TensorDataset(activations)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SparseAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    best_loss = float("inf")
    epochs_no_improve = 0
    epoch = 0

    while True:
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon, encoded = model(batch)
            mse  = nn.functional.mse_loss(recon, batch)
            l1   = l1_lambda * encoded.abs().mean()
            loss = mse + l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        epoch_loss /= len(activations)
        losses.append(epoch_loss)
        epoch += 1

        if (epoch) % 10 == 0:
            print(f"Epoch {epoch:4d}  loss={epoch_loss:.6f}  best={best_loss:.6f}  no_improve={epochs_no_improve}/{patience}")

        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stop at epoch {epoch}  best_loss={best_loss:.6f}")
                break

    os.makedirs(os.path.dirname(AE_PATH), exist_ok=True)
    torch.save(model.state_dict(), AE_PATH)
    print(f"Saved autoencoder → {AE_PATH}")

    _save_top_fens(model, activations, fens)
    return model, losses


def _save_top_fens(model: "SparseAutoencoder", activations: torch.Tensor,
                   fens: np.ndarray) -> None:
    model.eval()
    with torch.no_grad():
        _, encoded = model(activations.to(DEVICE))
        encoded_np = encoded.cpu().numpy()  # (N, 512)

    n_concepts = encoded_np.shape[1]
    top_fens   = []
    for c in range(n_concepts):
        top_idx = np.argsort(encoded_np[:, c])[-TOP_K:][::-1]
        top_fens.append(fens[top_idx])

    np.savez_compressed(TOP_FENS_PATH, top_fens=np.array(top_fens))
    print(f"Saved top-{TOP_K} FENs per concept → {TOP_FENS_PATH}")


def load_ae() -> "SparseAutoencoder":
    model = SparseAutoencoder().to(DEVICE)
    model.load_state_dict(torch.load(AE_PATH, map_location=DEVICE))
    model.eval()
    return model


if __name__ == "__main__":
    train()
