"""Variational Autoencoder for anomaly detection on sales windows.

Train on windows of normal sales; windows the VAE reconstructs poorly
(high reconstruction error) are flagged as anomalies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.preprocessing import make_windows

SAVE_PATH = Path(__file__).resolve().parent / "saved" / "vae.pt"


class VAE(nn.Module):
    def __init__(self, input_dim: int = 28, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    rec = F.mse_loss(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (rec + beta * kld) / x.size(0)


class AnomalyDetector:
    def __init__(self, window: int = 28, hidden_dim: int = 32, latent_dim: int = 8,
                 lr: float = 1e-3, epochs: int = 50, batch_size: int = 64,
                 threshold_percentile: float = 99.0, device: str | None = None):
        self.window, self.hidden_dim, self.latent_dim = window, hidden_dim, latent_dim
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.threshold_percentile = threshold_percentile
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE(window, hidden_dim, latent_dim).to(self.device)
        self.threshold = None

    @staticmethod
    def _norm(w: np.ndarray) -> np.ndarray:
        """Scale each window by its own mean so products with different volumes compare."""
        m = w.mean(axis=1, keepdims=True)
        return w / np.clip(m, 1.0, None)

    def fit(self, series_list):
        X = np.concatenate([make_windows(s, self.window) for s in series_list])
        Xn = torch.tensor(self._norm(X))
        loader = DataLoader(TensorDataset(Xn), batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device)
                opt.zero_grad()
                recon, mu, logvar = self.model(xb)
                loss = vae_loss(recon, xb, mu, logvar)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            if epoch % 10 == 0 or epoch == 1:
                print(f"epoch {epoch:03d} loss={total / len(Xn):.4f}")
        errors = self.reconstruction_error(X)
        self.threshold = float(np.percentile(errors, self.threshold_percentile))

    @torch.no_grad()
    def reconstruction_error(self, windows: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = torch.tensor(self._norm(windows)).to(self.device)
        mu, _ = self.model.encode(x)               # deterministic: use the mean
        recon = self.model.dec(mu)
        return ((recon - x) ** 2).mean(dim=1).cpu().numpy()

    def detect(self, series: np.ndarray, stride: int = 1) -> dict:
        """Score every window; return indices (end day of window) flagged as anomalous."""
        windows = make_windows(series, self.window, stride)
        if len(windows) == 0:
            return {"scores": np.array([]), "flagged_end_idx": np.array([], dtype=int)}
        scores = self.reconstruction_error(windows)
        end_idx = np.arange(self.window - 1, self.window - 1 + len(windows) * stride, stride)
        return {"scores": scores, "flagged_end_idx": end_idx[scores > self.threshold]}

    def save(self, path: Path = SAVE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict(), "threshold": self.threshold,
                    "cfg": dict(window=self.window, hidden_dim=self.hidden_dim,
                                latent_dim=self.latent_dim)}, path)

    @classmethod
    def load(cls, path: Path = SAVE_PATH) -> "AnomalyDetector":
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(**ckpt["cfg"], device="cpu")
        obj.model.load_state_dict(ckpt["state"])
        obj.threshold = ckpt["threshold"]
        return obj


def zscore_fallback(series: np.ndarray, window: int = 28, z: float = 3.0) -> np.ndarray:
    """Simple rolling z-score detector used until the VAE is trained."""
    s = np.asarray(series, dtype="float32")
    flagged = []
    for i in range(window, len(s)):
        ref = s[i - window:i]
        if ref.std() > 0 and abs(s[i] - ref.mean()) > z * ref.std():
            flagged.append(i)
    return np.array(flagged, dtype=int)


if __name__ == "__main__":
    from utils.preprocessing import clean_sales, generate_sample_data, load_sales_data, product_series

    sales = load_sales_data()
    if sales is None:
        sales, _ = generate_sample_data()
    sales = clean_sales(sales)
    series = [product_series(sales, p) for p in sales["product_id"].unique()]

    det = AnomalyDetector(epochs=50)
    det.fit(series)
    det.save()
    print("Threshold:", det.threshold)
