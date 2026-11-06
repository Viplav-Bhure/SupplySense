"""LSTM demand forecaster (PyTorch): past `lookback` days -> next `horizon` days."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.preprocessing import make_supervised_windows

SAVE_PATH = Path(__file__).resolve().parent / "saved" / "lstm.pt"


class LSTMNet(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2,
                 horizon: int = 7, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):                 # x: (batch, lookback, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # (batch, horizon)


class LSTMForecaster:
    def __init__(self, lookback: int = 28, horizon: int = 7, hidden_size: int = 64,
                 num_layers: int = 2, lr: float = 1e-3, epochs: int = 30,
                 batch_size: int = 128, device: str | None = None):
        self.lookback, self.horizon = lookback, horizon
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mean, self.std = 0.0, 1.0
        self.model = LSTMNet(hidden_size, num_layers, horizon).to(self.device)

    # ---- helpers --------------------------------------------------------- #
    def _scale(self, a):   return (a - self.mean) / self.std
    def _unscale(self, a): return a * self.std + self.mean

    def _windows(self, series_list: List[np.ndarray]):
        Xs, ys = [], []
        for s in series_list:
            X, y = make_supervised_windows(s, self.lookback, self.horizon)
            if len(X):
                Xs.append(X); ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    # ---- training -------------------------------------------------------- #
    def fit(self, train_series: List[np.ndarray], val_series: List[np.ndarray] | None = None,
            on_epoch=None):
        """One global model trained on many product series.

        on_epoch(epoch, total_epochs, train_loss) is an optional progress callback.
        """
        all_vals = np.concatenate(train_series)
        self.mean, self.std = float(all_vals.mean()), float(all_vals.std() + 1e-6)

        X, y = self._windows(train_series)
        ds = TensorDataset(torch.tensor(self._scale(X)).unsqueeze(-1),
                           torch.tensor(self._scale(y)))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(xb)
            msg = f"epoch {epoch:02d} train_loss={total / len(ds):.4f}"
            if val_series:
                msg += f" val_mae={self.evaluate(val_series)['mae']:.3f}"
            print(msg)
            if on_epoch:
                on_epoch(epoch, self.epochs, total / len(ds))

    # ---- inference ------------------------------------------------------- #
    @torch.no_grad()
    def predict(self, history: np.ndarray) -> np.ndarray:
        """Forecast the next `horizon` days from the last `lookback` days."""
        self.model.eval()
        x = np.asarray(history[-self.lookback:], dtype="float32")
        if len(x) < self.lookback:
            x = np.pad(x, (self.lookback - len(x), 0), mode="edge")
        t = torch.tensor(self._scale(x)).view(1, -1, 1).to(self.device)
        pred = self._unscale(self.model(t).cpu().numpy()[0])
        return np.clip(pred, 0, None)

    @torch.no_grad()
    def _predict_batch(self, X: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """Predict many windows at once (much faster than one call per window)."""
        self.model.eval()
        out = []
        for i in range(0, len(X), chunk):
            t = torch.tensor(self._scale(X[i:i + chunk]), dtype=torch.float32).unsqueeze(-1).to(self.device)
            out.append(self._unscale(self.model(t).cpu().numpy()))
        return np.clip(np.concatenate(out), 0, None)

    # ---- evaluation ------------------------------------------------------ #
    def evaluate(self, series_list: List[np.ndarray]) -> dict:
        X, y = self._windows(series_list)
        return metrics(y, self._predict_batch(X))

    # ---- persistence ----------------------------------------------------- #
    def save(self, path: Path = SAVE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.model.state_dict(), "mean": self.mean, "std": self.std,
                    "cfg": dict(lookback=self.lookback, horizon=self.horizon,
                                hidden_size=self.hidden_size, num_layers=self.num_layers)}, path)

    @classmethod
    def load(cls, path: Path = SAVE_PATH) -> "LSTMForecaster":
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(**ckpt["cfg"], device="cpu")
        obj.model.load_state_dict(ckpt["state"])
        obj.mean, obj.std = ckpt["mean"], ckpt["std"]
        return obj


# ---- baselines & metrics ------------------------------------------------- #
def naive_baseline(history: np.ndarray, horizon: int = 7) -> np.ndarray:
    """Seasonal naive: repeat the last week."""
    last = np.asarray(history[-7:], dtype="float32")
    return np.resize(last, horizon)


def moving_average_baseline(history: np.ndarray, horizon: int = 7, window: int = 14) -> np.ndarray:
    return np.full(horizon, float(np.mean(history[-window:])), dtype="float32")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err) / np.clip(np.abs(y_true), 1, None)) * 100),
    }


if __name__ == "__main__":
    from utils.preprocessing import (clean_sales, generate_sample_data, load_sales_data,
                                     product_series, train_val_split)

    sales = load_sales_data()
    if sales is None:                       # dataset reading not implemented yet
        sales, _ = generate_sample_data()
    sales = clean_sales(sales)

    train, val = [], []
    for pid in sales["product_id"].unique():
        tr, va = train_val_split(product_series(sales, pid))
        train.append(tr); val.append(va)

    model = LSTMForecaster(epochs=20)
    model.fit(train, val)
    model.save()
    print("LSTM metrics :", model.evaluate(val))

    # Seasonal-naive baseline on the same validation windows
    ys, ps = [], []
    for v in val:
        for i in range(len(v) - 28 - 7 + 1):
            ps.append(naive_baseline(v[i:i + 28])); ys.append(v[i + 28:i + 35])
    print("Naive metrics:", metrics(np.array(ys), np.array(ps)))