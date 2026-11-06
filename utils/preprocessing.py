"""Data loading, cleaning and windowing helpers.

Dataset: Kaggle "Store Item Demand Forecasting Challenge"
         (competition slug: demand-forecasting-kernels-only)
Download `train.csv` and place it at:  data/train.csv
Columns in train.csv: date, store, item, sales  (daily, 10 stores x 50 items, 2013-2017)
Each (store, item) pair is one product, so the full file has 500 products.

The dataset has no stock levels, lead times or supplier delays, so the inventory
table is SIMULATED from the sales history (see `build_simulated_inventory`).
State this clearly in your report.

If data/train.csv is missing, the loaders return None and the app falls back to
`generate_sample_data()` so the dashboard still runs.

Sales format    : [date, product_id, units_sold]
Inventory format: [product_id, current_stock, on_order, backorders,
                   lead_time_days, supplier_delay_days]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
INVENTORY_CSV = DATA_DIR / "inventory.csv"
N_PRODUCTS = 100  # store-item products kept for the dashboard (the dataset has 500)


# --------------------------------------------------------------------------- #
# Dataset reading
# --------------------------------------------------------------------------- #
def load_sales_data(path: Optional[Path] = None,
                    n_products: int = N_PRODUCTS) -> Optional[pd.DataFrame]:
    """Read Store Item Demand `train.csv` and return [date, product_id, units_sold].

    Each (store, item) pair is one product, e.g. "S03-I17". If the file has more than
    `n_products` pairs, an evenly spaced selection is kept so stores and items stay mixed.
    Returns None if the file is not found.
    """
    path = Path(path) if path else TRAIN_CSV
    if not path.exists():
        return None

    raw = pd.read_csv(path, parse_dates=["date"]).rename(columns={"sales": "units_sold"})
    if "store" not in raw.columns:          # item-only file
        raw["store"] = 0

    pairs = raw[["store", "item"]].drop_duplicates().sort_values(["store", "item"]).reset_index(drop=True)
    if n_products < len(pairs):
        pick = np.unique(np.linspace(0, len(pairs) - 1, n_products).round().astype(int))
        pairs = pairs.iloc[pick]
    raw = raw.merge(pairs, on=["store", "item"])

    raw["product_id"] = ("S" + raw["store"].astype(int).astype(str).str.zfill(2)
                         + "-I" + raw["item"].astype(int).astype(str).str.zfill(2))
    return raw[["date", "product_id", "units_sold"]]


def load_inventory_data(path: Optional[Path] = None,
                        sales: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Return the inventory table.

    Reads data/inventory.csv if it exists. Products that are in the sales data but not in
    the file (for example after changing N_PRODUCTS) get simulated rows, and the file is
    updated so the numbers stay stable between runs. Returns None if sales are unavailable.
    """
    path = Path(path) if path else INVENTORY_CSV
    if sales is None:
        sales = load_sales_data()
    if sales is None:
        return None

    ids = set(sales["product_id"].unique())
    inv = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["product_id"])
    inv = inv[inv["product_id"].isin(ids)]
    missing = sorted(ids - set(inv["product_id"]))
    if missing:
        sim = build_simulated_inventory(sales[sales["product_id"].isin(missing)])
        inv = pd.concat([inv, sim], ignore_index=True) if len(inv) else sim
        path.parent.mkdir(parents=True, exist_ok=True)
        inv.to_csv(path, index=False)
    return inv.reset_index(drop=True)


def build_simulated_inventory(sales: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simulate stock, orders and supplier lead times from recent demand."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid, g in sales.sort_values("date").groupby("product_id"):
        avg = float(g["units_sold"].tail(60).mean())
        rows.append({
            "product_id": pid,
            "current_stock": int(avg * rng.uniform(5, 30)),
            "on_order": int(rng.choice([0, 0, avg * 3])),
            "backorders": int(rng.choice([0, 0, 0, avg])),
            "lead_time_days": int(rng.integers(3, 12)),
            "supplier_delay_days": int(rng.choice([0, 0, 1, 3, 5])),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, drop bad rows, aggregate to one row per product per day and
    fill missing days with zero sales."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce")
    df = df.dropna(subset=["date", "product_id", "units_sold"])
    df = df[df["units_sold"] >= 0]  # drop returns / negative quantities

    daily = (
        df.groupby(["product_id", pd.Grouper(key="date", freq="D")])["units_sold"]
        .sum()
        .reset_index()
    )

    # Fill gaps so every product has a continuous daily series
    full_index = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    filled = []
    for pid, g in daily.groupby("product_id"):
        g = g.set_index("date")["units_sold"].reindex(full_index, fill_value=0)
        filled.append(
            pd.DataFrame({"date": full_index, "product_id": pid, "units_sold": g.values})
        )
    return pd.concat(filled, ignore_index=True)


def product_series(sales: pd.DataFrame, product_id) -> np.ndarray:
    """Daily units sold for one product as a float array, sorted by date."""
    s = sales[sales["product_id"] == product_id].sort_values("date")
    return s["units_sold"].to_numpy(dtype="float32")


# --------------------------------------------------------------------------- #
# Windowing (used by LSTM and VAE)
# --------------------------------------------------------------------------- #
def make_supervised_windows(
    series: np.ndarray, lookback: int = 28, horizon: int = 7
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a 1-D series into (X, y): X has `lookback` past days,
    y has the next `horizon` days."""
    X, y = [], []
    for i in range(len(series) - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    if not X:
        return np.empty((0, lookback), dtype="float32"), np.empty((0, horizon), dtype="float32")
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def make_windows(series: np.ndarray, window: int = 28, stride: int = 1) -> np.ndarray:
    """Overlapping windows of a series (for the VAE)."""
    out = [series[i : i + window] for i in range(0, len(series) - window + 1, stride)]
    return np.asarray(out, dtype="float32") if out else np.empty((0, window), dtype="float32")


def train_val_split(series: np.ndarray, val_fraction: float = 0.2):
    """Chronological split (never shuffle time series)."""
    cut = int(len(series) * (1 - val_fraction))
    return series[:cut], series[cut:]


# --------------------------------------------------------------------------- #
# Synthetic sample data (fallback when data/train.csv is missing)
# --------------------------------------------------------------------------- #
def generate_sample_data(
    n_products: int = 12, n_days: int = 365, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    rows, inv_rows = [], []

    for i in range(n_products):
        pid = f"P{i + 1:03d}"
        base = rng.uniform(8, 40)
        weekly = 1 + 0.25 * np.sin(2 * np.pi * (np.arange(n_days) % 7) / 7)
        trend = 1 + 0.0004 * np.arange(n_days) * rng.uniform(-1, 2)
        noise = rng.normal(0, base * 0.15, n_days)
        units = np.clip(base * weekly * trend + noise, 0, None)

        # inject a few anomalies
        for idx in rng.choice(np.arange(30, n_days), size=3, replace=False):
            units[idx] *= rng.choice([0.1, 3.0])
        units = np.round(units)

        rows.append(pd.DataFrame({"date": dates, "product_id": pid, "units_sold": units}))
        lead = int(rng.integers(3, 12))
        inv_rows.append(
            {
                "product_id": pid,
                "current_stock": int(base * rng.uniform(2, 14)),
                "on_order": int(rng.choice([0, 0, base * 3])),
                "backorders": int(rng.choice([0, 0, 0, base])),
                "lead_time_days": lead,
                "supplier_delay_days": int(rng.choice([0, 0, 1, 3, 5])),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(inv_rows)