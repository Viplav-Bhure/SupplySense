"""Helpers for user-uploaded data: reading, column guessing, validation, templates."""
from __future__ import annotations

import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.preprocessing import build_simulated_inventory, clean_sales

MIN_DAYS = 90        # minimum history (calendar days) needed to forecast
MIN_ACTIVE_DAYS = 30  # a product needs at least this many days with sales rows
MAX_PRODUCTS = 60    # keep the dashboard fast

DATE_HINTS = ["date", "day", "ds", "order_date", "invoicedate", "timestamp", "time"]
PRODUCT_HINTS = ["product_id", "product", "item", "item_id", "sku", "stockcode", "stock_code", "description"]
UNITS_HINTS = ["units_sold", "units", "sales", "quantity", "qty", "demand", "sold"]

INVENTORY_ALIASES = {
    "product_id": ["product_id", "product", "item", "item_id", "sku"],
    "current_stock": ["current_stock", "stock", "on_hand", "inventory", "quantity_on_hand"],
    "on_order": ["on_order", "open_orders", "incoming"],
    "backorders": ["backorders", "back_orders"],
    "lead_time_days": ["lead_time_days", "lead_time", "leadtime"],
    "supplier_delay_days": ["supplier_delay_days", "supplier_delay", "delay_days", "delay"],
}
INVENTORY_DEFAULTS = {"on_order": 0, "backorders": 0, "lead_time_days": 7, "supplier_delay_days": 0}


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_table(file) -> pd.DataFrame:
    """Read an uploaded CSV / Excel file into a DataFrame."""
    name = file.name.lower()
    raw = file.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    if df.shape[1] == 1 and ";" in str(df.columns[0]):  # semicolon-separated export
        df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1")
    return df


def guess_column(columns: List[str], hints: List[str]) -> int:
    """Index of the column whose name best matches the hints (0 if none)."""
    lowered = [str(c).strip().lower().replace(" ", "_") for c in columns]
    for h in hints:
        if h in lowered:
            return lowered.index(h)
    for h in hints:
        for i, c in enumerate(lowered):
            if h in c:
                return i
    return 0


# --------------------------------------------------------------------------- #
# Sales
# --------------------------------------------------------------------------- #
def parse_dates(values: pd.Series) -> Tuple[pd.Series, bool]:
    """Parse dates, trying month-first and day-first and keeping whichever parses more rows."""
    month_first = pd.to_datetime(values, errors="coerce")
    day_first = pd.to_datetime(values, errors="coerce", dayfirst=True)
    if day_first.isna().sum() < month_first.isna().sum():
        return day_first, True
    return month_first, False


def build_sales(df: pd.DataFrame, date_col: str, product_col: Optional[str],
                units_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and convert an uploaded table to [date, product_id, units_sold]
    (continuous daily series per product). Raises ValueError with a friendly message."""
    notes: List[str] = []
    dates, day_first = parse_dates(df[date_col])
    if day_first:
        notes.append("Dates were read as day/month/year.")
    work = pd.DataFrame({
        "date": dates,
        "units_sold": pd.to_numeric(df[units_col], errors="coerce"),
    })
    work["product_id"] = "PRODUCT" if product_col is None else df[product_col].astype(str).str.strip()

    bad = work["date"].isna() | work["units_sold"].isna()
    if bad.all():
        raise ValueError("No valid rows found. Check that the date and quantity columns are correct.")
    if bad.any():
        notes.append(f"Dropped {int(bad.sum()):,} rows with an invalid date or quantity.")
    work = work[~bad]

    neg = work["units_sold"] < 0
    if neg.any():
        notes.append(f"Removed {int(neg.sum()):,} rows with negative quantities (returns).")
        work = work[~neg]

    work = work.assign(date=work["date"].dt.normalize())
    span = (work["date"].max() - work["date"].min()).days + 1
    if span < MIN_DAYS:
        raise ValueError(f"Only {span} days of history found; at least {MIN_DAYS} are needed to forecast.")

    active = work.groupby("product_id")["date"].nunique()
    keep = active[active >= MIN_ACTIVE_DAYS]
    dropped = len(active) - len(keep)
    if dropped:
        notes.append(f"Skipped {dropped} product(s) with fewer than {MIN_ACTIVE_DAYS} days of sales.")
    if keep.empty:
        raise ValueError("No product has enough sales history.")

    totals = work[work["product_id"].isin(keep.index)].groupby("product_id")["units_sold"].sum()
    if len(totals) > MAX_PRODUCTS:
        notes.append(f"Kept the {MAX_PRODUCTS} highest-volume products out of {len(totals)}.")
        totals = totals.nlargest(MAX_PRODUCTS)

    cleaned = clean_sales(work[work["product_id"].isin(totals.index)])
    notes.append(f"Ready: {cleaned['product_id'].nunique()} product(s), "
                 f"{cleaned['date'].nunique()} days of history.")
    return cleaned, notes


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def standardize_inventory(df: pd.DataFrame, sales: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Map an uploaded inventory table to the expected columns, fill defaults and
    simulate rows for products that are in the sales data but not in the file."""
    notes: List[str] = []
    lowered = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    out = pd.DataFrame()
    for target, aliases in INVENTORY_ALIASES.items():
        src = next((lowered[a] for a in aliases if a in lowered), None)
        if src is not None:
            out[target] = df[src]
    for required in ("product_id", "current_stock"):
        if required not in out:
            raise ValueError(f"Inventory file needs a '{required}' column "
                             f"(accepted names: {', '.join(INVENTORY_ALIASES[required])}).")
    for col, default in INVENTORY_DEFAULTS.items():
        if col not in out:
            out[col] = default
            notes.append(f"No '{col}' column: assumed {default}.")

    out["product_id"] = out["product_id"].astype(str).str.strip()
    for col in out.columns.drop("product_id"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(INVENTORY_DEFAULTS.get(col, 0))
    out = out.drop_duplicates("product_id")

    ids = set(sales["product_id"].unique())
    out = out[out["product_id"].isin(ids)]
    missing = sorted(ids - set(out["product_id"]))
    if missing:
        sim = build_simulated_inventory(sales[sales["product_id"].isin(missing)])
        out = pd.concat([out, sim], ignore_index=True)
        notes.append(f"{len(missing)} product(s) were not in the inventory file: values were simulated.")
    cols = ["product_id", "current_stock", "on_order", "backorders", "lead_time_days", "supplier_delay_days"]
    return out[cols].reset_index(drop=True), notes


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
def sales_template_csv() -> bytes:
    dates = pd.date_range("2025-01-01", periods=4)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "product_id": "SKU-001",
                       "units_sold": [12, 15, 9, 14]})
    return df.to_csv(index=False).encode()


def inventory_template_csv() -> bytes:
    df = pd.DataFrame([{"product_id": "SKU-001", "current_stock": 80, "on_order": 0, "backorders": 0,
                        "lead_time_days": 8, "supplier_delay_days": 0}])
    return df.to_csv(index=False).encode()
