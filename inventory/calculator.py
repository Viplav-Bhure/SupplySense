"""Inventory Calculation Engine.

Converts predicted daily demand into reorder points, safety stock, stockout
warnings and recommended order quantities.

Assumptions (v1, document these in your report)
-----------------------------------------------
* Daily demand is roughly normal: mean = forecast, std = recent demand std.
* Delivery wait ("lead time" below) = supplier lead time + any current supplier delay.
* Safety stock = z * sigma_daily * sqrt(delivery wait), z from the target service level.
* Reorder point = expected demand during the delivery wait + safety stock.
* Inventory position = on-hand + on-order - backorders. Orders already on the way
  are assumed to arrive within the delivery wait.
* Target inventory level = reorder point + demand over the review period.
* Status: Critical  -> position < expected demand during the wait (will run out)
          Reorder   -> position < reorder point
          OK        -> otherwise
* Stockout probability = P(demand during the wait > position) under the normal assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist

import pandas as pd


@dataclass
class InventoryInput:
    current_stock: float
    predicted_daily_demand: float
    lead_time_days: float          # delivery wait: supplier lead time + current delay
    demand_std: float = 0.0
    on_order: float = 0.0
    backorders: float = 0.0
    service_level: float = 0.95
    review_period_days: float = 7.0


def z_score(service_level: float) -> float:
    service_level = min(max(service_level, 0.5), 0.999)
    return NormalDist().inv_cdf(service_level)


def expected_lead_time_demand(avg_daily_demand: float, lead_time_days: float) -> float:
    return avg_daily_demand * lead_time_days


def safety_stock(demand_std: float, lead_time_days: float, service_level: float = 0.95) -> float:
    return z_score(service_level) * demand_std * sqrt(max(lead_time_days, 0))


def reorder_point(lead_time_demand: float, safety: float) -> float:
    return lead_time_demand + safety


def inventory_position(on_hand: float, on_order: float = 0.0, backorders: float = 0.0) -> float:
    return on_hand + on_order - backorders


def target_inventory_level(rop: float, avg_daily_demand: float, review_period_days: float) -> float:
    return rop + avg_daily_demand * review_period_days


def recommended_order_quantity(target_level: float, position: float) -> float:
    return max(target_level - position, 0.0)


def days_of_cover(position: float, avg_daily_demand: float) -> float:
    """How many days the inventory position lasts at the forecast daily demand."""
    if avg_daily_demand <= 0:
        return float("inf") if position > 0 else 0.0
    return max(position, 0.0) / avg_daily_demand


def stockout_probability(position: float, avg_daily_demand: float, demand_std: float,
                         wait_days: float) -> float:
    """Chance that demand during the delivery wait is larger than the inventory position."""
    mean = avg_daily_demand * wait_days
    sd = demand_std * sqrt(max(wait_days, 0))
    if position <= 0 < mean:
        return 1.0
    if sd < 1e-9:
        return 1.0 if position < mean else 0.0
    return float(1.0 - NormalDist().cdf((position - mean) / sd))


def stockout_status(position: float, rop: float, lead_time_demand: float) -> str:
    if position < lead_time_demand:
        return "Critical"
    if position < rop:
        return "Reorder"
    return "OK"


def status_reason(status: str, position: float, rop: float, lead_time_demand: float,
                  wait_days: float, avg_daily_demand: float) -> str:
    if status == "Critical":
        cover = days_of_cover(position, avg_daily_demand)
        return (f"Position {position:.0f} units lasts {cover:.1f} days, but delivery takes "
                f"{wait_days:.0f} days (needs {lead_time_demand:.0f} units)")
    if status == "Reorder":
        return f"Position {position:.0f} is {rop - position:.0f} units below the reorder point ({rop:.0f})"
    return f"Position {position:.0f} is above the reorder point ({rop:.0f})"


def calculate(inp: InventoryInput) -> dict:
    ltd = expected_lead_time_demand(inp.predicted_daily_demand, inp.lead_time_days)
    ss = safety_stock(inp.demand_std, inp.lead_time_days, inp.service_level)
    rop = reorder_point(ltd, ss)
    pos = inventory_position(inp.current_stock, inp.on_order, inp.backorders)
    target = target_inventory_level(rop, inp.predicted_daily_demand, inp.review_period_days)
    qty = recommended_order_quantity(target, pos)
    status = stockout_status(pos, rop, ltd)
    return {
        "lead_time_demand": ltd,
        "safety_stock": ss,
        "reorder_point": rop,
        "inventory_position": pos,
        "target_level": target,
        "order_quantity": qty,
        "days_of_cover": days_of_cover(pos, inp.predicted_daily_demand),
        "stockout_prob": stockout_probability(pos, inp.predicted_daily_demand, inp.demand_std,
                                              inp.lead_time_days),
        "reorder_required": pos < rop,
        "status": status,
        "status_reason": status_reason(status, pos, rop, ltd, inp.lead_time_days,
                                       inp.predicted_daily_demand),
    }


def evaluate_inventory(
    df: pd.DataFrame, service_level: float = 0.95, review_period_days: float = 7.0
) -> pd.DataFrame:
    """Vectorised wrapper. Required columns: product_id, current_stock,
    predicted_daily_demand, lead_time_days (delivery wait). Optional: demand_std,
    on_order, backorders."""
    results = []
    for _, r in df.iterrows():
        results.append(calculate(
            InventoryInput(
                current_stock=float(r["current_stock"]),
                predicted_daily_demand=float(r["predicted_daily_demand"]),
                lead_time_days=float(r["lead_time_days"]),
                demand_std=float(r.get("demand_std", 0.0)),
                on_order=float(r.get("on_order", 0.0)),
                backorders=float(r.get("backorders", 0.0)),
                service_level=service_level,
                review_period_days=review_period_days,
            )
        ))
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)


if __name__ == "__main__":
    # Worked example from the project brief
    ltd = expected_lead_time_demand(15, 8)          # 120
    rop = reorder_point(ltd, 33)                    # 153
    print("Lead-time demand:", ltd, "| Reorder point:", rop, "| Replenish?", 80 < rop)
    print(calculate(InventoryInput(80, 15, 8, demand_std=4.0)))