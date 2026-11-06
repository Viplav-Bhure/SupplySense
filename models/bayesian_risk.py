"""Bayesian Network for stockout risk (pgmpy).

Structure:
    DemandLevel ---\
    InventoryLevel --> StockoutRisk
    SupplierDelay --/

How the risk table (CPT) is built
---------------------------------
Each parent state is turned into points, and the points are added up:
    stock cover  : High stock = 0, Medium = 1, Low = 2   -> counted x4 (main driver)
    demand level : Low = 0, Medium = 1, High = 2
    supplier delay: None = 0, Short = 1, Long = 2
Total score runs from 0 to 12. Scores near 1.5 give Low risk, near 5 Medium risk and
near 8.5 High risk (smoothly, via a Gaussian weight). Because stock cover counts four
times, an item whose stock will not last the delivery wait (cover = Low) always ends up High.

Replace `build_cpd` or call `fit` with real labelled data later to learn the table.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

try:  # pgmpy >= 0.1.26
    from pgmpy.models import DiscreteBayesianNetwork as _BN
except ImportError:  # older pgmpy
    from pgmpy.models import BayesianNetwork as _BN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

DEMAND_STATES = ["Low", "Medium", "High"]
INVENTORY_STATES = ["Low", "Medium", "High"]
DELAY_STATES = ["None", "Short", "Long"]
RISK_STATES = ["Low", "Medium", "High"]

COVER_WEIGHT = 4
RISK_CENTERS = np.array([1.5, 5.0, 8.5])  # score centres for Low / Medium / High risk


# ---- discretisation helpers --------------------------------------------- #
def discretize_demand(predicted_daily: float, historical_daily: float) -> str:
    ratio = predicted_daily / max(historical_daily, 1e-6)
    return "Low" if ratio < 0.9 else "Medium" if ratio < 1.1 else "High"


def discretize_inventory(position: float, predicted_daily: float, wait_days: float) -> str:
    """Stock cover = inventory position / daily demand, compared with the delivery wait."""
    cover = max(position, 0.0) / max(predicted_daily, 1e-6)
    return "Low" if cover < wait_days else "Medium" if cover < 2 * wait_days else "High"


def discretize_delay(delay_days: float) -> str:
    return "None" if delay_days <= 0 else "Short" if delay_days <= 3 else "Long"


# ---- network ------------------------------------------------------------- #
def _root_cpd(name: str, states: list[str], probs: list[float]) -> TabularCPD:
    return TabularCPD(name, len(states), [[p] for p in probs], state_names={name: states})


def build_cpd() -> TabularCPD:
    """P(StockoutRisk | Demand, Inventory, Delay) from the weighted score described above."""
    cols = []
    for d, i, s in itertools.product(range(3), range(3), range(3)):
        score = COVER_WEIGHT * (2 - i) + d + s          # 0 (safe) .. 12 (dangerous)
        w = np.exp(-((score - RISK_CENTERS) ** 2) / 2.0)
        cols.append(w / w.sum())
    values = np.array(cols).T                            # shape (3, 27)
    return TabularCPD(
        "StockoutRisk", 3, values,
        evidence=["DemandLevel", "InventoryLevel", "SupplierDelay"],
        evidence_card=[3, 3, 3],
        state_names={"StockoutRisk": RISK_STATES, "DemandLevel": DEMAND_STATES,
                     "InventoryLevel": INVENTORY_STATES, "SupplierDelay": DELAY_STATES},
    )


class StockoutRiskModel:
    def __init__(self):
        self.model = _BN([("DemandLevel", "StockoutRisk"),
                          ("InventoryLevel", "StockoutRisk"),
                          ("SupplierDelay", "StockoutRisk")])
        self.model.add_cpds(
            _root_cpd("DemandLevel", DEMAND_STATES, [0.25, 0.5, 0.25]),
            _root_cpd("InventoryLevel", INVENTORY_STATES, [0.3, 0.4, 0.3]),
            _root_cpd("SupplierDelay", DELAY_STATES, [0.6, 0.25, 0.15]),
            build_cpd(),
        )
        assert self.model.check_model()
        self.infer = VariableElimination(self.model)
        self._cache: dict = {}   # only 27 possible evidence combinations: query each once

    def risk_distribution(self, demand: str, inventory: str, delay: str) -> dict:
        key = (demand, inventory, delay)
        if key not in self._cache:
            q = self.infer.query(
                ["StockoutRisk"],
                evidence={"DemandLevel": demand, "InventoryLevel": inventory, "SupplierDelay": delay},
                show_progress=False,
            )
            self._cache[key] = dict(zip(q.state_names["StockoutRisk"], (float(v) for v in q.values)))
        return self._cache[key]

    def assess(self, predicted_daily: float, historical_daily: float, position: float,
               wait_days: float, delay_days: float) -> dict:
        """wait_days = supplier lead time + current delay (counted once)."""
        d = discretize_demand(predicted_daily, historical_daily)
        i = discretize_inventory(position, predicted_daily, wait_days)
        s = discretize_delay(delay_days)
        dist = self.risk_distribution(d, i, s)
        label = max(dist, key=dist.get)
        return {"risk_level": label, "p_high": dist["High"], "p_medium": dist["Medium"],
                "p_low": dist["Low"], "demand_state": d, "inventory_state": i, "delay_state": s}

    def assess_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """df needs: predicted_daily_demand, historical_daily_demand, inventory_position,
        lead_time_days (delivery wait) and supplier_delay_days."""
        rows = [self.assess(r.predicted_daily_demand, r.historical_daily_demand,
                            r.inventory_position, r.lead_time_days, r.supplier_delay_days)
                for r in df.itertuples()]
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    def fit(self, data: pd.DataFrame):
        """TODO: learn CPTs from labelled history. `data` needs columns
        DemandLevel, InventoryLevel, SupplierDelay, StockoutRisk (state strings)."""
        from pgmpy.estimators import BayesianEstimator
        self.model.fit(data, estimator=BayesianEstimator, prior_type="BDeu")
        self.infer = VariableElimination(self.model)
        self._cache = {}


if __name__ == "__main__":
    m = StockoutRiskModel()
    print(m.assess(predicted_daily=15, historical_daily=12, position=80, wait_days=11, delay_days=3))