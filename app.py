"""SupplySense web app.   Run with:  streamlit run app.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inventory.calculator import evaluate_inventory
from models.bayesian_risk import StockoutRiskModel
from models.lstm_forecaster import (SAVE_PATH as LSTM_PATH, LSTMForecaster, metrics,
                                    moving_average_baseline, naive_baseline)
from models.vae_anomaly import SAVE_PATH as VAE_PATH, AnomalyDetector, zscore_fallback
from utils import ui
from utils.preprocessing import (TRAIN_CSV, build_simulated_inventory, clean_sales,
                                 generate_sample_data, load_inventory_data, load_sales_data,
                                 train_val_split)
from utils.uploads import (DATE_HINTS, MIN_DAYS, PRODUCT_HINTS, UNITS_HINTS, build_sales,
                           guess_column, inventory_template_csv, read_table,
                           sales_template_csv, standardize_inventory)

st.set_page_config(page_title="SupplySense", page_icon="📦", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()

HORIZON = 7
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.svg"
SOURCE_LABELS = {"kaggle": "", "sample": "Synthetic sample data", "upload": "Your uploaded data"}
STATUS_LABELS = {"Critical": "🚨 Critical", "Reorder": "⚠️ Reorder", "OK": "✅ OK"}
RISK_RANK = {"Low": 0, "Medium": 1, "High": 2}

REORDER_HELP = ("Reorder point is the stock level at which you should place a new order: the demand "
                "expected during the delivery wait plus a safety cushion. Position is stock on hand + on "
                "order − backorders. When the Position bar drops below the Reorder point bar, order now.")

SERVICE_LEVEL_HELP = ("How sure you want to be of not running out while waiting for a delivery. "
                      "95% means you accept roughly a 1-in-20 chance of a stockout. "
                      "Higher = more safety stock and more money tied up in inventory.")
REVIEW_PERIOD_HELP = ("How often you check your stock and place orders (7 = every week). "
                      "Between checks you can't react, so each order must also cover those days: "
                      "a longer review period means bigger, less frequent orders.")
ENGINE_HELP = ("How demand is forecast. Auto uses the trained LSTM model when one is available. "
               "Moving-average baseline uses the average of the last 14 days, a simple method you can "
               "compare the LSTM against.")

GLOSSARY_TERMS = [
    ("Reorder point", "The stock level at which you should place a new order. It equals the demand you expect "
                      "while waiting for delivery, plus a safety cushion. It is compared with Position, not with Stock."),
    ("Position", "Stock on hand + stock already on order − backorders."),
    ("Delivery wait", "Supplier lead time + any current supplier delay."),
    ("Safety stock", "The extra cushion for unexpected demand. Grows with the service level and demand swings."),
    ("Days of cover", "How many days your position lasts at the forecast daily demand."),
    ("Target service level", "How sure you want to be of not running out during a delivery wait."),
    ("Review period", "How often you check stock and order. It only changes order quantities."),
    ("Stockout chance", "Probability that demand during the delivery wait exceeds your position "
                        "(assumes demand varies normally around the forecast)."),
    ("Status", "Critical = position will not last the delivery wait. Reorder = position is below the reorder "
               "point. OK = above it."),
    ("Risk level", "Points are added up from stock cover (the main factor, counts 4×), demand trend (forecast vs "
                   "last 60 days) and supplier delay (0–2 each). Low stock cover is always High. Medium cover "
                   "becomes High when demand and delay together are high; high cover becomes Medium only when "
                   "both are bad."),
]


# =========================================================================== #
# Data sources
# =========================================================================== #
@st.cache_data(show_spinner="Loading dataset…")
def load_kaggle():
    sales = load_sales_data()
    if sales is None:
        return None
    sales = clean_sales(sales)
    return sales, load_inventory_data(sales=sales)


@st.cache_data(show_spinner=False)
def load_sample():
    sales, inv = generate_sample_data()
    return clean_sales(sales), inv


def set_dataset(sales: pd.DataFrame, inventory: pd.DataFrame, kind: str) -> None:
    ss = st.session_state
    ss["sales"] = sales
    ss["inventory"] = inventory.reset_index(drop=True)
    ss["source_kind"] = kind
    ss["source_label"] = SOURCE_LABELS[kind]
    ss["data_version"] = ss.get("data_version", 0) + 1
    for key in ("lstm_custom", "vae_custom", "lstm_metrics"):  # trained on the old data
        ss.pop(key, None)


def init_state() -> None:
    ss = st.session_state
    if "sales" in ss:
        return
    ss.setdefault("lstm_version", 0)
    ss.setdefault("vae_version", 0)
    loaded = load_kaggle()
    if loaded is not None:
        set_dataset(loaded[0], loaded[1], "kaggle")
    else:
        sales, inv = load_sample()
        set_dataset(sales, inv, "sample")


@st.cache_data(show_spinner=False)
def get_series_map(sales: pd.DataFrame) -> dict:
    """product_id -> (dates, daily units) sorted by date."""
    out = {}
    for pid, g in sales.groupby("product_id"):
        g = g.sort_values("date")
        out[pid] = (g["date"].to_numpy(), g["units_sold"].to_numpy(dtype="float32"))
    return out


# =========================================================================== #
# Models
# =========================================================================== #
@st.cache_resource
def load_saved_lstm():
    return LSTMForecaster.load(LSTM_PATH) if Path(LSTM_PATH).exists() else None


@st.cache_resource
def load_saved_vae():
    return AnomalyDetector.load(VAE_PATH) if Path(VAE_PATH).exists() else None


@st.cache_resource
def get_risk_model():
    return StockoutRiskModel()


def scale_mismatch(model, sales: pd.DataFrame) -> bool:
    """True if the saved LSTM was trained on data of a very different size (e.g. old dataset)."""
    mean = getattr(model, "mean", None)
    data_mean = float(sales["units_sold"].mean())
    return bool(mean and data_mean > 0 and not (1 / 3 <= mean / data_mean <= 3))


def active_lstm():
    ss = st.session_state
    if str(ss.get("engine", "Auto")).startswith("Moving"):
        return None, "baseline"
    if "lstm_custom" in ss:
        return ss["lstm_custom"], f"custom{ss.get('lstm_version', 0)}"
    saved = load_saved_lstm()
    if saved is None:
        return None, "baseline"
    if scale_mismatch(saved, ss["sales"]):
        return None, "stale"
    return saved, "saved"


def active_vae():
    ss = st.session_state
    if "vae_custom" in ss:
        return ss["vae_custom"], f"custom{ss.get('vae_version', 0)}"
    saved = load_saved_vae()
    return (saved, "saved") if saved is not None else (None, "zscore")


def engine_label(tag: str) -> str:
    if tag.startswith("custom"):
        return "LSTM (trained on your data)"
    return "LSTM (pre-trained)" if tag == "saved" else "Moving average"


def detector_label(tag: str) -> str:
    if tag.startswith("custom"):
        return "VAE (trained on your data)"
    return "VAE (pre-trained)" if tag == "saved" else "Rolling z-score"


def stale_notice() -> None:
    if active_lstm()[1] == "stale":
        st.warning("The saved LSTM was trained on data of a different size, so a moving-average forecast "
                   "is used for now. Retrain with `python -m models.lstm_forecaster` "
                   "(or use **Data & Models**).")


# =========================================================================== #
# Core computations (cached)
# =========================================================================== #
@st.cache_data(show_spinner="Crunching the numbers…")
def compute_table(sales, inventory, demand_pct, extra_delay, service_level, review_days,
                  model_tag, _lstm):
    series = get_series_map(sales)
    rows = []
    for r in inventory.itertuples():
        if r.product_id not in series:
            continue
        hist = series[r.product_id][1]
        pred = _lstm.predict(hist) if _lstm is not None else moving_average_baseline(hist, HORIZON)
        pred = pred * (1 + demand_pct / 100)
        delay = float(r.supplier_delay_days) + extra_delay          # counted exactly once
        rows.append({
            "product_id": r.product_id,
            "current_stock": r.current_stock,
            "predicted_daily_demand": float(pred.mean()),
            "historical_daily_demand": float(hist[-60:].mean()),
            "demand_std": float(hist[-28:].std()),
            "lead_time_days": float(r.lead_time_days) + delay,     # delivery wait = lead time + delay
            "supplier_delay_days": delay,
            "on_order": r.on_order,
            "backorders": r.backorders,
        })
    table = evaluate_inventory(pd.DataFrame(rows), service_level=service_level,
                               review_period_days=review_days)
    table = get_risk_model().assess_table(table)
    table["days_of_cover"] = table["days_of_cover"].replace([np.inf], 999).clip(upper=999)
    table["risk_label"] = table["risk_level"].map(ui.RISK_LABELS)
    table["risk_rank"] = table["risk_level"].map(RISK_RANK)
    table["stockout_pct"] = table["stockout_prob"] * 100
    table["status_label"] = table["status"].map(STATUS_LABELS)
    return table


def get_table(demand_pct: float = 0.0, extra_delay: int = 0) -> pd.DataFrame:
    ss = st.session_state
    lstm, tag = active_lstm()
    return compute_table(ss["sales"], ss["inventory"], float(demand_pct), int(extra_delay),
                         float(ss.get("service_level", 0.95)), float(ss.get("review_days", 7)),
                         tag, lstm)


@st.cache_data(show_spinner="Scanning for anomalies…")
def detect_anomalies(sales, tag, _vae):
    rows = []
    for pid, (dates, hist) in get_series_map(sales).items():
        found = {}  # day index -> VAE error / threshold (0 when unknown)
        if _vae is not None:
            w = _vae.window
            res = _vae.detect(hist)
            scores = res["scores"]
            for e in res["flagged_end_idx"]:
                e = int(e)
                win = hist[e - w + 1:e + 1]
                day = e - w + 1 + int(np.argmax(np.abs(win - np.median(win))))
                k = e - (w - 1)
                ratio = float(scores[k] / _vae.threshold) if len(scores) > k and _vae.threshold else 0.0
                found[day] = max(found.get(day, 0.0), ratio)
        else:
            for i in zscore_fallback(hist):
                found[int(i)] = 0.0

        for i, ratio in sorted(found.items()):
            ref = hist[max(0, i - 28):i]
            expected = float(np.median(ref)) if len(ref) else float(hist[i])
            dev = (float(hist[i]) - expected) / max(expected, 1.0)
            if abs(dev) < 0.2:
                continue
            sev = "High" if abs(dev) >= 0.6 else "Medium" if abs(dev) >= 0.35 else "Low"
            recent = hist[max(0, i - 89):i + 1]
            if hist[i] == 0 and expected > 0:
                parts = [f"No sales recorded vs usual {expected:.0f}"]
            else:
                parts = [f"Sold {hist[i]:.0f} vs usual {expected:.0f} ({dev * 100:+.0f}%)"]
            if len(recent) > 14 and hist[i] >= recent.max():
                parts.append(f"highest day in {len(recent)} days")
            elif len(recent) > 14 and hist[i] <= recent.min():
                parts.append(f"lowest day in {len(recent)} days")
            if _vae is not None and ratio > 0:
                parts.append(f"VAE error {ratio:.1f}x its threshold")
            elif _vae is None and len(ref) >= 7 and ref.std() > 0:
                parts.append(f"z-score {abs(hist[i] - ref.mean()) / ref.std():.1f} (limit 3)")
            rows.append({"product_id": pid, "date": pd.Timestamp(dates[i]), "units_sold": float(hist[i]),
                         "expected": expected, "deviation_pct": dev * 100,
                         "type": "Spike" if dev > 0 else "Drop", "severity": sev,
                         "why": " · ".join(parts)})
    cols = ["product_id", "date", "units_sold", "expected", "deviation_pct", "type", "severity", "why"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["date", "deviation_pct"], ascending=[False, False]).reset_index(drop=True)


def get_anomalies() -> pd.DataFrame:
    vae, tag = active_vae()
    return detect_anomalies(st.session_state["sales"], tag, vae)


def require_table() -> pd.DataFrame:
    table = get_table()
    if table.empty:
        st.error("No products to show. Go to **Data & Models** and load a dataset.")
        st.stop()
    return table


def badge() -> str:
    return st.session_state.get("source_label", "")


def page_head(title: str, tagline: str = "", badge: str = "") -> None:
    """Page title on the left, Settings button on the right (like the Filter button in the design)."""
    left, right = st.columns([6, 1.4], vertical_alignment="center")
    with left:
        ui.page_title(title, tagline, badge)
    with right:
        with st.popover("Settings", icon=":material/tune:", width="stretch"):
            st.slider("Target service level", 0.80, 0.99, 0.95, 0.01, key="service_level", help=SERVICE_LEVEL_HELP)
            st.number_input("Review period (days)", 1, 30, 7, key="review_days", help=REVIEW_PERIOD_HELP)
            st.radio("Forecast engine", ["Auto (best available)", "Moving-average baseline"], key="engine",
                     help=ENGINE_HELP)
    st.write("")


# =========================================================================== #
# Pages
# =========================================================================== #
def page_overview() -> None:
    table = require_table()
    anomalies = get_anomalies()
    page_head("SupplySense", "Forecast demand, spot stockout risk and catch unusual sales before they hurt.",
              badge=badge())
    stale_notice()

    last_date = pd.Timestamp(st.session_state["sales"]["date"].max())
    recent = int((anomalies["date"] >= last_date - pd.Timedelta(days=30)).sum()) if len(anomalies) else 0
    n_high = int((table["risk_level"] == "High").sum())
    n_reorder = int(table["reorder_required"].sum())
    ui.kpi_row([
        ui.kpi("Products tracked", len(table), f"{engine_label(active_lstm()[1])} forecasts", ui.BLUE),
        ui.kpi("High stockout risk", n_high, "need attention now", ui.RED if n_high else ui.GREEN),
        ui.kpi("Reorder required", n_reorder, "below reorder point", ui.AMBER),
        ui.kpi("Units to order", f"{int(table['order_quantity'].sum()):,}", "recommended in total", ui.MAGENTA),
        ui.kpi("Anomalies (30 days)", recent, "unusual sales flagged", ui.ORANGE),
    ])
    st.write("")

    left, right = st.columns([1, 1.6])
    with left:
        with st.container(key="panel_ov_risk"):
            ui.section("Risk breakdown", "How many products sit at each stockout risk level.")
            counts = table["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
            fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.62, sort=False,
                                   marker=dict(colors=[ui.RISK_COLORS[k] for k in counts.index],
                                               line=dict(color="#151C23", width=3)),
                                   textinfo="value"))
            fig.update_layout(showlegend=True, annotations=[dict(text=f"<b>{len(table)}</b><br>products",
                                                                 showarrow=False, font=dict(size=15, color="#FFFFFF"))])
            st.plotly_chart(ui.style_fig(fig, 320), width="stretch")
    with right:
        with st.container(key="panel_ov_attention"):
            ui.section("Needs attention", "Products most likely to run out first.")
            att = table.sort_values(["risk_rank", "stockout_prob", "days_of_cover"],
                                    ascending=[False, False, True]).head(8)
            st.dataframe(
                att[["product_id", "days_of_cover", "risk_label", "order_quantity", "status_reason"]],
                hide_index=True, width="stretch",
                column_config={
                    "product_id": "Product",
                    "days_of_cover": st.column_config.NumberColumn("Days of cover", format="%.1f"),
                    "risk_label": "Risk",
                    "order_quantity": st.column_config.NumberColumn("Order qty", format="%d"),
                    "status_reason": st.column_config.TextColumn("Why", width="large"),
                })

    st.write("")
    with st.container(key="panel_ov_position"):
        ui.section("Position vs reorder point",
                   "The 15 products closest to (or below) their reorder point. Order when the position "
                   "bar drops below the reorder point bar.", help=REORDER_HELP)
        gap = table.assign(gap=table["inventory_position"] - table["reorder_point"]).sort_values("gap").head(15)
        fig = go.Figure()
        fig.add_bar(x=gap["product_id"], y=gap["inventory_position"],
                    name="Position (stock + on order − backorders)", marker_color=ui.BLUE)
        fig.add_bar(x=gap["product_id"], y=gap["reorder_point"], name="Reorder point", marker_color=ui.AMBER)
        fig.update_layout(barmode="group")
        st.plotly_chart(ui.style_fig(fig, 360), width="stretch")


def page_forecast() -> None:
    table = require_table()
    page_head("Demand forecast", "Expected demand for the next 7 days, per product.", badge=badge())
    stale_notice()
    pid = st.selectbox("Product", table["product_id"].tolist())

    dates, hist = get_series_map(st.session_state["sales"])[pid]
    lstm, tag = active_lstm()
    pred = lstm.predict(hist) if lstm is not None else moving_average_baseline(hist, HORIZON)
    future = pd.date_range(pd.Timestamp(dates[-1]) + pd.Timedelta(days=1), periods=HORIZON)
    change = (pred.mean() / max(float(hist[-7:].mean()), 1e-6) - 1) * 100

    ui.kpi_row([
        ui.kpi("Next 7 days", f"{pred.sum():,.0f}", "units expected", ui.BLUE),
        ui.kpi("Average per day", f"{pred.mean():,.1f}", "units / day", ui.MAGENTA),
        ui.kpi("Vs last 7 days", f"{change:+.1f}%", "forecast vs recent actuals",
               ui.AMBER, ui.RED if change > 15 else ui.SLATE),
        ui.kpi("Forecast engine", engine_label(tag), "change in Settings", ui.GREEN),
    ])
    st.write("")

    with st.container(key="panel_fc_chart"):
        ui.section("Actual vs predicted")
        band = 1.645 * float(hist[-28:].std())
        upper, lower = pred + band, np.clip(pred - band, 0, None)
        fig = go.Figure()
        fig.add_scatter(x=future.tolist() + future.tolist()[::-1], y=list(upper) + list(lower)[::-1],
                        fill="toself", fillcolor="rgba(34,197,94,0.14)", line=dict(width=0),
                        hoverinfo="skip", name="Typical range")
        fig.add_scatter(x=dates[-60:], y=hist[-60:], name="Actual", line=dict(color=ui.BLUE, width=2.5))
        fig.add_scatter(x=future, y=pred, name="Predicted", mode="lines+markers",
                        line=dict(color=ui.GREEN, width=2.5, dash="dash"))
        fig.update_yaxes(title="Units sold")
        st.plotly_chart(ui.style_fig(fig, 420), width="stretch")
        st.caption("The shaded band is a rough range based on recent day-to-day variation, "
                   "not a statistical prediction interval.")

    st.write("")
    with st.container(key="panel_fc_table"):
        out = pd.DataFrame({"date": future.date, "predicted_units": pred.round(1)})
        c1, c2 = st.columns([3, 1])
        c1.dataframe(out, hide_index=True, width="stretch")
        c2.download_button("⬇ Download forecast", out.to_csv(index=False).encode(),
                           f"forecast_{pid}.csv", "text/csv", width="stretch")


def page_inventory() -> None:
    table = require_table()
    page_head("Inventory status", "Reorder points, safety stock, risk level and recommended orders.",
              badge=badge())
    stale_notice()

    f1, f2, f3 = st.columns([2, 2, 1.4])
    risks = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    query = f2.text_input("Search product", placeholder="e.g. S03-I17")
    only_reorder = f3.toggle("Reorder only", value=False)

    view = table[table["risk_level"].isin(risks)]
    if query:
        view = view[view["product_id"].str.contains(query, case=False)]
    if only_reorder:
        view = view[view["reorder_required"]]
    view = view.sort_values(["risk_rank", "stockout_prob", "days_of_cover"], ascending=[False, False, True])

    ui.kpi_row([
        ui.kpi("Showing", f"{len(view)} / {len(table)}", "products", ui.BLUE),
        ui.kpi("Critical", int((view["status"] == "Critical").sum()), "stock won't last the delivery wait",
               ui.RED),
        ui.kpi("Units to order", f"{int(view['order_quantity'].sum()):,}", "for this selection", ui.MAGENTA),
    ])
    st.write("")
    with st.container(key="panel_inv_table"):
        ui.section("Product table", "See the Glossary tab for what each column means.")
        show = view[["product_id", "current_stock", "predicted_daily_demand", "days_of_cover", "lead_time_days",
                     "reorder_point", "safety_stock", "inventory_position", "order_quantity", "risk_label",
                     "stockout_pct", "status_label", "status_reason"]].round(1)
        st.dataframe(show, hide_index=True, width="stretch", height=min(60 + 35 * len(show), 560),
                     column_config={
                         "product_id": "Product",
                         "current_stock": st.column_config.NumberColumn("Stock", format="%d"),
                         "predicted_daily_demand": st.column_config.NumberColumn("Demand / day", format="%.1f"),
                         "days_of_cover": st.column_config.NumberColumn("Days of cover", format="%.1f"),
                         "lead_time_days": st.column_config.NumberColumn("Delivery wait (days)", format="%.0f"),
                         "reorder_point": st.column_config.NumberColumn("Reorder point", format="%d"),
                         "safety_stock": st.column_config.NumberColumn("Safety stock", format="%d"),
                         "inventory_position": st.column_config.NumberColumn("Position", format="%d"),
                         "order_quantity": st.column_config.NumberColumn("Order qty", format="%d"),
                         "risk_label": st.column_config.TextColumn("Risk"),
                         "stockout_pct": st.column_config.ProgressColumn(
                             "Stockout chance", min_value=0, max_value=100, format="%.0f%%"),
                         "status_label": st.column_config.TextColumn("Status"),
                         "status_reason": st.column_config.TextColumn("Reason", width="large"),
                     })
        st.download_button("⬇ Download table (CSV)", view.drop(columns=["risk_label", "status_label"])
                           .to_csv(index=False).encode(), "inventory_recommendations.csv", "text/csv")


def page_anomalies() -> None:
    require_table()
    anomalies = get_anomalies()
    _, tag = active_vae()
    page_head("Anomaly alerts", "Sales patterns that look unusual and deserve a second look.", badge=badge())

    last_date = pd.Timestamp(st.session_state["sales"]["date"].max())
    recent = anomalies[anomalies["date"] >= last_date - pd.Timedelta(days=30)] if len(anomalies) else anomalies
    ui.kpi_row([
        ui.kpi("Flagged events", len(anomalies), "in the full history", ui.BLUE),
        ui.kpi("Last 30 days", len(recent), "recent alerts", ui.AMBER),
        ui.kpi("Spikes / drops", f"{int((anomalies['type'] == 'Spike').sum())} / "
                                  f"{int((anomalies['type'] == 'Drop').sum())}", "", ui.MAGENTA),
        ui.kpi("Detector", detector_label(tag), "", ui.GREEN),
    ])
    st.write("")
    with st.expander("How are alerts decided?"):
        st.markdown(
            "- **Detector:** the VAE learns what a normal 28-day sales pattern looks like. A window it cannot "
            "rebuild well (error above a threshold set at the 99th percentile of training errors) is flagged. "
            "Without a trained VAE, a day more than 3 standard deviations from the previous 28 days is flagged.\n"
            "- **Which day:** the day that differs most from the window's median is reported.\n"
            "- **Filter:** alerts under a 20% deviation from the usual level (median of the previous 28 days) "
            "are hidden.\n"
            "- **Severity:** High ≥ 60% deviation, Medium ≥ 35%, Low otherwise.\n"
            "- **Why:** the *Why flagged* column shows the numbers behind each alert.")
    if tag == "zscore":
        st.info("The VAE isn't trained yet, so a simple rolling z-score is used. "
                "Train the VAE in **Data & Models** for better detection.")
    if anomalies.empty:
        st.success("No anomalies flagged.")
        return

    with st.container(key="panel_an_chart"):
        ui.section("Explore a product")
        flagged_products = anomalies["product_id"].value_counts().index.tolist()
        pid = st.selectbox("Product", flagged_products)
        dates, hist = get_series_map(st.session_state["sales"])[pid]
        mine = anomalies[anomalies["product_id"] == pid]
        fig = go.Figure()
        fig.add_scatter(x=dates[-365:], y=hist[-365:], name="Units sold", line=dict(color=ui.BLUE, width=1.8))
        in_range = mine[mine["date"] >= pd.Timestamp(dates[-365:][0])]
        fig.add_scatter(x=in_range["date"], y=in_range["units_sold"], mode="markers", name="Flagged",
                        marker=dict(color=ui.RED, size=10, line=dict(color="white", width=1.5)),
                        text=in_range["why"], hovertemplate="%{x|%d %b %Y}<br>%{text}<extra></extra>")
        st.plotly_chart(ui.style_fig(fig, 380), width="stretch")

    st.write("")
    with st.container(key="panel_an_table"):
        ui.section("All alerts", "Newest first.")
        sev = st.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        shown = anomalies[anomalies["severity"].isin(sev)].copy()
        shown["date"] = shown["date"].dt.date
        st.dataframe(shown, hide_index=True, width="stretch", height=420, column_config={
            "product_id": "Product", "date": "Date",
            "units_sold": st.column_config.NumberColumn("Units sold", format="%d"),
            "expected": st.column_config.NumberColumn("Expected", format="%.0f"),
            "deviation_pct": st.column_config.NumberColumn("Deviation", format="%+.0f%%"),
            "type": "Type", "severity": "Severity",
            "why": st.column_config.TextColumn("Why flagged", width="large")})
        st.download_button("⬇ Download alerts (CSV)", shown.to_csv(index=False).encode(), "anomalies.csv",
                           "text/csv")


def _why_changed(r) -> str:
    parts = []
    if r.demand_state_b != r.demand_state_s:
        parts.append(f"Demand level: {r.demand_state_b} → {r.demand_state_s}")
    if r.inventory_state_b != r.inventory_state_s:
        parts.append(f"Stock cover: {r.inventory_state_b} → {r.inventory_state_s}")
    if r.delay_state_b != r.delay_state_s:
        parts.append(f"Supplier delay: {r.delay_state_b} → {r.delay_state_s}")
    return " · ".join(parts) if parts else "No input level changed"


def page_whatif() -> None:
    require_table()
    page_head("What-if simulator", "Stress-test your inventory against demand surges and supplier delays.",
              badge=badge())
    c1, c2, _ = st.columns([1, 1, 0.6])
    inc = c1.slider("Demand change (%)", -50, 150, 20, 5)
    delay = c2.slider("Extra supplier delay (days)", 0, 21, 0)
    st.caption("Extra delay is added to every product's delivery wait. Leave it at 0 to see the effect of "
               "demand alone.")

    base, scen = get_table(), get_table(inc, delay)
    d_high = int((scen["risk_level"] == "High").sum()) - int((base["risk_level"] == "High").sum())
    d_re = int(scen["reorder_required"].sum()) - int(base["reorder_required"].sum())
    d_qty = int(scen["order_quantity"].sum()) - int(base["order_quantity"].sum())

    def note(d):
        return f"{d:+,} vs baseline", (ui.RED if d > 0 else ui.GREEN if d < 0 else ui.SLATE)

    n1, c_1 = note(d_high); n2, c_2 = note(d_re); n3, c_3 = note(d_qty)
    ui.kpi_row([
        ui.kpi("High-risk products", int((scen["risk_level"] == "High").sum()), n1, ui.RED, c_1),
        ui.kpi("Reorder required", int(scen["reorder_required"].sum()), n2, ui.AMBER, c_2),
        ui.kpi("Units to order", f"{int(scen['order_quantity'].sum()):,}", n3, ui.MAGENTA, c_3),
    ])
    st.write("")

    cols = ["product_id", "risk_label", "order_quantity", "demand_state", "inventory_state", "delay_state"]
    merged = base[cols].merge(scen[cols], on="product_id", suffixes=("_b", "_s"))
    merged["extra_units"] = merged["order_quantity_s"] - merged["order_quantity_b"]
    merged["why"] = [_why_changed(r) for r in merged.itertuples()]

    with st.container(key="panel_wi_chart"):
        ui.section("Extra units you would need to order", "Top 12 products affected by this scenario.")
        top = merged.sort_values("extra_units", ascending=False).head(12)
        fig = go.Figure()
        fig.add_bar(x=top["product_id"], y=top["order_quantity_b"], name="Baseline order", marker_color=ui.BLUE)
        fig.add_bar(x=top["product_id"], y=top["order_quantity_s"], name="Scenario order", marker_color=ui.AMBER)
        fig.update_layout(barmode="group")
        st.plotly_chart(ui.style_fig(fig, 360), width="stretch")

    st.write("")
    with st.container(key="panel_wi_table"):
        ui.section("Product comparison", "The last column shows which input levels moved the risk.")
        only_changed = st.toggle("Only show products whose risk level changed", value=True)
        shown = merged[merged["risk_label_b"] != merged["risk_label_s"]] if only_changed else merged
        if shown.empty:
            st.info("No product changes risk level in this scenario. Try a bigger change.")
        else:
            shown = shown[["product_id", "risk_label_b", "risk_label_s", "why", "order_quantity_b",
                           "order_quantity_s", "extra_units"]]
            st.dataframe(shown.sort_values("extra_units", ascending=False), hide_index=True, width="stretch",
                         column_config={"product_id": "Product", "risk_label_b": "Risk (baseline)",
                                        "risk_label_s": "Risk (scenario)",
                                        "why": st.column_config.TextColumn("What changed", width="large"),
                                        "order_quantity_b": st.column_config.NumberColumn("Order (baseline)", format="%d"),
                                        "order_quantity_s": st.column_config.NumberColumn("Order (scenario)", format="%d"),
                                        "extra_units": st.column_config.NumberColumn("Extra units", format="%+d")})


# --------------------------------------------------------------------------- #
# Data & Models page
# --------------------------------------------------------------------------- #
def train_lstm_in_app(sales: pd.DataFrame, epochs: int) -> None:
    ss = st.session_state
    train, val = [], []
    for _, hist in get_series_map(sales).values():
        tr, va = train_val_split(hist)
        train.append(tr); val.append(va)
    model = LSTMForecaster(epochs=epochs)
    bar = st.progress(0.0, text="Starting training…")
    model.fit(train, on_epoch=lambda e, n, loss: bar.progress(e / n, text=f"Epoch {e}/{n} · loss {loss:.4f}"))
    try:
        ys, ps = [], []
        for v in val:
            for i in range(len(v) - 35 + 1):
                ys.append(v[i + 28:i + 35]); ps.append(naive_baseline(v[i:i + 28]))
        ss["lstm_metrics"] = {"lstm": model.evaluate(val)["mae"],
                              "naive": metrics(np.array(ys), np.array(ps))["mae"]} if ys else None
    except Exception:
        ss["lstm_metrics"] = None
    ss["lstm_custom"] = model
    ss["lstm_version"] = ss.get("lstm_version", 0) + 1
    bar.empty()


def page_data() -> None:
    ss = st.session_state
    sales = ss["sales"]
    page_head("Data & Models", "Choose a dataset, upload your own sales history, and train the models.",
              badge=badge())

    ui.kpi_row([
        ui.kpi("Products", sales["product_id"].nunique(), "", ui.BLUE),
        ui.kpi("Days of history", sales["date"].nunique(), "", ui.MAGENTA),
        ui.kpi("From", str(sales["date"].min().date()), "", ui.AMBER),
        ui.kpi("To", str(sales["date"].max().date()), "", ui.GREEN),
    ])
    st.write("")

    # ---- 1. source --------------------------------------------------------- #
    with st.container(key="panel_data_source"):
        ui.section("1 · Choose a data source")
        kinds = (["kaggle"] if TRAIN_CSV.exists() else []) + ["sample", "upload"]
        current = ss.get("source_kind", "sample")
        option_names = {"kaggle": "📊 Built-in dataset (data/train.csv)", "sample": "🧪 Synthetic sample",
                        "upload": "⬆ Upload my own"}
        choice = st.radio("Data source", kinds, index=kinds.index(current) if current in kinds else 0,
                          format_func=lambda k: option_names[k], horizontal=True, label_visibility="collapsed")

        if choice in ("kaggle", "sample") and choice != current:
            if st.button("Load built-in dataset" if choice == "kaggle" else "Load synthetic sample",
                         type="primary"):
                loaded = load_kaggle() if choice == "kaggle" else load_sample()
                set_dataset(loaded[0], loaded[1], choice)
                st.rerun()

        if choice == "upload":
            st.caption("Upload daily (or transaction-level) sales. One row per product per day or per sale; "
                       f"at least {MIN_DAYS} days of history.")
            with st.expander("Need a template?"):
                t1, t2 = st.columns(2)
                t1.download_button("⬇ Sales template", sales_template_csv(), "sales_template.csv", "text/csv")
                t2.download_button("⬇ Inventory template", inventory_template_csv(), "inventory_template.csv",
                                   "text/csv")
            up = st.file_uploader("Sales history (CSV or Excel)", type=["csv", "xlsx", "xls"], key="sales_upload")
            inv_up = st.file_uploader("Inventory table (optional: stock, lead time, delays)",
                                      type=["csv", "xlsx", "xls"], key="inv_upload")
            if up is not None:
                try:
                    raw = read_table(up)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read that file: {exc}")
                    raw = None
                if raw is not None:
                    st.dataframe(raw.head(8), hide_index=True, width="stretch")
                    cols = [str(c) for c in raw.columns]
                    raw.columns = cols
                    m1, m2, m3 = st.columns(3)
                    date_col = m1.selectbox("Date column", cols, index=guess_column(cols, DATE_HINTS))
                    none_opt = "(single product, no product column)"
                    prod_pick = m2.selectbox("Product column", [none_opt] + cols,
                                             index=1 + guess_column(cols, PRODUCT_HINTS))
                    units_col = m3.selectbox("Units sold column", cols, index=guess_column(cols, UNITS_HINTS))
                    if st.button("Use this data", type="primary"):
                        try:
                            new_sales, notes = build_sales(raw, date_col,
                                                           None if prod_pick == none_opt else prod_pick, units_col)
                            if inv_up is not None:
                                new_inv, inv_notes = standardize_inventory(read_table(inv_up), new_sales)
                            else:
                                new_inv = build_simulated_inventory(new_sales)
                                inv_notes = ["No inventory file: stock levels, lead times and supplier delays "
                                             "were simulated. Edit them below."]
                            set_dataset(new_sales, new_inv, "upload")
                            ss["upload_notes"] = notes + inv_notes
                            st.rerun()
                        except ValueError as exc:
                            st.error(str(exc))
        if ss.get("source_kind") == "upload":
            for n in ss.get("upload_notes", []):
                st.info(n)

    # ---- 2. inventory -------------------------------------------------------- #
    st.write("")
    with st.container(key="panel_data_inv"):
        ui.section("2 · Inventory inputs", "Edit stock levels, lead times and delays, then apply.")
        edited = st.data_editor(ss["inventory"], hide_index=True, width="stretch", disabled=["product_id"],
                                key=f"inv_editor_{ss['data_version']}")
        if st.button("Apply inventory changes"):
            ss["inventory"] = edited.copy()
            ss["data_version"] += 1
            st.rerun()

    # ---- 3. models ------------------------------------------------------------ #
    st.write("")
    with st.container(key="panel_data_models"):
        ui.section("3 · Models", "Pre-trained models come from `models/saved/`. Train on the current data "
                                "for the best results with your own dataset.")
        stale_notice()
        lstm, ltag = active_lstm()
        vae, vtag = active_vae()
        c1, c2 = st.columns(2)
        c1.markdown(ui.status_card("LSTM demand forecaster", engine_label(ltag),
                                   ui.GREEN if lstm else ui.AMBER), unsafe_allow_html=True)
        c2.markdown(ui.status_card("VAE anomaly detector", detector_label(vtag),
                                   ui.GREEN if vae else ui.AMBER), unsafe_allow_html=True)
        if ss.get("source_kind") == "upload" and ltag == "saved":
            st.warning("The pre-trained LSTM learned from a different dataset. Train it on your data for "
                       "accurate forecasts.")
        m = ss.get("lstm_metrics")
        if m and m["naive"] > 0:
            st.success(f"Validation MAE: LSTM {m['lstm']:.1f} vs naive baseline {m['naive']:.1f} "
                       f"({(1 - m['lstm'] / m['naive']) * 100:+.0f}% better)")

        t1, t2 = st.columns(2)
        with t1:
            epochs = st.slider("LSTM epochs", 5, 30, 10)
            if st.button("Train LSTM on current data"):
                train_lstm_in_app(sales, epochs)
                st.rerun()
        with t2:
            v_epochs = st.slider("VAE epochs", 10, 60, 30)
            if st.button("Train VAE on current data"):
                with st.spinner("Training VAE…"):
                    det = AnomalyDetector(epochs=v_epochs)
                    det.fit([h for _, h in get_series_map(sales).values()])
                ss["vae_custom"] = det
                ss["vae_version"] = ss.get("vae_version", 0) + 1
                st.rerun()
        st.caption("Models trained here live in this browser session. To keep them, use "
                   "`python -m models.lstm_forecaster` and `python -m models.vae_anomaly`.")


# --------------------------------------------------------------------------- #
# Glossary page
# --------------------------------------------------------------------------- #
def page_glossary() -> None:
    page_head("Glossary", "Plain-language meaning of every term used in the app.", badge=badge())
    half = (len(GLOSSARY_TERMS) + 1) // 2
    left, right = st.columns(2)
    with left:
        with st.container(key="panel_gl_left"):
            st.markdown(ui.glossary_terms(GLOSSARY_TERMS[:half]), unsafe_allow_html=True)
    with right:
        with st.container(key="panel_gl_right"):
            st.markdown(ui.glossary_terms(GLOSSARY_TERMS[half:]), unsafe_allow_html=True)


# =========================================================================== #
# Navigation
# =========================================================================== #
def main() -> None:
    init_state()
    try:
        st.logo(str(LOGO_PATH), size="large")
    except Exception:  # noqa: BLE001  (logo is decoration only)
        pass
    pages = [
        st.Page(page_overview, title="Overview", icon=":material/space_dashboard:", url_path="overview",
                default=True),
        st.Page(page_forecast, title="Demand forecast", icon=":material/show_chart:", url_path="forecast"),
        st.Page(page_inventory, title="Inventory", icon=":material/inventory_2:", url_path="inventory"),
        st.Page(page_anomalies, title="Anomalies", icon=":material/warning:", url_path="anomalies"),
        st.Page(page_whatif, title="What-if", icon=":material/tune:", url_path="what-if"),
        st.Page(page_data, title="Data & Models", icon=":material/database:", url_path="data"),
        st.Page(page_glossary, title="Glossary", icon=":material/menu_book:", url_path="glossary"),
    ]
    nav = st.navigation(pages, position="top")
    nav.run()


if __name__ == "__main__":
    main()