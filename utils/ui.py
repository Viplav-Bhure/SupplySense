"""Look-and-feel helpers: dark theme CSS, page title, KPI tiles, chart styling."""
from __future__ import annotations

import html

import streamlit as st

# Palette taken from the reference dashboard image
BG = "#0E141A"
PANEL = "#151C23"
LINE = "#232D37"
ORANGE = "#FF9A1F"
BLUE = "#1B9AF0"
GREEN = "#22C55E"
AMBER = "#E39A1D"
MAGENTA = "#D6336C"
RED = "#D6336C"
MUTED = "#8B97A3"
TEXT = "#E9EEF3"

PRIMARY = BLUE
SLATE = MUTED

RISK_COLORS = {"Low": GREEN, "Medium": AMBER, "High": MAGENTA}
RISK_LABELS = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}

_CSS = """
<style>
.stApp{background:
  radial-gradient(900px 480px at 4% -6%, rgba(34,197,94,.13), transparent 60%),
  radial-gradient(900px 480px at 100% -8%, rgba(255,150,0,.10), transparent 60%),
  #0E141A;}
.block-container{padding-top:4rem !important;padding-bottom:3rem;max-width:1320px;}

/* ---- chrome: no sidebar, no deploy button ---- */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"]{display:none !important;}
[data-testid="stAppDeployButton"], .stAppDeployButton, #MainMenu, footer{display:none !important;}
[data-testid="stDecoration"]{display:none;}
[data-testid="stHeader"]{background:transparent !important;}

/* ---- top navigation: pill buttons, orange when active ---- */
[data-testid="stTopNavLink"]{border:1px solid #2A343E !important;border-radius:999px !important;
  padding:6px 16px !important;background:rgba(255,255,255,.025) !important;
  color:#B6C0CA !important;font-weight:500 !important;transition:all .15s ease;}
[data-testid="stTopNavLink"]:hover{border-color:#FF9A1F !important;color:#FFFFFF !important;}
[data-testid="stTopNavLink"][aria-current="page"]{background:#FF9A1F !important;border-color:#FF9A1F !important;
  color:#1A1205 !important;font-weight:700 !important;}
[data-testid="stTopNavLink"][aria-current="page"] *{color:#1A1205 !important;}

/* ---- page title ---- */
.ss-title{color:#fff;font-size:1.9rem;font-weight:700;letter-spacing:-.01em;line-height:1.2;margin:.2rem 0 0;}
.ss-tagline{color:#8B97A3;font-size:.95rem;margin:.15rem 0 0;}
.ss-chip{display:inline-block;margin-top:.5rem;background:rgba(255,154,31,.12);color:#FF9A1F;
  border:1px solid rgba(255,154,31,.35);padding:2px 12px;border-radius:999px;font-size:.75rem;font-weight:600;}

/* ---- KPI tiles (green tinted, like the reference) ---- */
.ss-kpi{background:linear-gradient(145deg, rgba(34,197,94,.16), rgba(34,197,94,.04) 70%);
  border:1px solid rgba(34,197,94,.20);border-radius:20px;padding:16px 16px 14px;height:100%;}
.ss-kpi .ss-head{display:flex;align-items:center;gap:10px;margin-bottom:12px;}
.ss-kpi .ss-dot{width:26px;height:26px;border-radius:50%;flex:none;background:#1C2630;
  display:flex;align-items:center;justify-content:center;}
.ss-kpi .ss-dot i{display:block;width:9px;height:9px;border-radius:50%;}
.ss-kpi .ss-label{color:#B6C0CA;font-size:.82rem;font-weight:500;line-height:1.2;}
.ss-kpi .ss-box{background:rgba(14,20,26,.45);border:1px solid rgba(34,197,94,.18);border-radius:16px;padding:12px 14px;}
.ss-kpi .ss-value{color:#FFFFFF;font-size:1.75rem;font-weight:700;line-height:1.15;word-break:break-word;}
.ss-kpi .ss-note{display:inline-block;margin-top:6px;font-size:.74rem;font-weight:600;padding:1px 9px;
  border-radius:999px;background:rgba(255,255,255,.05);}

/* ---- section headings ---- */
.ss-section{font-size:1.1rem;font-weight:600;color:#fff;margin:6px 0 2px;}
.ss-caption{color:#8B97A3;font-size:.86rem;margin-bottom:8px;}
[data-testid="stHeading"] h3{font-size:1.1rem !important;font-weight:600 !important;margin:0 !important;padding:6px 0 2px !important;}

/* ---- panels (containers whose key starts with "panel") ---- */
[class*="st-key-panel"]{background:#151C23;border:1px solid #232D37;border-radius:20px;padding:1.1rem 1.3rem;}

/* ---- status cards & glossary ---- */
.ss-status{background:#151C23;border:1px solid #232D37;border-radius:16px;padding:16px 18px;}
.ss-status .ss-name{font-weight:600;color:#fff;}
.ss-status .ss-text{color:#8B97A3;font-size:.9rem;margin-top:4px;}
.ss-term{padding:12px 0;border-bottom:1px solid #232D37;}
.ss-term:last-child{border-bottom:none;}
.ss-term .ss-t{color:#FF9A1F;font-weight:700;font-size:.95rem;}
.ss-term .ss-d{color:#CBD3DB;font-size:.92rem;margin-top:3px;line-height:1.5;}

/* ---- buttons ---- */
div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button,
div[data-testid="stPopover"] button{font-weight:600;}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_title(title: str, tagline: str = "", badge: str = "") -> None:
    tag = f'<div class="ss-tagline">{html.escape(tagline)}</div>' if tagline else ""
    chip = f'<span class="ss-chip">{html.escape(badge)}</span>' if badge else ""
    st.markdown(f'<div class="ss-title">{html.escape(title)}</div>{tag}{chip}', unsafe_allow_html=True)


def kpi(label: str, value, note: str = "", accent: str = BLUE, note_color: str = MUTED) -> str:
    note_html = f'<div class="ss-note" style="color:{note_color}">{html.escape(note)}</div>' if note else ""
    return (
        f'<div class="ss-kpi"><div class="ss-head">'
        f'<div class="ss-dot"><i style="background:{accent}"></i></div>'
        f'<div class="ss-label">{html.escape(label)}</div></div>'
        f'<div class="ss-box"><div class="ss-value">{html.escape(str(value))}</div>{note_html}</div></div>'
    )


def kpi_row(cards: list[str]) -> None:
    for col, card in zip(st.columns(len(cards)), cards):
        col.markdown(card, unsafe_allow_html=True)


def section(title: str, caption: str = "", help: str | None = None) -> None:
    """Section heading. Pass `help` to show a ? icon with a hover explanation."""
    if help:
        st.subheader(title, help=help)
    else:
        st.markdown(f'<div class="ss-section">{html.escape(title)}</div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="ss-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)


def status_card(name: str, text: str, color: str = BLUE) -> str:
    return (
        f'<div class="ss-status" style="border-left:5px solid {color}">'
        f'<div class="ss-name">{html.escape(name)}</div>'
        f'<div class="ss-text">{html.escape(text)}</div></div>'
    )


def glossary_terms(items: list[tuple[str, str]]) -> str:
    rows = "".join(
        f'<div class="ss-term"><div class="ss-t">{html.escape(t)}</div><div class="ss-d">{html.escape(d)}</div></div>'
        for t, d in items
    )
    return rows


def style_fig(fig, height: int = 380):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, Segoe UI, Roboto, sans-serif", color="#CBD3DB"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color="#B6C0CA")),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="#232D37", tickfont=dict(color="#8B97A3"))
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zeroline=False, tickfont=dict(color="#8B97A3"))
    return fig