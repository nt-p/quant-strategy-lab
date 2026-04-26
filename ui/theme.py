"""Shared visual theme — CSS injection and Plotly chart defaults.

Call ``inject_css()`` once at the top of every Streamlit page (after
``st.set_page_config``) to apply Google Fonts and global styles globally.
Use the ``CHART_LAYOUT`` / ``AXIS_STYLE`` dicts in ``fig.update_layout``
calls to ensure every Plotly chart matches the app theme.
"""

from __future__ import annotations

import streamlit as st

# ── Chart colour constants ──────────────────────────────────────────────────
BG      = "#0f1116"   # page / paper background
PLOT_BG = "#161b27"   # chart plot area (slightly lighter)
GRID    = "#252836"   # gridline colour
FONT    = "#c4cad6"   # axis labels and tick text

# ── Plotly layout defaults ──────────────────────────────────────────────────
CHART_LAYOUT: dict = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PLOT_BG,
    font=dict(color=FONT, family="DM Sans, sans-serif", size=11),
)

AXIS_STYLE: dict = dict(
    gridcolor=GRID,
    gridwidth=0.5,
    zeroline=False,
    showline=False,
    tickfont=dict(color=FONT, size=10),
)

# ── Global CSS ──────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global font ── */
html, body, [class*="css"],
.stMarkdown, .stText, .stCaption,
[data-testid="stSidebar"], button, input, select, textarea, label {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Monospace numbers in tables and metrics ── */
[data-testid="stMetricValue"],
.stDataFrame td,
.stDataFrame th,
[data-testid="stTable"] td,
[data-testid="stTable"] th {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}

/* ── Main content area ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #1a1e2a !important;
    border-right: 1px solid #252836 !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}
/* Sidebar section headers produced by st.sidebar.header() */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.70rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.8px !important;
    color: #636b78 !important;
    margin: 1.1rem 0 0.3rem 0 !important;
    padding-bottom: 0.25rem !important;
    border-bottom: 1px solid #252836 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.2px !important;
    border-radius: 6px !important;
    transition: background-color 0.15s ease, opacity 0.15s ease !important;
}
.stButton > button[kind="primary"] {
    background-color: #4ecdc4 !important;
    color: #0f1116 !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #3dbdb5 !important;
}
.stButton > button[kind="secondary"] {
    background-color: #1e2333 !important;
    color: #f0f2f5 !important;
    border: 1px solid #2e3347 !important;
}
.stButton > button[kind="secondary"]:hover {
    background-color: #252c3f !important;
}
.stButton > button:disabled {
    opacity: 0.4 !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background-color: #1a1e2a !important;
    border: 1px solid #252836 !important;
    border-radius: 8px !important;
    padding: 0.9rem 1rem !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.70rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: #636b78 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: #f0f2f5 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricDelta"] > div {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Horizontal divider ── */
hr {
    border: none !important;
    border-top: 1px solid #252836 !important;
    margin: 0.75rem 0 !important;
}

/* ── Alert / info boxes ── */
.stAlert {
    border-radius: 6px !important;
}

/* ── Dataframe table ── */
.stDataFrame {
    border-radius: 6px !important;
    overflow: hidden !important;
    font-size: 0.82rem !important;
}

/* ── Caption text ── */
.stCaption p, [data-testid="stCaptionContainer"] p {
    color: #636b78 !important;
    font-size: 0.76rem !important;
}

/* ── Headings ── */
h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px !important;
    line-height: 1.2 !important;
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
}
h3 {
    font-size: 1.0rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.2px !important;
}

/* ── Radio buttons ── */
[data-testid="stRadio"] label {
    font-size: 0.85rem !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label {
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    color: #a0a8b8 !important;
}

/* ── Checkbox labels in sidebar ── */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    font-size: 0.84rem !important;
    color: #d0d6e0 !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label:hover {
    color: #f0f2f5 !important;
}

/* ── Status / spinner boxes ── */
[data-testid="stStatusWidget"] {
    border-radius: 6px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid #252836 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 0.45rem 0.9rem !important;
}
</style>
"""


_LIGHT_CSS = """
<style>
/* ── Light mode overrides (applied on top of base CSS) ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.main .block-container {
    background-color: #f5f7fa !important;
    color: #1a1e2a !important;
}

[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #dde2eb !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #4a5568 !important;
    border-bottom: 1px solid #dde2eb !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
    color: #2d3748 !important;
}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label:hover {
    color: #1a202c !important;
}

[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 1px solid #dde2eb !important;
}
[data-testid="stMetricLabel"] > div { color: #718096 !important; }
[data-testid="stMetricValue"]       { color: #1a202c !important; }

hr {
    border-top: 1px solid #dde2eb !important;
}

.stCaption p, [data-testid="stCaptionContainer"] p {
    color: #718096 !important;
}

h1, h2, h3 { color: #1a202c !important; }

.stMarkdown p, .stMarkdown li, .stMarkdown { color: #2d3748 !important; }

[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stMultiSelect"] label {
    color: #4a5568 !important;
}

.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid #dde2eb !important;
}

.stDataFrame { background-color: #ffffff !important; }

[data-testid="stAlert"] { background-color: #ebf8ff !important; }
</style>
"""


def inject_css() -> None:
    """Inject global CSS into the current page, respecting the dark_mode toggle."""
    st.markdown(_CSS, unsafe_allow_html=True)
    if not st.session_state.get("dark_mode", True):
        st.markdown(_LIGHT_CSS, unsafe_allow_html=True)


def apply_theme() -> None:
    """Preferred alias for inject_css() — call once at the top of every page."""
    inject_css()


def apply_theme_to_plotly_figure(fig: object) -> object:
    """Apply the app's dark/light Plotly theme to a figure in-place.

    Parameters
    ----------
    fig : go.Figure
        The figure to style.

    Returns
    -------
    The same figure, for chaining.
    """
    fig.update_layout(**CHART_LAYOUT)  # type: ignore[union-attr]
    fig.update_xaxes(**AXIS_STYLE)     # type: ignore[union-attr]
    fig.update_yaxes(**AXIS_STYLE)     # type: ignore[union-attr]
    return fig
