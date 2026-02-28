"""dashboard_theme.py — Fusion 360-style dark theme for Streamlit."""

BG       = "#1a1a2e"
PANEL    = "#16213e"
ACCENT   = "#e94560"
TEXT     = "#eaeaea"
MUTED    = "#888888"
SUCCESS  = "#00b894"
WARNING  = "#fdcb6e"

CSS = f"""
<style>
/* Page background */
.stApp {{ background-color: {BG}; color: {TEXT}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {PANEL};
    border-right: 1px solid {ACCENT}22;
}}

/* Columns / panels */
div[data-testid="stVerticalBlock"] {{
    background-color: {PANEL};
    border-radius: 6px;
}}

/* Buttons */
.stButton button {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 4px;
    font-weight: 600;
}}
.stButton button:hover {{
    background-color: {ACCENT}cc;
}}

/* Stop button (secondary) */
button[kind="secondary"] {{
    background-color: #333 !important;
    color: {TEXT} !important;
    border: 1px solid #555 !important;
}}

/* Inputs */
.stSlider, .stNumberInput, .stSelectbox {{
    color: {TEXT};
}}

/* Metric values */
[data-testid="stMetricValue"] {{
    color: {ACCENT};
    font-size: 1.4rem;
    font-weight: 700;
}}

/* Progress bar */
.stProgress > div > div {{ background-color: {ACCENT}; }}

/* Dataframe */
.stDataFrame {{ background-color: {PANEL}; }}

/* Bottom bar separator */
hr {{ border-color: {ACCENT}33; }}
</style>
"""


PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="monospace"),
    margin=dict(l=0, r=0, t=30, b=0),
    scene=dict(
        bgcolor=BG,
        xaxis=dict(backgroundcolor=PANEL, gridcolor="#333", color=TEXT),
        yaxis=dict(backgroundcolor=PANEL, gridcolor="#333", color=TEXT),
        zaxis=dict(backgroundcolor=PANEL, gridcolor="#333", color=TEXT),
    ),
)
