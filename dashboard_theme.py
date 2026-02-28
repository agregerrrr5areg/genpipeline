"""dashboard_theme.py — dark monochrome theme."""

BG      = "#000000"
PANEL   = "#111111"
ACCENT  = "#ffffff"
TEXT    = "#ffffff"
MUTED   = "#666666"
SUCCESS = "#aaaaaa"
WARNING = "#888888"

CSS = f"""
<style>
/* Background */
.stApp {{ background-color: {BG}; color: {TEXT}; }}

/* Remove panel backgrounds */
div[data-testid="stVerticalBlock"] {{ background-color: transparent; }}
section[data-testid="stSidebar"] {{ background-color: {PANEL}; }}

/* Buttons — plain text, no box */
.stButton button {{
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    color: {MUTED} !important;
    font-size: 0.85rem;
    padding: 2px 6px;
    text-align: left;
}}
.stButton button:hover {{
    color: {TEXT} !important;
    background: none !important;
}}
.stButton button:focus {{
    box-shadow: none !important;
    outline: none !important;
}}

/* Download button — keep minimal */
.stDownloadButton button {{
    background: none !important;
    border: 1px solid {MUTED} !important;
    color: {MUTED} !important;
    border-radius: 2px;
    font-size: 0.8rem;
}}
.stDownloadButton button:hover {{
    color: {TEXT} !important;
    border-color: {TEXT} !important;
}}

/* Inputs */
.stSlider, .stNumberInput, .stSelectbox {{ color: {TEXT}; }}
input, select {{ background-color: {PANEL} !important; color: {TEXT} !important; border-color: #333 !important; }}

/* Progress bar */
.stProgress > div > div {{ background-color: #444; }}

/* HR */
hr {{ border-color: #222; }}

/* Metric */
[data-testid="stMetricValue"] {{ color: {TEXT}; }}
</style>
"""

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=MUTED, family="monospace", size=11),
    margin=dict(l=0, r=0, t=30, b=0),
    scene=dict(
        bgcolor=BG,
        xaxis=dict(backgroundcolor=BG, gridcolor="#1a1a1a", color=MUTED, showticklabels=False),
        yaxis=dict(backgroundcolor=BG, gridcolor="#1a1a1a", color=MUTED, showticklabels=False),
        zaxis=dict(backgroundcolor=BG, gridcolor="#1a1a1a", color=MUTED, showticklabels=False),
    ),
)
