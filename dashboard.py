import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DB_PATH           = Path("mask_detection.db")
VIOLATIONS_DIR    = Path("violations")
REFRESH_MS        = 3000


# ──────────────────────────────────────────────────────────────────────────────
# PAGE SETUP
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaskGuard · Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if AUTO_REFRESH_AVAILABLE:
    st_autorefresh(interval=REFRESH_MS, key="auto_refresh")


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  — Bright, vibrant, modern design
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4ff 0%, #fafbff 40%, #f5f0ff 100%);
    color: #1a1a2e;
    font-family: 'Outfit', sans-serif;
}

/* Animated gradient mesh background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(236,72,153,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(34,197,94,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 20% 80%, rgba(245,158,11,0.06) 0%, transparent 50%);
    z-index: 0;
    pointer-events: none;
}

[data-testid="stMain"] { position: relative; z-index: 1; }
[data-testid="stMainBlockContainer"] { padding-top: 20px !important; padding-bottom: 20px !important; }
[data-testid="stSidebarContent"] { position: relative; z-index: 1; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8ecf0 !important;
    box-shadow: 4px 0 20px rgba(0,0,0,0.06) !important;
}
[data-testid="stSidebar"] * { color: #2d3748 !important; }
[data-testid="stSidebar"] hr { border-color: #e8ecf0 !important; margin: 10px 0 !important; }

/* Remove Streamlit's default top padding in sidebar */
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebarContent"] { padding: 12px 14px !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

/* Hide default radio widget entirely */
[data-testid="stSidebar"] [data-testid="stRadio"] { display: none !important; }

/* Sidebar brand */
.sidebar-brand {
    padding: 4px 2px 12px;
    border-bottom: 1px solid #e8ecf0;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sidebar-brand-icon {
    font-size: 26px;
    line-height: 1;
    flex-shrink: 0;
}
.sidebar-brand-text {
    font-size: 17px;
    font-weight: 900;
    color: #1a1a2e !important;
    letter-spacing: -0.3px;
    line-height: 1.2;
}

/* ── Nav Items ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 12px;
    border-radius: 12px;
    margin-bottom: 2px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 600;
    color: #4a5568;
    transition: all 0.15s ease;
    border: none;
    background: transparent;
    width: 100%;
}
.nav-item:hover {
    background: #f1f3f9;
    color: #1a1a2e;
    transform: translateX(2px);
}
.nav-item.active {
    background: linear-gradient(135deg, #f43f5e, #fb7185) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(244,63,94,0.3);
}
.nav-icon {
    font-size: 17px;
    width: 22px;
    text-align: center;
    flex-shrink: 0;
}
.nav-label { flex: 1; }

/* Sidebar section label */
.sidebar-section-label {
    font-size: 10px;
    font-weight: 700;
    color: #a0aec0 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 8px 4px 4px;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 20px !important;
    padding: 20px 24px !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.08), 0 1px 3px rgba(0,0,0,0.04) !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #ec4899, #f59e0b);
    border-radius: 20px 20px 0 0;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(99,102,241,0.15), 0 4px 8px rgba(0,0,0,0.06) !important;
}
[data-testid="stMetricLabel"]  { color: #64748b !important; font-size: 12px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricValue"]  { color: #1a1a2e !important; font-size: 30px !important; font-weight: 800 !important; font-family: 'Outfit', sans-serif !important; }
[data-testid="stMetricDelta"]  { font-size: 12px !important; font-weight: 600 !important; }

/* ── Page Title ── */
.page-title {
    font-size: 32px;
    font-weight: 900;
    background: linear-gradient(135deg, #1a1a2e 0%, #6366f1 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 4px;
}
.page-subtitle {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 400;
    letter-spacing: 0.02em;
    margin-bottom: 24px;
}

/* ── Section Headers ── */
.section-header {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 28px 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.2), transparent);
}

/* ── Chart Cards ── */
.chart-card {
    background: #ffffff;
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.06);
    margin-bottom: 16px;
}
.chart-title {
    font-size: 13px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
}

/* ── Badges ── */
.badge-mask {
    background: linear-gradient(135deg, #dcfce7, #bbf7d0);
    color: #15803d;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid #86efac;
    letter-spacing: 0.03em;
}
.badge-nomask {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    color: #dc2626;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid #fca5a5;
    letter-spacing: 0.03em;
}
.badge-count {
    background: linear-gradient(135deg, #ede9fe, #ddd6fe);
    color: #7c3aed;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 700;
    border: 1px solid #c4b5fd;
}

/* ── Alert Banner ── */
.alert-banner {
    background: linear-gradient(135deg, #fef2f2, #fff5f5);
    border: 2px solid #fca5a5;
    border-left: 6px solid #ef4444;
    border-radius: 16px;
    padding: 16px 20px;
    color: #dc2626;
    font-weight: 700;
    font-size: 14px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 4px 20px rgba(239,68,68,0.15);
    animation: alertPulse 2s ease-in-out infinite;
}
@keyframes alertPulse {
    0%, 100% { box-shadow: 0 4px 20px rgba(239,68,68,0.15); }
    50% { box-shadow: 0 4px 32px rgba(239,68,68,0.35); }
}

/* ── Recent Detection Cards ── */
.detection-card {
    background: #ffffff;
    border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px;
    padding: 14px 12px;
    text-align: center;
    transition: all 0.25s ease;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.detection-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.12);
}
.detection-name {
    font-weight: 700;
    font-size: 13px;
    color: #1a1a2e;
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.detection-time {
    font-size: 10px;
    color: #94a3b8;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 5px;
}
.detection-conf {
    font-size: 10px;
    color: #cbd5e1;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid rgba(99,102,241,0.1) !important; margin: 24px 0 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 16px !important; overflow: hidden; }
[data-testid="stDataFrame"] * { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    padding: 8px 20px !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.4) !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9) !important;
    color: #475569 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* ── Selectbox, radio ── */
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: rgba(99,102,241,0.2) !important;
    background: #ffffff !important;
}
[data-testid="stRadio"] label {
    background: rgba(99,102,241,0.05);
    border-radius: 8px;
    padding: 4px 10px !important;
    margin: 2px 0 !important;
    transition: background 0.15s;
}
[data-testid="stRadio"] label:hover { background: rgba(99,102,241,0.12); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 10px; }

/* ── Info/success/warning boxes ── */
[data-testid="stAlert"] { border-radius: 16px !important; }

/* Status badge for live indicator */
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(34,197,94,0.6);
    animation: livePulse 1.5s ease-in-out infinite;
    margin-right: 6px;
}
@keyframes livePulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(34,197,94,0.6); }
    50% { opacity: 0.6; box-shadow: 0 0 14px rgba(34,197,94,0.9); }
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_conn():
    return sqlite3.connect(str(DB_PATH), check_same_thread=False)


def load_detections(conn, hours: int = 24) -> pd.DataFrame:
    since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    query = f"""
        SELECT id, person_name, mask_status,
               ROUND(confidence * 100, 1) AS confidence_pct,
               image_path, timestamp
        FROM detections
        WHERE timestamp >= '{since}'
        ORDER BY id DESC
    """
    df = pd.read_sql_query(query, conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_sessions(conn) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT 20", conn
        )
    except Exception:
        return pd.DataFrame()


def load_all_detections(conn) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CHART THEME  — Bright, clean
# ──────────────────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,252,0.8)",
    font=dict(color="#64748b", size=12, family="Outfit"),
    margin=dict(t=20, b=20, l=20, r=20),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(99,102,241,0.15)",
        borderwidth=1,
        font=dict(color="#475569"),
    ),
    xaxis=dict(
        gridcolor="rgba(99,102,241,0.08)",
        zerolinecolor="rgba(99,102,241,0.1)",
        linecolor="rgba(99,102,241,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(99,102,241,0.08)",
        zerolinecolor="rgba(99,102,241,0.1)",
        linecolor="rgba(99,102,241,0.1)",
    ),
)

COLOR_MASK   = "#22c55e"
COLOR_NOMASK = "#f43f5e"
COLOR_INDIGO = "#6366f1"
COLOR_AMBER  = "#f59e0b"
COLOR_MAP    = {"Mask": COLOR_MASK, "No Mask": COLOR_NOMASK}


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
# ── Nav pages definition
NAV_PAGES = [
    ("📊", "Dashboard",  "📊 Dashboard"),
    ("📋", "Records",    "📋 Records"),
    ("🚨", "Violations", "🚨 Violations"),
    ("📈", "Analytics",  "📈 Analytics"),
    ("⚙️", "Settings",   "⚙️ Settings"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "📊 Dashboard"

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="sidebar-brand-icon">🛡️</span>
        <span class="sidebar-brand-text">Mask Detection</span>
    </div>
    """, unsafe_allow_html=True)

    for icon, label, key in NAV_PAGES:
        is_active = st.session_state.active_page == key
        active_cls = "active" if is_active else ""
        st.markdown(
            f'<div class="nav-item {active_cls}">'
            f'<span class="nav-icon">{icon}</span>'
            f'<span class="nav-label">{label}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.active_page = key
            st.rerun()

    st.markdown("""
    <style>
    [data-testid="stSidebar"] .stButton {
        margin-top: -46px !important;
        opacity: 0 !important;
        height: 44px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-section-label">Time Range</div>', unsafe_allow_html=True)
    time_range = st.selectbox(
        "Show data for",
        ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days", "All time"],
        index=2,
        label_visibility="collapsed",
    )

    hours_map = {
        "Last 1 hour": 1, "Last 6 hours": 6, "Last 24 hours": 24,
        "Last 7 days": 168, "All time": 99999,
    }
    selected_hours = hours_map[time_range]

    st.markdown("---")
    db_exists = DB_PATH.exists()
    if db_exists:
        st.markdown(
            f'<div style="font-size:13px;padding:4px 0">'
            f'<span class="live-dot"></span>'
            f'<span style="color:#22c55e;font-weight:700">Database connected</span></div>',
            unsafe_allow_html=True
        )
    else:
        st.error("⚠️ Database not found")

    st.markdown(
        f"<div style='font-size:10px;color:#a0aec0;margin-top:6px;"
        f"font-family:JetBrains Mono,monospace'>Updated {datetime.now().strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:10px;color:#a0aec0;text-align:center;"
        "letter-spacing:0.06em'>MASKGUARD v2.0 · INDUSTRY EDITION</div>",
        unsafe_allow_html=True
    )

page = st.session_state.active_page


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
if not DB_PATH.exists():
    st.error("⚠️  Database not found. Start `detection.py` first.")
    st.stop()

conn   = get_conn()
df     = load_detections(conn, hours=selected_hours)
df_all = load_all_detections(conn)

# ── Violation-only metrics (model only detects "No Mask") ─────────────────────
violations_df     = df[df["mask_status"] == "No Mask"].copy()    if not df.empty else pd.DataFrame()
violations_all_df = df_all[df_all["mask_status"] == "No Mask"].copy() if not df_all.empty else pd.DataFrame()

total_violations  = len(violations_df)
unique_offenders  = violations_df["person_name"].nunique() if not violations_df.empty else 0
avg_conf          = round(violations_df["confidence_pct"].mean(), 1) if not violations_df.empty else 0
today_count       = len(violations_df[violations_df["timestamp"].dt.date == datetime.now().date()]) if not violations_df.empty else 0


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":

    st.markdown('<div class="page-title">🚫 No-Mask Violation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">Monitoring violations · {time_range} · Last refreshed {datetime.now().strftime("%H:%M:%S")}</div>',
        unsafe_allow_html=True
    )

    # ── Active Alert Banner ───────────────────────────────────────────────────
    if not violations_df.empty:
        recent_v = violations_df.iloc[0]
        mins_ago = int((datetime.now() - recent_v["timestamp"]).total_seconds() / 60)
        if mins_ago < 5:
            st.markdown(
                f'<div class="alert-banner">'
                f'⚠️ ACTIVE VIOLATION — <strong>{recent_v["person_name"]}</strong> detected without mask {mins_ago}m ago'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── KPI Row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🚫 Total Violations",    total_violations)
    k2.metric("👤 Unique Offenders",    unique_offenders)
    k3.metric("📅 Violations Today",    today_count)
    k4.metric("🎯 Avg Confidence",      f"{avg_conf}%")

    st.markdown("---")

    if violations_df.empty:
        st.success("✅ No violations detected in the selected time range!")
        st.stop()

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<p class="section-header">Top Offenders</p>', unsafe_allow_html=True)
        top_off = violations_df["person_name"].value_counts().head(8).reset_index()
        top_off.columns = ["Person", "Violations"]
        fig_top = px.bar(
            top_off, x="Violations", y="Person",
            orientation="h",
            color="Violations",
            color_continuous_scale=["#fecaca", "#f43f5e", "#be123c"],
            text_auto=True,
        )
        fig_top.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("legend",)},
            height=260,
            showlegend=False,
            xaxis_title="Violations",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        fig_top.update_traces(textfont_size=11)
        st.plotly_chart(fig_top, use_container_width=True)

    with c2:
        st.markdown('<p class="section-header">Violations Over Time</p>', unsafe_allow_html=True)
        df_time = violations_df.copy()
        freq = "10min" if selected_hours <= 6 else "H"
        df_time["bucket"] = df_time["timestamp"].dt.floor(freq)
        timeline = df_time.groupby("bucket").size().reset_index(name="violations")
        fig_line = px.area(
            timeline, x="bucket", y="violations",
            color_discrete_sequence=[COLOR_NOMASK],
        )
        fig_line.update_traces(
            opacity=0.8,
            line=dict(color=COLOR_NOMASK, width=2),
            fillcolor="rgba(244,63,94,0.12)",
        )
        fig_line.update_layout(**CHART_LAYOUT, height=260,
                               xaxis_title="", yaxis_title="Violations",
                               showlegend=False)
        st.plotly_chart(fig_line, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<p class="section-header">Violations by Hour of Day</p>', unsafe_allow_html=True)
        violations_df["hour"] = violations_df["timestamp"].dt.hour
        hourly = violations_df.groupby("hour").size().reset_index(name="count")
        fig_hourly = px.bar(
            hourly, x="hour", y="count",
            color="count",
            color_continuous_scale=["#fef2f2", "#fca5a5", "#f43f5e"],
            text_auto=True,
        )
        fig_hourly.update_layout(
            **{k: v for k, v in CHART_LAYOUT.items() if k not in ("legend",)},
            height=240,
            showlegend=False,
            xaxis_title="Hour", yaxis_title="Violations",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with c4:
        st.markdown('<p class="section-header">Confidence Distribution</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            violations_df, x="confidence_pct",
            color_discrete_sequence=[COLOR_NOMASK],
            nbins=20, opacity=0.85,
        )
        fig_hist.update_layout(**CHART_LAYOUT, height=240,
                               xaxis_title="Confidence %",
                               yaxis_title="Count", showlegend=False,
                               bargap=0.05)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Recent Violations Strip ───────────────────────────────────────────────
    st.markdown('<p class="section-header">Recent Violations</p>', unsafe_allow_html=True)
    recent = violations_df.head(8)
    cols = st.columns(len(recent)) if len(recent) > 0 else st.columns(1)
    for i, (_, row) in enumerate(recent.iterrows()):
        with cols[i]:
            st.markdown(
                f"<div class='detection-card' style='border-color:#fca5a5'>"
                f"<div class='detection-name'>{row['person_name']}</div>"
                f"<div style='margin:6px 0'><span class='badge-nomask'>❌ No Mask</span></div>"
                f"<div class='detection-time'>{row['timestamp'].strftime('%H:%M:%S')}</div>"
                f"<div class='detection-conf'>conf {row['confidence_pct']}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECORDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Records":
    st.markdown('<div class="page-title">📋 Violation Records</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">All detected no-mask events — filter, search & export</div>', unsafe_allow_html=True)

    if violations_df.empty:
        st.success("✅ No violation records in this time range.")
        st.stop()

    f1, f2, f3 = st.columns(3)
    person_filter = f1.selectbox(
        "Person", ["All"] + sorted(violations_df["person_name"].unique().tolist())
    )
    date_start = f2.date_input("From date", value=violations_df["timestamp"].min().date())
    date_end   = f3.date_input("To date",   value=violations_df["timestamp"].max().date())

    filtered = violations_df.copy()
    if person_filter != "All":
        filtered = filtered[filtered["person_name"] == person_filter]
    filtered = filtered[
        (filtered["timestamp"].dt.date >= date_start) &
        (filtered["timestamp"].dt.date <= date_end)
    ]

    st.markdown(
        f'<span class="badge-nomask">🚫 {len(filtered)} violations found</span>',
        unsafe_allow_html=True
    )
    st.markdown("")

    display = filtered[["id", "person_name", "confidence_pct", "timestamp"]].copy()
    display.columns = ["ID", "Person", "Confidence %", "Timestamp"]

    def highlight_violation(row):
        return ["background-color:#fff1f2; color:#dc2626"] * len(row)

    st.dataframe(
        display.style.apply(highlight_violation, axis=1),
        use_container_width=True,
        height=460,
    )

    e1, e2 = st.columns(2)
    csv = filtered.to_csv(index=False).encode("utf-8")
    e1.download_button(
        "⬇️ Export CSV",
        csv,
        f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        "text/csv",
    )

    summary_txt = f"""MaskGuard — No-Mask Violation Report
Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Time Range : {time_range}

Total Violations : {len(filtered)}
Unique Offenders : {filtered['person_name'].nunique()}
Avg Confidence   : {round(filtered['confidence_pct'].mean(), 1) if not filtered.empty else 0}%
"""
    e2.download_button(
        "📄 Export Report",
        summary_txt,
        f"violation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        "text/plain",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VIOLATIONS  (image gallery)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Violations":
    st.markdown('<div class="page-title">🚨 Violation Gallery</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Captured images of no-mask detections</div>', unsafe_allow_html=True)

    if violations_df.empty:
        st.success("✅ No violations recorded in this time period!")
        st.stop()

    st.markdown(
        f'<span class="badge-nomask">🚫 {len(violations_df)} violations captured</span>',
        unsafe_allow_html=True
    )
    st.markdown("")

    # Per-person violation count bar
    viol_summary = violations_df["person_name"].value_counts().reset_index()
    viol_summary.columns = ["Person", "Violations"]
    fig_viol = px.bar(
        viol_summary, x="Person", y="Violations",
        color_discrete_sequence=[COLOR_NOMASK],
        text_auto=True,
    )
    fig_viol.update_layout(**CHART_LAYOUT, height=200,
                           xaxis_title="", yaxis_title="Violations",
                           showlegend=False)
    st.plotly_chart(fig_viol, use_container_width=True)

    st.markdown("---")

    # Image grid
    n_cols = 4
    rows = [violations_df.iloc[i:i+n_cols] for i in range(0, min(len(violations_df), 24), n_cols)]
    for row_df in rows:
        cols = st.columns(n_cols)
        for j, (_, v) in enumerate(row_df.iterrows()):
            with cols[j]:
                ip = str(v.get("image_path", ""))
                if ip and os.path.exists(ip):
                    st.image(ip, use_container_width=True)
                else:
                    st.markdown(
                        "<div style='background:linear-gradient(135deg,#fff1f2,#ffe4e6);"
                        "border:2px solid #fca5a5;border-radius:12px;height:120px;"
                        "display:flex;align-items:center;justify-content:center;"
                        "color:#f43f5e;font-size:32px'>🚫</div>",
                        unsafe_allow_html=True
                    )
                st.markdown(
                    f"<div style='text-align:center;font-size:12px;color:#64748b;margin-top:6px'>"
                    f"<b style='color:#dc2626'>{v['person_name']}</b><br>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:10px'>"
                    f"{v['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</span><br>"
                    f"<span style='font-size:10px;color:#94a3b8'>conf {v['confidence_pct']}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.markdown('<div class="page-title">📈 Violation Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Patterns, trends, and repeat offender analysis</div>', unsafe_allow_html=True)

    if violations_all_df.empty:
        st.info("No violation data available for analytics.")
        st.stop()

    # ── Heatmap: hour vs weekday ──────────────────────────────────────────────
    st.markdown('<p class="section-header">Violation Heatmap — Hour × Weekday</p>', unsafe_allow_html=True)
    df_heat = violations_all_df.copy()
    df_heat["hour"]    = df_heat["timestamp"].dt.hour
    df_heat["weekday"] = df_heat["timestamp"].dt.day_name()
    pivot = df_heat.groupby(["weekday", "hour"]).size().reset_index(name="count")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot["weekday"] = pd.Categorical(pivot["weekday"], categories=order, ordered=True)
    heat_mat = pivot.sort_values("weekday").pivot(
        index="weekday", columns="hour", values="count"
    ).fillna(0)

    fig_heat = px.imshow(
        heat_mat,
        color_continuous_scale=["#fff1f2", "#fca5a5", "#f43f5e", "#be123c"],
        labels=dict(x="Hour of Day", y="", color="Violations"),
        aspect="auto",
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.8)",
        font=dict(color="#64748b", family="Outfit"),
        margin=dict(t=16, b=16, l=16, r=16),
        height=280,
        coloraxis_colorbar=dict(tickfont=dict(color="#64748b"),
                                title=dict(font=dict(color="#64748b"))),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Daily violation trend ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">Daily Violation Trend</p>', unsafe_allow_html=True)
    daily_v = violations_all_df.copy()
    daily_v["day"] = daily_v["timestamp"].dt.date
    daily_counts = daily_v.groupby("day").size().reset_index(name="violations")

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=daily_counts["day"], y=daily_counts["violations"],
        marker=dict(
            color=daily_counts["violations"],
            colorscale=[[0, "#fecaca"], [0.5, "#f43f5e"], [1, "#be123c"]],
            showscale=False,
        ),
        name="Violations",
    ))
    fig_daily.update_layout(**CHART_LAYOUT, height=240,
                            xaxis_title="", yaxis_title="Violations",
                            showlegend=False)
    st.plotly_chart(fig_daily, use_container_width=True)

    # ── Repeat offenders table ────────────────────────────────────────────────
    st.markdown('<p class="section-header">Repeat Offenders</p>', unsafe_allow_html=True)

    offenders = (
        violations_all_df.groupby("person_name")
        .agg(
            Violations=("mask_status", "count"),
            Avg_Confidence=("confidence", "mean"),
            First_Seen=("timestamp", "min"),
            Last_Seen=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"person_name": "Person", "Avg_Confidence": "Avg Conf %"})
        .sort_values("Violations", ascending=False)
    )
    offenders["Avg Conf %"] = offenders["Avg Conf %"].round(1)
    offenders["First_Seen"] = offenders["First_Seen"].dt.strftime("%Y-%m-%d %H:%M")
    offenders["Last_Seen"]  = offenders["Last_Seen"].dt.strftime("%Y-%m-%d %H:%M")

    def highlight_offender(row):
        if row["Violations"] >= 5:
            return ["background-color:#fff1f2; color:#dc2626; font-weight:700"] * len(row)
        return ["background-color:#fef9f9; color:#374151"] * len(row)

    st.dataframe(
        offenders.style.apply(highlight_offender, axis=1),
        use_container_width=True,
    )

    # ── Sessions ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Session History</p>', unsafe_allow_html=True)
    sessions = load_sessions(conn)
    if not sessions.empty:
        st.dataframe(sessions, use_container_width=True, height=200)
    else:
        st.info("No session data found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown('<div class="page-title">⚙️ Settings & Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Database management, exports, and system info</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Database Stats</p>', unsafe_allow_html=True)

    total_all_v = len(violations_all_df)
    s1, s2, s3 = st.columns(3)
    s1.metric("Total Violations (All Time)", total_all_v)
    if not violations_all_df.empty:
        s2.metric("First Violation", violations_all_df["timestamp"].min().strftime("%Y-%m-%d"))
        s3.metric("Latest Violation", violations_all_df["timestamp"].max().strftime("%Y-%m-%d"))

    st.markdown("")
    b1, b2, b3 = st.columns(3)

    if b1.button("🗑️ Delete All Records"):
        conn.execute("DELETE FROM detections")
        conn.commit()
        st.warning("All records deleted.")
        st.rerun()

    if b2.button("🔄 Reset DB (incl. Auto-Increment)"):
        conn.execute("DELETE FROM detections")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='detections'")
        conn.commit()
        st.warning("Database fully reset.")
        st.rerun()

    if b3.button("🗑️ Delete Violation Images"):
        deleted = 0
        for f in VIOLATIONS_DIR.glob("*.jpg"):
            f.unlink()
            deleted += 1
        st.warning(f"Deleted {deleted} violation image(s).")

    st.markdown("---")
    st.markdown('<p class="section-header">Export</p>', unsafe_allow_html=True)

    if not violations_all_df.empty:
        csv_all = violations_all_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export All Violations (CSV)",
            csv_all,
            f"all_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
        )

    st.markdown("---")
    st.markdown('<p class="section-header">System Info</p>', unsafe_allow_html=True)

    db_size    = round(DB_PATH.stat().st_size / 1024, 1)  if DB_PATH.exists()        else 0
    viol_count = len(list(VIOLATIONS_DIR.glob("*.jpg")))   if VIOLATIONS_DIR.exists() else 0

    i1, i2, i3 = st.columns(3)
    i1.metric("DB File Size",     f"{db_size} KB")
    i2.metric("Violation Images", viol_count)
    i3.metric("Auto-Refresh",     f"{REFRESH_MS // 1000}s interval")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#94a3b8;font-size:11px;letter-spacing:0.06em'>"
        f"MASKGUARD · NO-MASK DETECTION · © {datetime.now().year}"
        "</div>",
        unsafe_allow_html=True
    )