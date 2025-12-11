import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv
import os

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

load_dotenv()  # Fallback for local development

# Internal Imports
from alphacouncil.tools.vol_tools import VolSenseService
from alphacouncil.persistence import get_daily_cache
from volsense_inference.sector_mapping import get_sector_map

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Technician's Console",
    page_icon="📐",
    layout="wide"
)

# Custom CSS (Terminal Vibe - matching Risk Vault)
st.markdown("""
<style>
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stDataFrame { border: 1px solid #333; border-radius: 5px; }
    .filter-section {
        background-color: #1a1c24;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# 2. INITIALIZE SERVICES
vol_service = VolSenseService.get_instance()
cache = get_daily_cache()
sector_map = get_sector_map("v507")

# 3. STALE CACHE CHECK (runs on every page load, BEFORE cached function)
def _check_and_refresh_cache():
    """Check if cache is stale and trigger refresh if needed. Clears Streamlit cache if stale."""
    is_stale = hasattr(cache, 'is_stale') and cache.is_stale()
    if not cache._cache or is_stale:
        # Clear Streamlit's function cache to force reload
        st.cache_data.clear()
        with st.spinner("🌊 Hydrating Market Data (Cache stale or empty)..."):
            vol_service.hydrate_market()
        return True
    return False

# Run stale check on every page load
_check_and_refresh_cache()

# 4. DATA LOADING FUNCTION (now only builds DataFrame, no stale check)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_market_data():
    """Load and prepare universe-wide data from cache."""
    
    # Build DataFrame from cache
    rows = []
    for ticker, payload in cache._cache.items():
        if "error" in payload:
            continue
        
        signal_block = payload.get("signal", {}) or {}
        metrics = payload.get("metrics", {}) or {}
        context = payload.get("context", {}) or {}
        
        rows.append({
            "ticker": ticker,
            "sector": payload.get("sector", "Unknown"),
            "signal": signal_block.get("position", "NEUTRAL"),
            "action": signal_block.get("action", "Wait"),
            "strength": signal_block.get("strength", 0.0),
            "z_score": metrics.get("z_score", 0.0),
            "z_score_1d": metrics.get("z_score_1d", 0.0),
            "z_score_5d": metrics.get("z_score_5d", 0.0),
            "z_score_10d": metrics.get("z_score_10d", 0.0),
            "current_vol": metrics.get("current_vol", 0.0),
            "forecast_1d": metrics.get("forecast_1d", 0.0),
            "forecast_5d": metrics.get("forecast_5d", 0.0),
            "forecast_10d": metrics.get("forecast_10d", 0.0),
            "vol_spread_pct": metrics.get("vol_spread_pct", 0.0),
            "term_spread_10v5": metrics.get("term_spread_10v5", 0.0),
            "momentum_5d": metrics.get("momentum_5d", 0.0),
            "momentum_20d": metrics.get("momentum_20d", 0.0),
            "regime": context.get("regime", "Normal"),
            "sector_z_score": context.get("sector_z_score", 0.0),
            "rank_in_sector": context.get("rank_in_sector", 0.5),
        })
    
    return pd.DataFrame(rows)

# 4. SIDEBAR CONTROLS
with st.sidebar:
    st.header("📐 Console Controls")
    
    if st.button("🔄 Hydrate Market", type="primary"):
        with st.spinner("Refreshing VolSense Engine..."):
            vol_service.hydrate_market()
            st.cache_data.clear()  # Clear cache to reload data
            st.success("✅ Market Data Refreshed!")
            st.rerun()
    
    st.divider()
    st.subheader("🎯 Filters")
    
    # Load data for filter options
    df_full = load_market_data()
    
    # Signal Type Filter
    signal_options = ["ALL"] + sorted(df_full["signal"].unique().tolist())
    selected_signal = st.selectbox("Signal Type", signal_options, index=0)
    
    # Sector Filter (Multi-select) - Don't use default parameter, show all by default
    sector_options = sorted(df_full["sector"].unique().tolist())
    selected_sectors = st.multiselect("Sectors", sector_options)
    
    # Z-Score Threshold (default to minimum to show all tickers)
    z_threshold = st.slider("Min Z-Score",  -5.0, 5.0, -5.0, 0.5)
    
    # Horizon Toggle for Heatmap
    st.divider()
    st.subheader("📅 Heatmap Horizon")
    horizon_option = st.radio(
        "Forecast Horizon",
        ["1-Day", "5-Day", "10-Day"],
        index=1,  # Default to 5-Day
        horizontal=True
    )
    horizon_col_map = {"1-Day": "z_score_1d", "5-Day": "z_score_5d", "10-Day": "z_score_10d"}
    selected_z_col = horizon_col_map[horizon_option]
    
    st.divider()
    st.caption(f"📊 Last Cache Update: {cache._today_str()}")

# 5. APPLY FILTERS
df = df_full.copy()

if selected_signal != "ALL":
    df = df[df["signal"] == selected_signal]

# Only filter by sectors if user explicitly selected some
if selected_sectors and len(selected_sectors) > 0:
    df = df[df["sector"].isin(selected_sectors)]

df = df[df[selected_z_col] >= z_threshold]

# 6. MAIN UI HEADER
st.title("📐 The Technician's Console")
st.caption(f"Universe-wide VolSense Analysis • {len(df)} / {len(df_full)} Tickers • Updated {datetime.now().strftime('%H:%M:%S')}")

# 7. ROW 1: KPI STRIP
st.markdown("### 📊 Market Overview")
c1, c2, c3, c4 = st.columns(4)

total_tickers = len(df)
strong_buys = len(df[(df["signal"] == "BUY") & (df[selected_z_col] > 2.0)])
regime_counts = df["regime"].value_counts()
avg_z_score = df[selected_z_col].mean() if len(df) > 0 else 0.0

c1.metric("Tickers Analyzed", total_tickers)
c2.metric("Strong Buy Signals", strong_buys, help="BUY signal with Z-Score > 2.0")
c3.metric("Top Regime", regime_counts.index[0] if len(regime_counts) > 0 else "N/A", 
          delta=f"{regime_counts.iloc[0]} tickers" if len(regime_counts) > 0 else "")
c4.metric("Avg Z-Score", f"{avg_z_score:.2f}")

st.divider()

# 8. ROW 2: VOLATILITY HEATMAP (Treemap)
st.markdown(f"### 🗺️ Volatility Universe Heatmap ({horizon_option} Forecast)")

if len(df) > 0:
    # Prepare data for treemap
    df_tree = df.copy()
    df_tree["abs_strength"] = df_tree["strength"].abs()  # Size by absolute strength
    df_tree["hover_text"] = (
        df_tree["ticker"] + "<br>" +
        "Signal: " + df_tree["signal"] + "<br>" +
        "Z-Score (" + horizon_option + "): " + df_tree[selected_z_col].round(2).astype(str) + "<br>" +
        "Regime: " + df_tree["regime"] + "<br>" +
        "Sector: " + df_tree["sector"]
    )
    
    fig_tree = px.treemap(
        df_tree,
        path=['sector', 'ticker'],
        values='abs_strength',
        color=selected_z_col,
        color_continuous_scale='RdYlGn',  # Red-Yellow-Green
        color_continuous_midpoint=0,
        hover_data=['signal', 'regime'],
        height=500
    )
    
    fig_tree.update_layout(
        margin=dict(t=10, l=0, r=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=10)
    )
    
    fig_tree.update_traces(
        hovertemplate='<b>%{label}</b><br>Z-Score: %{color:.2f}<br>Signal: %{customdata[0]}<br>Regime: %{customdata[1]}<extra></extra>',
        textposition='middle center'
    )
    
    st.plotly_chart(fig_tree, width='stretch')
else:
    st.info("No tickers match the current filter criteria.")

st.divider()

# 9. ROW 3: TOP SETUPS TABLE & REGIME DISTRIBUTION
col_table, col_regime = st.columns([3, 2])

with col_table:
    st.markdown("### 🎯 Top Ranked Setups")
    
    if len(df) > 0:
        # Sort by signal strength and take top 20
        df_top = df.nlargest(20, "strength").copy()
        
        # Format for display
        df_display = df_top[[
            "ticker", "sector", "signal", "z_score", 
            "vol_spread_pct", "term_spread_10v5", "momentum_5d"
        ]].copy()
        
        df_display.columns = [
            "Ticker", "Sector", "Signal", "Z-Score", 
            "Vol Spread %", "Term Spread", "Mom 5d"
        ]
        
        # Format numeric columns
        df_display["Z-Score"] = df_display["Z-Score"].round(2)
        df_display["Vol Spread %"] = (df_display["Vol Spread %"] * 100).round(2).astype(str) + "%"
        df_display["Term Spread"] = df_display["Term Spread"].round(4)
        df_display["Mom 5d"] = (df_display["Mom 5d"] * 100).round(2).astype(str) + "%"
        
        st.dataframe(df_display, hide_index=True, width='stretch')
    else:
        st.info("No setups available with current filters.")

with col_regime:
    st.markdown("### 🌡️ Regime Distribution")
    
    if len(df) > 0:
        regime_dist = df["regime"].value_counts().reset_index()
        regime_dist.columns = ["Regime", "Count"]
        
        fig_regime = px.pie(
            regime_dist,
            names="Regime",
            values="Count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=400
        )
        
        fig_regime.update_layout(
            margin=dict(t=0, l=0, r=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation="v", x=1.1, y=0.5)
        )
        
        st.plotly_chart(fig_regime, width='stretch')
    else:
        st.info("No regime data available.")

st.divider()

# 10. ROW 4: SECTOR SIGNAL STRENGTH
st.markdown("### 📈 Sector Signal Strength")

if len(df) > 0:
    sector_strength = df.groupby("sector")["strength"].mean().reset_index()
    sector_strength = sector_strength.sort_values("strength", ascending=False)
    
    # Color code by strength (positive = green, negative = red)
    colors = ['#00FF00' if x > 0 else '#FF4444' for x in sector_strength["strength"]]
    
    fig_sector = go.Figure(data=[
        go.Bar(
            x=sector_strength["sector"],
            y=sector_strength["strength"],
            marker_color=colors,
            text=sector_strength["strength"].round(3),
            textposition='outside'
        )
    ])
    
    fig_sector.update_layout(
        height=400,
        margin=dict(t=20, l=0, r=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Sector",
        yaxis_title="Avg Signal Strength",
        showlegend=False,
        xaxis=dict(tickangle=-45)
    )
    
    st.plotly_chart(fig_sector, width='stretch')
else:
    st.info("No sector data available with current filters.")

# 11. FOOTER: DEBUG INFO
with st.expander("🛠️ Raw Data Preview"):
    st.dataframe(df.head(10), width='stretch')
    st.caption(f"Showing 10 of {len(df)} filtered tickers")