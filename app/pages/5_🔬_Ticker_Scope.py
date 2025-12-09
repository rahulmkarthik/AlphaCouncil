import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# STREAMLIT CLOUD
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
if "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

load_dotenv()  # Fallback for local development
# --- Internal Imports ---
from alphacouncil.graph import app as graph_app
from alphacouncil.tools.vol_tools import VolSenseService
from volsense_inference.sector_mapping import get_color
from volsense_inference.sector_mapping import get_sector_map

# Load the allowed universe keys for validation
VALID_UNIVERSE = set(get_sector_map("v507").keys())

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Ticker Scope",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .agent-card {
        background-color: #1a1c24;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .agent-header {
        font-size: 1.2rem;
        font-weight: bold;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .metric-row { display: flex; gap: 15px; margin-top: 10px; font-size: 0.9rem; }
    .metric-box { background: #2d2f3a; padding: 5px 10px; border-radius: 5px; }
    
    /* Signal Colors */
    .sig-BUY { color: #00FF00; border-left: 5px solid #00FF00; }
    .sig-SELL { color: #FF4444; border-left: 5px solid #FF4444; }
    .sig-WAIT { color: #FFFF00; border-left: 5px solid #FFFF00; }
    .sig-HEDGE { color: #FFA500; border-left: 5px solid #FFA500; }
    
    /* Risk Levels */
    .risk-HIGH { color: #FF4444; }
    .risk-MEDIUM { color: #FFA500; }
    .risk-LOW { color: #00FF00; }
</style>
""", unsafe_allow_html=True)

# 2. HELPER FUNCTIONS
def render_chart_from_json(ticker, plot_data):
    if not plot_data or "history" not in plot_data:
        st.warning("No historical data available for plotting.")
        return

    history = plot_data["history"]
    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    
    forecasts = plot_data.get("forecasts", {})

    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot Realized Vol
    ax.plot(df["date"], df["realized_vol"], label="Realized Vol", color="#4da6ff", linewidth=2)
    
    # Plot Forecast Lines
    colors = {"1": "#ffcc00", "5": "#ff6666", "10": "#cc99ff"}
    for horizon, val in forecasts.items():
        if val is not None:
            ax.axhline(y=val, color=colors.get(horizon, "white"), linestyle="--", alpha=0.8, label=f"{horizon}d Forecast")

    ax.set_title(f"{ticker} — Volatility Term Structure", color="white")
    ax.set_facecolor("#1a1c24")
    fig.patch.set_facecolor("#1a1c24")
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.spines['bottom'].set_color('#444')
    ax.spines['left'].set_color('#444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.legend(facecolor="#2d2f3a", labelcolor="white", framealpha=1)
    st.pyplot(fig, width='stretch')


def render_volsense_stats(ticker, raw_data):
    if not raw_data or "metrics" not in raw_data:
        st.warning("No quantitative data returned.")
        return

    st.subheader("🔢 Quantitative Metrics")
    m = raw_data["metrics"]
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    
    metrics = [
        ("Forecast (1d)", m.get('forecast_1d')),
        ("Forecast (5d)", m.get('forecast_5d')),
        ("Forecast (10d)", m.get('forecast_10d')),
        ("Today Vol", m.get('current_vol')),
        ("Vol Spread", m.get('vol_spread_pct')),
        ("Z-Score", m.get('z_score')),
        ("Term Spread", m.get('term_spread_10v5'))
    ]
    
    for i, (label, val) in enumerate(metrics):
        if val is not None:
            fmt = "{:.2%}" if "Spread" in label or "Vol" in label else "{:.4f}"
            if "Z-Score" in label: fmt = "{:.2f}"
            
            # Use delta_color="inverse" for Z-score if needed, here keeping simple
            c_obj = [c1, c2, c3, c4, c5, c6, c7][i]
            c_obj.metric(label, fmt.format(val))

    st.markdown("### 📉 Volatility Chart")
    plot_data = raw_data.get("plot_data", {})
    if plot_data:
        render_chart_from_json(ticker, plot_data)
    else:
        st.info("Chart data not found in cache.")

# 3. SIDEBAR CONTROLS
with st.sidebar:
    st.header("🔬 Scope Controls")
    ticker_input = st.text_input("Target Ticker", value="NVDA").upper()
    
    if st.button("📡 Analyze Ticker", type="primary"):
        if ticker_input not in VALID_UNIVERSE:
            st.error(f"⛔ {ticker_input} is not in the v507 Universe.")
        else:
            with st.spinner(f"Running Deep Dive on {ticker_input}..."):
                try:
                    response = graph_app.invoke({"ticker": ticker_input})
                    st.session_state["scope_result"] = response
                    st.session_state["scope_ticker"] = ticker_input
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")
    
    st.divider()
    
    if st.button("🔄 Hydrate Market Data"):
        service = VolSenseService.get_instance()
        with st.spinner("Refreshing VolSense Engine..."):
            try:
                service._ensure_loaded()
                st.success("Engine Hydrated!")
            except Exception as e:
                st.error(f"Refresh failed: {e}")

# 4. MAIN LAYOUT
st.title("🔬 The Ticker Scope")

if "scope_result" in st.session_state:
    res = st.session_state["scope_result"]
    ticker = st.session_state["scope_ticker"]
    
    # Extract Models
    tech = res.get("technical_signal")
    fund = res.get("fundamental_signal")

    st.markdown(f"**Target:** {ticker}")
    
    # --- ROW 1: AGENTS ---
    col1, col2 = st.columns(2)
    
    # TECHNICIAN CARD (FIXED: Added Drivers)
    with col1:
        if tech:
            sig_cls = f"sig-{tech.signal}"
            # Join drivers list into string
            drivers_str = ', '.join(tech.key_drivers)
            st.markdown(f"""
            <div class='agent-card {sig_cls}'>
                <div class='agent-header'>
                    <span>📐 The Technician</span>
                    <span style='margin-left:auto'>{tech.signal}</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-box'>Conf: {tech.confidence:.0%}</span>
                    <span class='metric-box'>Regime: {tech.regime}</span>
                </div>
                <hr style='border-color: #444; margin: 15px 0;'>
                <p><i>"{tech.reasoning}"</i></p>
                <p style='font-size: 0.85rem; color: #aaa; margin-top: 10px;'>
                    <b>Drivers:</b> {drivers_str}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Technician is silent.")

    # FUNDAMENTALIST CARD (FIXED: Added Events)
    with col2:
        if fund:
            risk_cls = f"risk-{fund.risk_level}"
            # Join events list into string
            events_str = ', '.join(fund.major_events[:3])
            st.markdown(f"""
            <div class='agent-card'>
                <div class='agent-header'>
                    <span>📰 The Fundamentalist</span>
                    <span style='margin-left:auto' class='{risk_cls}'>Risk: {fund.risk_level}</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-box'>Sector: {fund.sector}</span>
                    <span class='metric-box'>Sent: {fund.sentiment_score:.2f}</span>
                </div>
                <hr style='border-color: #444; margin: 15px 0;'>
                <p><i>"{fund.relevance_to_ticker}"</i></p>
                <p style='font-size: 0.85rem; color: #aaa; margin-top: 10px;'>
                    <b>Events:</b> {events_str}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Fundamentalist is silent.")

    # --- ROW 2: DATA & CHARTS ---
    st.markdown("---")
    
    raw_vol_data = res.get("raw_vol_data", {})
    render_volsense_stats(ticker, raw_vol_data)
    
    # --- ROW 3: DEBUG ---
    with st.expander("🛠️ View Raw Agent JSON"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Technician**")
            st.json(tech.model_dump() if tech else {})
        with c2:
            st.markdown("**Fundamentalist**")
            st.json(fund.model_dump() if fund else {})

else:
    st.info("👈 Enter a ticker in the sidebar to begin analysis.")