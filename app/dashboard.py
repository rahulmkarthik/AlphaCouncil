import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load env vars for API keys
load_dotenv()

# --- Internal Imports ---
from alphacouncil.graph import app as graph_app
from alphacouncil.tools.vol_tools import VolSenseService
from volsense_inference.signal_engine import SignalEngine
from volsense_inference.sector_mapping import get_color
from volsense_inference.sector_mapping import get_sector_map

# Load the allowed universe keys for validation
VALID_UNIVERSE = set(get_sector_map("v507").keys())

# ─────────────────────────────────────────────────────────────
# 1. PAGE CONFIG & CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaCouncil Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Dark Mode" Agent Cards
st.markdown("""
<style>
    /* Global Text Color Fix */
    body { color: #E0E0E0; }
    
    /* Agent Card Styling */
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
        margin-bottom: 10px;
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

# ─────────────────────────────────────────────────────────────
# 2. HELPER FUNCTIONS (Adapted from dashboard.py)
# ─────────────────────────────────────────────────────────────
def _pretty_number(x):
    if pd.isna(x): return "-"
    return f"{x:.4f}"

def render_chart_from_json(ticker, plot_data):
    """
    Renders the Volatility Chart using cached JSON data.
    """
    if not plot_data or "history" not in plot_data:
        st.warning("No historical data available for plotting.")
        return

    # 1. Reconstruct DataFrame
    history = plot_data["history"]
    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    
    forecasts = plot_data.get("forecasts", {})

    # 2. Create Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot Realized Vol
    # Use a dark theme compatible style
    ax.plot(df["date"], df["realized_vol"], label="Realized Vol", color="#4da6ff", linewidth=2)
    
    # Plot Forecast Lines
    colors = {"1": "#ffcc00", "5": "#ff6666", "10": "#cc99ff"}
    for horizon, val in forecasts.items():
        if val is not None:
            ax.axhline(y=val, color=colors.get(horizon, "white"), linestyle="--", alpha=0.8, label=f"{horizon}d Forecast")

    # Styling
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
    """
    Main entry point for quantitative visualization.
    """
    if not raw_data or "metrics" not in raw_data:
        st.warning("No quantitative data returned from the Council.")
        return

    # 1. Display Metrics Grid
    st.subheader("🔢 Quantitative Metrics")
    m = raw_data["metrics"]
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Forecast (1d)", f"{m['forecast_1d']:.4f}")
    c2.metric("Forecast (5d)", f"{m['forecast_5d']:.4f}")
    c3.metric("Forecast (10d)", f"{m['forecast_10d']:.4f}")
    c4.metric("Today Vol", f"{m['current_vol']:.4f}")
    c5.metric("Vol Spread", f"{m['vol_spread_pct']:.2%}")
    c6.metric("Z-Score", f"{m['z_score']:.2f}", delta_color="inverse")
    c7.metric("Term Spread", f"{m['term_spread_10v5']:.2%}")

    # 2. Render Chart (From Cache)
    st.markdown("### 📉 Volatility Chart")
    plot_data = raw_data.get("plot_data", {})
    if plot_data:
        render_chart_from_json(ticker, plot_data)
    else:
        st.info("Chart data not found in cache. Run fresh analysis to generate.")

# ─────────────────────────────────────────────────────────────
# 3. SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("AlphaCouncil")
    st.caption("v1.0 • Gemini 3.0 • VolSense")
    
    # Input
    ticker_input = st.text_input("Ticker Symbol", value="NVDA", help="Enter a US Equity Ticker").upper()
    
    # Validation Status
    is_valid = ticker_input in VALID_UNIVERSE
    
    if st.button("Convene Council 🔔", type="primary", use_container_width=True):
        if not is_valid:
            st.error(f"⛔ {ticker_input} is not in the v507 Universe.")
            st.warning("Please enter a supported ticker (e.g., NVDA, SPY, BTC-USD).")
        else:
            # ONLY RUN IF VALID
            with st.spinner(f"📡 Summoning Agents for {ticker_input}..."):
                try:
                    # INVOKE THE GRAPH
                    response = graph_app.invoke({"ticker": ticker_input})
                    st.session_state["council_result"] = response
                    st.session_state["active_ticker"] = ticker_input
                except Exception as e:
                    st.error(f"Council Adjourned unexpectedly: {e}")

    st.markdown("---")
    st.markdown("**System Status**")
    
    service = VolSenseService.get_instance()
    
    # NEW: Manual Refresh Button
    if st.button("🔄 Hydrate Market (v507)", type="secondary"):
        with st.spinner("Running Batch Inference on 507 Tickers (This takes ~2 mins)"):
            try:
                service.hydrate_market()
                st.success("Daily Log Updated!")
            except Exception as e:
                st.error(f"Hydration failed: {e}")
    
    # 2. ALSO Check Session State (The "Memory" of the app)
    # If we have a result in session state, the engine MUST be live.
    is_engine_ready = (service._forecast_engine is not None) or ("council_result" in st.session_state)
    
    if is_engine_ready:
        st.success("🟢 VolSense Engine: Online")
    else:
        st.warning("⚪ VolSense Engine: Standby")

# ─────────────────────────────────────────────────────────────
# 4. MAIN LAYOUT
# ─────────────────────────────────────────────────────────────

if "council_result" in st.session_state:
    res = st.session_state["council_result"]
    ticker = st.session_state["active_ticker"]
    
    # Extract Pydantic Models
    tech = res.get("technical_signal")
    fund = res.get("fundamental_signal")

    st.header(f"🏛️ Council Verdict: {ticker}")
    
    # --- ROW 1: AGENTS ---
    col1, col2 = st.columns(2)
    
    # TECHNICIAN CARD
    with col1:
        if tech:
            sig_cls = f"sig-{tech.signal}"
            st.markdown(f"""
            <div class='agent-card {sig_cls}'>
                <div class='agent-header'>
                    <span>📐 The Technician</span>
                    <span style='margin-left:auto'>{tech.signal}</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-box'>Confidence: {tech.confidence:.0%}</span>
                    <span class='metric-box'>Regime: {tech.regime}</span>
                </div>
                <hr style='border-color: #444; margin: 15px 0;'>
                <p><i>"{tech.reasoning}"</i></p>
                <p style='font-size: 0.8rem; color: #aaa; margin-top: 10px;'>
                    <b>Drivers:</b> {', '.join(tech.key_drivers)}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Technician is silent.")

    # FUNDAMENTALIST CARD
    with col2:
        if fund:
            risk_cls = f"risk-{fund.risk_level}"
            st.markdown(f"""
            <div class='agent-card'>
                <div class='agent-header'>
                    <span>📰 The Fundamentalist</span>
                    <span style='margin-left:auto; font-size:1rem' class='{risk_cls}'>Risk: {fund.risk_level}</span>
                </div>
                <div class='metric-row'>
                    <span class='metric-box'>Sector: {fund.sector}</span>
                    <span class='metric-box'>Sentiment: {fund.sentiment_score:+.2f}</span>
                </div>
                <hr style='border-color: #444; margin: 15px 0;'>
                <p><i>"{fund.relevance_to_ticker}"</i></p>
                <p style='font-size: 0.8rem; color: #aaa; margin-top: 10px;'>
                    <b>Events:</b> {', '.join(fund.major_events[:3])}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Fundamentalist is silent.")

    # --- ROW 2: DATA & CHARTS ---
    st.markdown("---")
    st.subheader(f"📊 Quantitative Data: {ticker}")
    
    # Extract raw data from the graph result
    raw_vol_data = res.get("raw_vol_data", {})
    
    # INCORRECT: render_volsense_stats(ticker, service) 
    
    # CORRECT: Pass the data dictionary
    render_volsense_stats(ticker, raw_vol_data)
    
    # --- ROW 3: DEBUG EXPANDER ---
    with st.expander("🛠️ View Raw JSON Payload"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Technician JSON**")
            st.json(tech.model_dump() if tech else {})
        with c2:
            st.markdown("**Fundamentalist JSON**")
            st.json(fund.model_dump() if fund else {})

else:
    # LANDING STATE
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h1>Welcome to AlphaCouncil</h1>
        <p style='font-size: 1.2rem; color: #888;'>
            Enter a ticker in the sidebar to summon the Investment Committee.<br>
            The Council will analyze Volatility (VolSense) and Macro Risks (Tavily) in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)