import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from dotenv import load_dotenv

# Internal Imports
from alphacouncil.agents.fundamentalist import fundamentalist_agent
from alphacouncil.schema import SectorIntel
from volsense_inference.sector_mapping import get_sector_map

load_dotenv()

# 1. PAGE CONFIG
st.set_page_config(
    page_title="Fundamentalist's Study",
    page_icon="📰",
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
    .news-card {
        background-color: #1a1c24;
        border-left: 4px solid #4da6ff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .risk-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 12px;
        font-weight: bold;
        margin: 10px 0;
    }
    .risk-HIGH { background-color: #ff4444; color: white; }
    .risk-MEDIUM { background-color: #ffaa00; color: black; }
    .risk-LOW { background-color: #00ff00; color: black; }
</style>
""", unsafe_allow_html=True)

# 2. SECTOR REPRESENTATIVES (for sector scan mode)
SECTOR_REPRESENTATIVES = {
    "Technology": "AAPL",
    "Healthcare": "JNJ",
    "Financials": "JPM",
    "Energy": "XOM",
    "Consumer Discretionary": "AMZN",
    "Consumer Staples": "PG",
    "Industrials": "BA",
    "Materials": "LIN",
    "Real Estate": "AMT",
    "Utilities": "NEE",
    "Commodities": "GLD",
    "Crypto / Blockchain": "COIN",
    "FX / Currency": "UUP",
    "Fixed Income": "TLT",
    "Index/ETF": "SPY",
    "Volatility / Hedge": "VXX"
}

# 3. INITIALIZE SESSION STATE
if "analysis_history" not in st.session_state:
    st.session_state["analysis_history"] = []

if "current_analysis" not in st.session_state:
    st.session_state["current_analysis"] = None

# 4. HELPER FUNCTIONS
def render_risk_badge(risk_level: str):
    """Render color-coded risk level badge."""
    st.markdown(f'<div class="risk-badge risk-{risk_level}">{risk_level} RISK</div>', unsafe_allow_html=True)

def render_sentiment_gauge(score: float):
    """Render sentiment gauge using Plotly."""
    # Convert -1.0 to 1.0 → 0 to 100
    percentage = (score + 1) * 50
    
    # Determine color
    if score < -0.3:
        color = "#ff4444"
        label = "Bearish"
    elif score > 0.3:
        color = "#00ff00"
        label = "Bullish"
    else:
        color = "#ffaa00"
        label = "Neutral"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Sentiment: {label}"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [-1, -0.3], 'color': 'rgba(255, 68, 68, 0.3)'},
                {'range': [-0.3, 0.3], 'color': 'rgba(255, 170, 0, 0.3)'},
                {'range': [0.3, 1], 'color': 'rgba(0, 255, 0, 0.3)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white"},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def analyze_query(mode: str, query: str):
    """Run fundamentalist agent analysis."""
    with st.spinner(f"🔍 Analyzing {query}..."):
        try:
            # Determine ticker to use
            if mode == "Sector Scan":
                ticker = SECTOR_REPRESENTATIVES.get(query, "AAPL")
            else:
                ticker = query.upper()
            
            # Call agent
            state = {"ticker": ticker, "messages": []}
            result = fundamentalist_agent(state)
            intel: SectorIntel = result["fundamental_signal"]
            
            # Store in session state
            analysis = {
                "mode": mode,
                "query": query,
                "ticker": ticker,
                "timestamp": datetime.now(),
                "intel": intel
            }
            
            st.session_state["current_analysis"] = analysis
            
            # Add to history (limit to 5)
            st.session_state["analysis_history"].insert(0, analysis)
            st.session_state["analysis_history"] = st.session_state["analysis_history"][:5]
            
            return True
            
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return False

# 5. SIDEBAR
with st.sidebar:
    st.header("📰 Research Controls")
    
    # Mode selector
    mode = st.radio(
        "Analysis Mode",
        ["Sector Scan", "Ticker Deep Dive"],
        help="Sector Scan: broad sector news | Ticker Deep Dive: specific ticker focus"
    )
    
    st.divider()
    
    # Query input based on mode
    if mode == "Sector Scan":
        sector_options = sorted(SECTOR_REPRESENTATIVES.keys())
        query = st.selectbox("Select Sector", sector_options)
        st.caption(f"Representative: {SECTOR_REPRESENTATIVES[query]}")
    else:
        query = st.text_input("Enter Ticker", value="NVDA").upper()
        sector_map = get_sector_map("v507")
        if query in sector_map:
            st.caption(f"Sector: {sector_map[query]}")
        else:
            st.warning("⚠️ Ticker not in v507 universe")
    
    # Analyze button
    if st.button("🔍 Analyze", type="primary", width='stretch'):
        analyze_query(mode, query)
        st.rerun()
    
    st.divider()
    
    # Analysis history
    if st.session_state["analysis_history"]:
        st.subheader("📚 Recent Analyses")
        for i, analysis in enumerate(st.session_state["analysis_history"][:5]):
            timestamp = analysis["timestamp"].strftime("%H:%M:%S")
            label = f"{analysis['query']} ({timestamp})"
            
            if st.button(label, key=f"history_{i}", width='stretch'):
                st.session_state["current_analysis"] = analysis
                st.rerun()

# 6. MAIN LAYOUT
st.title("📰 The Fundamentalist's Study")
st.caption("Sector-Wide News Analysis & Sentiment Intelligence")

# Display current analysis
if st.session_state["current_analysis"]:
    analysis = st.session_state["current_analysis"]
    intel: SectorIntel = analysis["intel"]
    
    # Header with query info
    st.markdown(f"### Analysis: **{analysis['query']}**")
    st.caption(f"Mode: {analysis['mode']} | Ticker: {analysis['ticker']} | Updated: {analysis['timestamp'].strftime('%B %d, %Y at %H:%M:%S')}")
    
    st.divider()
    
    # Row 1: Risk & Sentiment
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Risk Assessment")
        render_risk_badge(intel.risk_level)
        st.markdown(f"**Sector**: {intel.sector}")
    
    with col2:
        st.markdown("#### Sentiment Analysis")
        fig_sentiment = render_sentiment_gauge(intel.sentiment_score)
        st.plotly_chart(fig_sentiment, width='stretch')
    
    st.divider()
    
    # Row 2: Relevance Explanation
    st.markdown("### 🎯 Relevance to Ticker")
    st.markdown(f'<div class="news-card">{intel.relevance_to_ticker}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Row 3: Major Events
    st.markdown("### 📰 Major Market-Moving Events")
    
    if intel.major_events:
        for i, event in enumerate(intel.major_events, 1):
            with st.expander(f"**Event {i}**: {event[:80]}..." if len(event) > 80 else f"**Event {i}**: {event}", expanded=(i==1)):
                st.markdown(event)
                st.caption("Source: Tavily Search Results")
    else:
        st.info("No major events detected for this sector.")
    
    st.divider()
    
    # Row 4: Raw Data
    with st.expander("🛠️ View Raw Agent Output"):
        st.json(intel.model_dump())

else:
    # Welcome state
    st.info("👈 Select a sector or enter a ticker in the sidebar to begin analysis.")
    
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    **Sector Scan Mode:**
    - Choose a sector (e.g., Technology)
    - Get broad market-moving news for that sector
    - See how sector trends affect representative stocks
    
    **Ticker Deep Dive Mode:**
    - Enter a specific ticker (e.g., NVDA)
    - Get sector news with ticker-specific relevance
    - Understand how macro sector events impact your stock
    
    **Features:**
    - 🔴🟡🟢 Risk Level Assessment (HIGH/MEDIUM/LOW)
    - 📊 Sentiment Scoring (-1.0 Bearish to +1.0 Bullish)
    - 📰 Major Events from Tavily News Search
    - 📚 Analysis History (last 5 searches)
    """)