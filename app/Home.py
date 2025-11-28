import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Internal Imports
from alphacouncil.execution.portfolio import PortfolioService
from alphacouncil.tools.vol_tools import VolSenseService
from alphacouncil.data.live_feed import LiveMarketFeed

load_dotenv()

st.set_page_config(
    page_title="AlphaCouncil Overview",
    page_icon="🏛️",
    layout="wide"
)

# Initialize Services
portfolio = PortfolioService()
pf_state = portfolio.get_state()
vol_service = VolSenseService.get_instance()
market = LiveMarketFeed.get_instance()

# --- HEADER ---
st.title("☕ Morning Briefing")
st.markdown(f"**Date:** {datetime.now().strftime('%A, %B %d, %Y')}")

# --- ROW 1: PORTFOLIO SNAPSHOT ---
st.subheader("🏦 Portfolio Snapshot")
c1, c2, c3 = st.columns(3)

# Calculate Est Equity
equity = pf_state.cash_balance
for p in pf_state.holdings.values():
    price = market.get_price(p.ticker) or p.avg_price
    equity += (p.quantity * price)

c1.metric("Net Liquidation Value", f"${equity:,.2f}", 
          delta=f"{equity - 100000:,.2f} P&L")
c2.metric("Cash Available", f"${pf_state.cash_balance:,.2f}")
c3.metric("Active Positions", len(pf_state.holdings))

st.divider()

# --- ROW 2: MARKET OVERVIEW (VolSense) ---
st.subheader("🌍 VolSense Market Scan")

# Simple check to see if we have data
if hasattr(vol_service, "_forecast_engine") and vol_service._forecast_engine:
    # In a real app, you'd cache the sector summary df
    st.info("Market Scan data would go here (Sector Heatmap from VolSense).")
    if st.button("Run Quick Market Scan"):
        st.write(vol_service.analyze_market_sectors())
else:
    st.warning("VolSense Engine is in Standby. Visit the **Technician's Console** to initialize.")

# --- NAVIGATION HINTS ---
st.divider()
st.markdown("### 🧭 Terminal Navigation")
c1, c2, c3, c4, c5 = st.columns(5)
c1.page_link("pages/1_📐_Technician.py", label="The Technician's Console", icon="📐")
c2.page_link("pages/2_📰_Fundamentalist.py", label="The Fundamentalist's Study", icon="📰")
c3.page_link("pages/3_🏦_Risk_Vault.py", label="The Execution Vault", icon="🏦")
c4.page_link("pages/4_⚔️_War_Room.py", label="The War Room", icon="⚔️")
c5.page_link("pages/5_🔬_Ticker_Scope.py", label="The Ticker Scope", icon="🔬")