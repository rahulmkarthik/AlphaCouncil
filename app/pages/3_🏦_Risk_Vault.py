import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from langchain_core.messages import HumanMessage

# Internal Imports
from alphacouncil.execution.portfolio import PortfolioService
from alphacouncil.agents.risk_manager import risk_manager_agent
from alphacouncil.data.live_feed import LiveMarketFeed
from alphacouncil.schema import TechnicalSignal 
from volsense_inference.sector_mapping import get_sector_map

# 1. PAGE CONFIG
st.set_page_config(page_title="Risk Vault", page_icon="🏦", layout="wide")

# Custom CSS for "Terminal" Vibe
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
</style>
""", unsafe_allow_html=True)

# 2. INITIALIZE SERVICES
portfolio = PortfolioService()
market_feed = LiveMarketFeed.get_instance()
state = portfolio.get_state()
sector_map = get_sector_map("v507")

# --- DATA PREP (Calculate Metrics Once) ---
cash = state.cash_balance
holdings_val = 0.0
open_pnl = 0.0
# Sum up realized P&L from Sell trades
realized_pnl = sum([t.pnl for t in state.trade_history if t.action == "SELL" and t.pnl is not None])
total_trades = len(state.trade_history)

# Allocation Data
alloc_data = []

if state.holdings:
    for t, p in state.holdings.items():
        price = market_feed.get_price(t) or p.avg_price
        mkt_val = p.quantity * price
        
        # Aggregates
        holdings_val += mkt_val
        open_pnl += (mkt_val - (p.quantity * p.avg_price))
        
        # For Charts
        alloc_data.append({
            "Ticker": t,
            "Sector": sector_map.get(t, "Unknown"),
            "Value": mkt_val,
            "Allocation": 0.0 # Calc later
        })

total_equity = cash + holdings_val

# Finalize Allocation %
for item in alloc_data:
    item["Allocation"] = item["Value"] / total_equity

# 3. SIDEBAR: LEDGER
with st.sidebar:
    st.header("🏦 Vault Ledger")
    st.metric("Net Liq. Value", f"${total_equity:,.2f}", 
              delta=f"${open_pnl + realized_pnl:,.2f} All-Time P&L")
    st.metric("Buying Power", f"${cash:,.2f}")
    st.divider()
    if st.button("🔄 Refresh Market Data"):
        market_feed.refresh_snapshot()
        st.rerun()

# 4. MAIN UI HEADER
st.title("Risk Manager's Vault")
st.caption(f"Portfolio Intelligence & Compliance • Last Updated: {datetime.now().strftime('%H:%M:%S')}")

# --- ROW 1: KPI STRIP ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Open P&L", f"${open_pnl:,.2f}", delta_color="normal")
c2.metric("Realized P&L", f"${realized_pnl:,.2f}", delta_color="normal")
c3.metric("Active Positions", len(state.holdings))
c4.metric("Total Trades", total_trades)

st.divider()

# --- ROW 2: VISUAL INTELLIGENCE ---
col_charts_1, col_charts_2 = st.columns(2)

with col_charts_1:
    st.subheader("🎨 Asset Allocation")
    if alloc_data:
        df_alloc = pd.DataFrame(alloc_data)
        # Sunburst Chart via Plotly
        fig = px.sunburst(
            df_alloc, 
            path=['Sector', 'Ticker'], 
            values='Value',
            color='Sector',
            color_discrete_sequence=px.colors.qualitative.Prism,
            height=300
        )
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, width='stretch') # Replaced use_container_width
    else:
        st.info("No active positions to display allocation.")

with col_charts_2:
    st.subheader("📈 Performance History")
    if state.trade_history:
        # Reconstruct Cumulative P&L Curve
        trades_df = pd.DataFrame([vars(t) for t in state.trade_history])
        # Filter for realized P&L events (Sells)
        pnl_events = trades_df[trades_df['pnl'].notnull()].copy()
        
        if not pnl_events.empty:
            pnl_events['timestamp'] = pd.to_datetime(pnl_events['timestamp'])
            pnl_events['cumulative_pnl'] = pnl_events['pnl'].cumsum()
            
            fig_pnl = px.area(
                pnl_events, 
                x='timestamp', 
                y='cumulative_pnl',
                line_shape='hv',
                markers=True,
                height=300
            )
            fig_pnl.update_layout(
                margin=dict(t=0, l=0, r=0, b=0), 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Realized P&L ($)",
                xaxis_title=""
            )
            fig_pnl.update_traces(line_color='#00FF00', fill_color='rgba(0,255,0,0.1)')
            st.plotly_chart(fig_pnl, width='stretch') # Replaced use_container_width
        else:
            st.info("No realized P&L events yet (Only Buys executed).")
    else:
        st.info("No trade history available.")

st.divider()

# --- ROW 3: HOLDINGS & HISTORY ---
col_hold, col_hist = st.columns([2, 1])

with col_hold:
    st.subheader("💼 Position Blotter")
    if not state.holdings:
        st.info("Portfolio is empty. Execute trades below.")
    else:
        rows = []
        for t, p in state.holdings.items():
            curr_price = market_feed.get_price(t) or p.avg_price
            mkt_val = p.quantity * curr_price
            pnl_val = mkt_val - (p.quantity * p.avg_price)
            pnl_pct = (pnl_val / (p.quantity * p.avg_price)) * 100
            
            rows.append({
                "Ticker": t,
                "Shares": p.quantity,
                "Avg Cost": f"${p.avg_price:.2f}",
                "Mark": f"${curr_price:.2f}",
                "Mkt Value": f"${mkt_val:,.2f}",
                "P&L ($)": f"${pnl_val:,.2f}",
                "P&L (%)": f"{pnl_pct:+.2f}%"
            })
        
        # Display with stretch width as requested
        st.dataframe(pd.DataFrame(rows), width='stretch')

with col_hist:
    st.subheader("📜 Tape")
    if state.trade_history:
        recent = state.trade_history[-5:][::-1]
        for t in recent:
            color = "green" if t.action == "BUY" else "red"
            emoji = "🟢" if t.action == "BUY" else "🔴"
            st.markdown(f"{emoji} **{t.action}** {t.quantity} **{t.ticker}** @ ${t.price:.2f}")
    else:
        st.caption("Tape is quiet.")

st.divider()


def _escape_currency(text: str) -> str:
    return text.replace("$", "\\$")


# --- ROW 4: TRADING TERMINAL (Preserved Logic) ---
st.subheader("⚡ Execution Console")

c1, c2 = st.columns([1, 2])

# Session state for trade flow
if "pending_trade" not in st.session_state:
    st.session_state["pending_trade"] = None

with c1:
    with st.form("trade_form"):
        st.caption("Order Ticket")
        ticker = st.text_input("Ticker", value="NVDA").upper()
        c_act, c_qty = st.columns(2)
        action = c_act.selectbox("Side", ["BUY", "SELL"])
        qty = c_qty.number_input("Qty", min_value=1, value=10)
        
        submitted = st.form_submit_button("🛡️ Submit Order")

if submitted:
    # 1. RUN RISK ANALYSIS
    with st.spinner("Compliance Engine running..."):
        mock_tech = TechnicalSignal(
            ticker=ticker,
            signal="STRONG_BUY" if action == "BUY" else "STRONG_SELL",
            confidence=1.0,
            regime="MANUAL_OVERRIDE",
            key_drivers=["User Discretion"],
            reasoning="Manual trade initiated from Risk Vault."
        )
        
        # Inject explicit message for Regex parsing
        request_text = f"User manually requests to {action} {qty} shares of {ticker}."
        
        agent_state = {
            "ticker": ticker,
            "technical_signal": mock_tech,
            "fundamental_signal": None,
            "messages": [HumanMessage(content=request_text)]
        }
        
        try:
            result = risk_manager_agent(agent_state)
            st.session_state["pending_trade"] = {
                "ticker": ticker,
                "action": action,
                "requested_qty": qty,
                "assessment": result["risk_assessment"]
            }
        except Exception as e:
            st.error(f"Agent Error: {e}")

# OUTSIDE FORM
with c2:
    if st.session_state["pending_trade"]:
        trade = st.session_state["pending_trade"]
        assessment = trade["assessment"]
        approved_qty = max(assessment.approved_quantity, 0)
        
        st.caption("Compliance Review")
        
        if assessment.verdict == "APPROVED":
            st.success(_escape_currency(f"✅ **APPROVED**: {assessment.reason}"))
            
            # Fetch Live Price for display
            from alphacouncil.tools.execution_tools import get_current_price
            price_str = get_current_price.invoke(trade["ticker"])
            exec_price = float(price_str) if "Unavailable" not in price_str else 0.0
            
            qty_to_execute = approved_qty or trade["requested_qty"]
            est_total = qty_to_execute * exec_price
            
            st.markdown(f"""
            <div style='background: #1e2a1e; padding: 10px; border-radius: 5px; border-left: 4px solid #00ff00;'>
                <h4 style='margin:0'>Confirm Execution</h4>
                <p style='margin:0'><b>{trade['action']} {qty_to_execute} {trade['ticker']}</b> @ ~${exec_price:.2f}</p>
                <p style='margin:0; font-size: 0.9em; color: #aaa'>Est. Total: ${est_total:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer
            
            col_confirm, col_cancel = st.columns(2)
            if col_confirm.button("🚀 EXECUTE", type="primary"): # Replaced use_container_width which isn't valid for button in some versions, keeping simple
                msg = portfolio.execute_trade(trade["ticker"], trade["action"], qty_to_execute, exec_price)
                st.toast(msg, icon="✅")
                st.session_state["pending_trade"] = None
                st.rerun()
                
            if col_cancel.button("CANCEL"):
                st.session_state["pending_trade"] = None
                st.rerun()
                
        else:
            # --- HARD vs SOFT STOP VISUALIZATION ---
            reason = assessment.reason
            if "HARD STOP" in reason:
                st.error(_escape_currency(f"⛔ **CRITICAL REJECTION**\n\n{reason}"))
                st.markdown("This trade violates strict portfolio mandates (Cash or Concentration).")
            elif "SOFT STOP" in reason:
                st.warning(_escape_currency(f"⚠️ **STRATEGY WARNING**\n\n{reason}"))
                st.markdown("The trade is risky but valid. You may override this.")
                
                # Allow Soft Override
                if st.button("⚠️ OVERRIDE & EXECUTE", type="secondary"):
                    from alphacouncil.tools.execution_tools import get_current_price
                    price_str = get_current_price.invoke(trade["ticker"])
                    exec_price = float(price_str) if "Unavailable" not in price_str else 0.0
                    qty_to_execute = approved_qty or trade["requested_qty"]
                    msg = portfolio.execute_trade(trade["ticker"], trade["action"], qty_to_execute, exec_price)
                    st.toast(f"Override Successful: {msg}", icon="⚠️")
                    st.session_state["pending_trade"] = None
                    st.rerun()
            else:
                # Generic Rejection
                st.error(_escape_currency(f"❌ **BLOCKED**: {reason}"))
            
            if st.button("Dismiss"):
                st.session_state["pending_trade"] = None
                st.rerun()