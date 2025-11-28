import streamlit as st
import pandas as pd

st.set_page_config(page_title="The War Room", page_icon="⚔️", layout="wide")

st.title("⚔️ The War Room")
st.caption("Multi-Agent Debate Chamber")

# Layout mimicking a chat interface
chat_col, context_col = st.columns([2, 1])

with chat_col:
    st.info("This module will orchestrate a debate between the Technician and Fundamentalist.")
    st.chat_message("user").write("Should we buy NVDA?")
    st.chat_message("assistant", avatar="📐").write("Technicals are bullish. Volatility is compressing (Z=-1.2).")
    st.chat_message("assistant", avatar="📰").write("I disagree. Regulatory headwinds in China are too high.")
    st.chat_message("assistant", avatar="🏦").write("Risk Manager Interjecting: We are also at max sector capacity.")

with context_col:
    st.subheader("Live Context")
    st.markdown("**Subject:** NVDA")
    st.markdown("**Consensus:** `DIVERGENT`")