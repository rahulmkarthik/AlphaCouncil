import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from alphacouncil.tools.vol_tools import get_vol_metrics
from alphacouncil.schema import TechnicalSignal

# 1. Initialize Gemini 3.0 Pro
# 'max_retries' helps if the API is momentarily busy
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", # Or "gemini-1.5-pro" if flash unavailable
    temperature=0,
    max_retries=2
)

# 2. Bind Tools
tools = [get_vol_metrics]

# 3. Create the "Structured" Agent
# This tells Gemini: "You MUST output a JSON object matching TechnicalSignal"
technician_runnable = llm.bind_tools(tools).with_structured_output(TechnicalSignal)

# 4. System Prompt
SYSTEM_PROMPT = SYSTEM_PROMPT = """You are 'The Technician', an expert Quantitative Researcher using VolSense.

YOUR DATA:
- `vol_spread_pct`: If positive, volatility is rising relative to today. If negative, it is mean-reverting.
- `rank_in_sector`: 
    - > 0.90: This stock is the most volatile in its sector (Idiosyncratic Risk).
    - ~ 0.50: It is moving with the herd (Systematic Risk).
- `heuristic_signal`: A baseline rule-based signal (long/short/neutral). Critique it.

DECISION LOGIC:
1. **Long Vol (BUY):** Z-Score > 1.5 AND vol_spread_pct > 0 (Breakout).
2. **Short Vol (SELL):** Z-Score > 2.0 BUT vol_spread_pct < -0.05 (Mean Reversion).
3. **Relative Value (HEDGE):** High Z-Score but rank_in_sector < 0.6 (The whole sector is crashing, do not short vol).

Output strictly valid JSON matching the TechnicalSignal schema.
"""

# 5. Node Function
def technician_agent(state):
    # Gemini forbids empty message lists.
    # If the graph started with just a ticker, we must manually create the first Human prompt.
    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=f"Analyze the volatility for {state['ticker']}")]

    # Now the input is [SystemMessage, HumanMessage] -> Valid for Gemini
    response = technician_runnable.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    
    return {"technical_signal": response}