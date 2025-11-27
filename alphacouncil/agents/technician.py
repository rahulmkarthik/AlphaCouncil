import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from alphacouncil.tools.vol_tools import get_vol_metrics
from alphacouncil.schema import TechnicalSignal

# Initialize Gemini 3.0 Pro
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", # Or gemini-1.5-pro
    temperature=0,
    max_retries=2
)

# Bind Tools
tools = [get_vol_metrics]
technician_runnable = llm.bind_tools(tools).with_structured_output(TechnicalSignal)

# SIMPLIFIED PROMPT
SYSTEM_PROMPT = """You are 'The Technician', an expert Quantitative Researcher. 
You are receiving pre-calculated signals from the VolSense 2.0 Engine.

INPUT DATA STRUCTURE:
- `signal.position`: The regime (e.g., BUY_DIP, DEFENSIVE, LONG_VOL_TREND).
- `signal.action`: The recommended trade (e.g., "Buy Calls", "Sell Iron Condor").
- `metrics.momentum_20d`: The long-term trend (Positive = Bullish).

INSTRUCTIONS:
1. **TRUST THE ENGINE**: Do not recalculate Z-scores. Use `signal.position` as your primary truth.
2. **TRANSLATE TO ACTION**:
   - `BUY_DIP` / `LONG_EQUITY` -> Output Signal: **BUY** (Reason: "Bullish Trend" or "Mean Reversion Opportunity")
   - `LONG_VOL_TREND` -> Output Signal: **BUY** (Reason: "Volatility Breakout")
   - `DEFENSIVE` / `FADE_RALLY` -> Output Signal: **SELL** (Reason: "Crash Risk" or "Bearish Trend")
   - `SHORT_VOL` / `NEUTRAL` -> Output Signal: **WAIT** or **HEDGE** (Reason: "Premium Selling" or "No Edge")
   
3. **EXPLAIN WITH MOMENTUM**: In your reasoning, explicitly mention if the 20d momentum supports or contradicts the volatility signal.

Output strictly valid JSON matching the TechnicalSignal schema.
"""

def technician_agent(state):
    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=f"Analyze the volatility for {state['ticker']}")]

    # 1. Run Agent
    ai_msg = technician_runnable.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    
    # 2. CAPTURE DATA (Same bridge logic, just ensuring it runs)
    raw_data = None
    try:
        data_json = get_vol_metrics.invoke({"ticker": state["ticker"]})
        raw_data = json.loads(data_json)
    except:
        raw_data = {}

    return {
        "technical_signal": ai_msg,
        "raw_vol_data": raw_data
    }