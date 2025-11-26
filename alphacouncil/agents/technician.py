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
SYSTEM_PROMPT = """You are 'The Technician'. Analyze the volatility metrics provided by the tool.

CRITICAL RULES:
1. **TRUST THE Z-SCORE**: If `z_score` > 1.5, you CANNOT output 'WAIT'. You must output 'BUY' (Long Vol) or 'HEDGE'.
2. **TRUST THE HEURISTIC**: If `heuristic_signal` is 'long', your default signal is 'BUY'. If 'short', your default is 'SELL'.
3. **REGIME AWARENESS**: If `regime` is 'Spike' or 'High Vol', treat this as a high-conviction setup.

Output strictly valid JSON matching the TechnicalSignal schema.
"""

def technician_agent(state):
    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=f"Analyze the volatility for {state['ticker']}")]

    # 1. Run the Agent Logic
    ai_msg = technician_runnable.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    
    # 2. CAPTURE DATA FOR DASHBOARD (The "Bridge" Fix)
    # We manually fetch the raw data so we can pass it to the UI
    raw_data = None
    try:
        # Manually invoke tool to get the payload for the UI state
        data_json = get_vol_metrics.invoke({"ticker": state["ticker"]})
        raw_data = json.loads(data_json)
    except:
        raw_data = {}

    return {
        "technical_signal": ai_msg,
        "raw_vol_data": raw_data
    }