import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Any, Optional, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load env vars first!
load_dotenv()

from alphacouncil.agents.technician import technician_agent
from alphacouncil.agents.fundamentalist import fundamentalist_agent
# Import Schema types for type hinting
from alphacouncil.schema import TechnicalSignal, SectorIntel

# 1. Update State to hold OBJECTS, not just strings
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    ticker: str
    technical_signal: Any 
    fundamental_signal: Any
    # NEW: This field carries the raw numbers to the dashboard
    raw_vol_data: Optional[Dict[str, Any]]

# 2. Graph Definition
workflow = StateGraph(AgentState)

# 3. Add Nodes
workflow.add_node("technician", technician_agent)
workflow.add_node("fundamentalist", fundamentalist_agent)

# 4. Edges (Linear Flow: Tech -> Fund -> End)
workflow.add_edge(START, "technician")
workflow.add_edge("technician", "fundamentalist")
workflow.add_edge("fundamentalist", END)

# 5. Compile
app = workflow.compile()