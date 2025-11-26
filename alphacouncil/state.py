import operator
from typing import Annotated, List, TypedDict, Union

class AgentState(TypedDict):
    # Inputs
    ticker: str
    
    # Agent Outputs (The "Debate")
    technical_signal: Annotated[str, "The raw signal from the Technician"]
    fundamental_signal: Annotated[str, "The news/macro check from the Fundamentalist"]
    risk_assessment: Annotated[str, "The constraints check from the Risk Manager"]
    
    # Decision
    final_action: str # "BUY", "SELL", "HEDGE", "WAIT"
    reasoning: str
    
    # Iteration control (to prevent infinite debates)
    iteration_count: int