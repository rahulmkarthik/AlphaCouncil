from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from alphacouncil.tools.execution_tools import check_trade_risk, get_portfolio_summary, get_current_price
from alphacouncil.schema import RiskAssessment, TechnicalSignal, SectorIntel

# 1. Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0
)

# 2. Bind Tools
tools = [get_portfolio_summary, check_trade_risk, get_current_price]
risk_runnable = llm.bind_tools(tools).with_structured_output(RiskAssessment)

# 3. Updated System Prompt - Balanced & Deterministic
SYSTEM_PROMPT = """You are 'The Risk Manager', the gatekeeper of the AlphaCouncil.
Your goal is to validate proposed trades against portfolio constraints and fundamental risks.

INPUT CONTEXT:
- `TechnicalSignal`: The directional bias (BUY/SELL).
- `SectorIntel`: Fundamental risks (earnings, regulation, sentiment).
- `Proposed Trade`: A pre-calculated quantity derived from the combined confidence of both agents.

INSTRUCTIONS:
1. **CHECK PORTFOLIO**: Call `get_portfolio_summary` to see Cash and Holdings.
2. **REVIEW FUNDAMENTALS**: Acknowledge the Sector Risk Level. If the Proposed Trade size is small, it likely reflects a penalty applied due to high risk.
3. **VALIDATE**: Call `check_trade_risk` with the *exact* proposed parameters provided.
4. **VERDICT**:
   - If `check_trade_risk` returns "APPROVED" -> Verdict: APPROVED.
   - If `check_trade_risk` returns "REJECTED" (e.g., Insufficient Cash, Concentration) -> Verdict: REJECTED.

Output strictly valid JSON matching the RiskAssessment schema.
"""

def risk_manager_agent(state):
    ticker = state["ticker"]
    tech_signal: TechnicalSignal = state.get("technical_signal")
    fund_signal: SectorIntel = state.get("fundamental_signal")
    
    # -------------------------------------------------------
    # 1. INITIAL SIGNAL CHECK
    # -------------------------------------------------------
    if not tech_signal or tech_signal.signal not in ["BUY", "STRONG_BUY"]:
        return {"risk_assessment": RiskAssessment(
            verdict="REJECTED", reason="No Buy Signal from Technician", approved_quantity=0, max_exposure_allowed=0.0, risk_score=0
        )}

    # -------------------------------------------------------
    # 2. DETERMINISTIC SIZING ENGINE (Python Layer)
    # -------------------------------------------------------
    
    # A. Base Size from Technical Confidence
    base_size = 5000.0
    if tech_signal.confidence >= 0.8:
        base_size = 8000.0
    elif tech_signal.confidence < 0.5:
        base_size = 2000.0
        
    # B. Apply Fundamental Modifiers (Risk Haircuts)
    risk_reason = "None"
    if fund_signal:
        # HARD VETO: High Risk + Bearish Sentiment
        if fund_signal.risk_level == "HIGH" and fund_signal.sentiment_score < -0.2:
             return {"risk_assessment": RiskAssessment(
                verdict="REJECTED", 
                reason=f"Fundamental Veto: High Risk Sector with Negative Sentiment ({fund_signal.sentiment_score})", 
                approved_quantity=0, max_exposure_allowed=0.0, risk_score=9
            )}
        
        # SIZING PENALTY: High Risk but Neutral/Bullish Sentiment -> 50% Cut
        if fund_signal.risk_level == "HIGH":
            base_size *= 0.5
            risk_reason = "High Sector Risk (50% size reduction)"
        
        # SIZING PENALTY: Medium Risk -> 20% Cut
        elif fund_signal.risk_level == "MEDIUM":
            base_size *= 0.8
            risk_reason = "Medium Sector Risk (20% size reduction)"

    # C. Fetch Price & Calculate Qty
    try:
        price_str = get_current_price.invoke(ticker)
        price = float(price_str) if "Unavailable" not in price_str else None
    except:
        price = None
        
    if price:
        target_qty = int(base_size / price)
        est_value = target_qty * price
    else:
        target_qty = 0
        est_value = 0.0

    # -------------------------------------------------------
    # 3. CONSTRUCT CONTEXT FOR LLM
    # -------------------------------------------------------
    context = f"""
    REQUEST: Validate Trade for {ticker}
    
    --- INPUTS ---
    TECHNICIAN: {tech_signal.signal} (Conf: {tech_signal.confidence:.2f})
    FUNDAMENTALIST: Risk={fund_signal.risk_level if fund_signal else 'N/A'}, Sentiment={fund_signal.sentiment_score if fund_signal else 0}
    
    --- SIZING LOGIC ---
    Base Allocation: ${base_size / (0.5 if fund_signal and fund_signal.risk_level=='HIGH' else 1.0):.2f} (Before Adjustments)
    Risk Adjustment: {risk_reason}
    Final Allocation: ${base_size:.2f}
    
    --- PROPOSED EXECUTION ---
    Live Price: ${price}
    PROPOSED QUANTITY: {target_qty} shares
    EST. TOTAL VALUE: ${est_value:.2f}
    """
    
    messages = [HumanMessage(content=context)]
    
    # Run Agent
    response = risk_runnable.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
    
    return {"risk_assessment": response}