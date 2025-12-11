import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from alphacouncil.tools.execution_tools import (
    check_trade_risk,
    get_portfolio_summary,
    get_current_price,
)
from alphacouncil.execution.limits import compute_position_headroom
from alphacouncil.execution.risk_rules import DEFAULT_LIMITS
from alphacouncil.execution.portfolio import PortfolioService
from alphacouncil.schema import RiskAssessment, TechnicalSignal, SectorIntel

# 1. Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0
)

# 2. Bind Tools
tools = [get_portfolio_summary, check_trade_risk, get_current_price]
risk_runnable = llm.bind_tools(tools).with_structured_output(RiskAssessment)

# 3. System Prompt
SYSTEM_PROMPT = """You are 'The Risk Manager'.
Your goal is to formalize the approval or rejection of a trade.

INPUT CONTEXT:
- A `VALIDATION_RESULT`: This comes from the hard-coded execution engine.
- If VALIDATION_RESULT is 'REJECTED', you MUST output a REJECTED verdict.
- If VALIDATION_RESULT is 'APPROVED', you may approve.

Output strictly valid JSON matching the RiskAssessment schema.
"""
import math
from datetime import date


def _resolve_live_price(ticker: str, fallback: float = 100.0) -> float:
    """Fetch the latest price, falling back to a neutral placeholder on failures."""
    try:
        raw = get_current_price.invoke(ticker)
        if raw and "Unavailable" not in raw:
            price = float(raw)
            # Check for NaN or invalid values
            if math.isnan(price) or price <= 0:
                return fallback
            return price
        return fallback
    except (ValueError, TypeError, Exception):
        return fallback


def _calculate_daily_pnl(state, default_price: float = 100.0) -> float:
    """
    Calculate today's realized P&L from trade history.
    Returns negative value for losses.
    """
    today_str = date.today().isoformat()
    daily_pnl = 0.0
    
    for trade in state.trade_history:
        # TradeRecord is a Pydantic model, use attribute access
        trade_date = trade.timestamp[:10]  # Extract YYYY-MM-DD
        if trade_date == today_str:
            # Add realized P&L from sells (pnl field is only set for SELL trades)
            if trade.pnl is not None:
                daily_pnl += trade.pnl
    
    return daily_pnl


def risk_manager_agent(state):
    ticker = state["ticker"].upper()
    tech_signal: TechnicalSignal = state.get("technical_signal")
    fund_signal: SectorIntel = state.get("fundamental_signal")
    messages = state.get("messages", [])
    
    # Check if Manual Override (The "User Card")
    is_manual = (tech_signal and tech_signal.regime == "MANUAL_OVERRIDE")

    live_price = _resolve_live_price(ticker)

    # -------------------------------------------------------
    # 0. PARSE ACTION FROM REQUEST (BUY or SELL)
    # -------------------------------------------------------
    action = "BUY"  # default
    if is_manual and messages:
        last_msg = messages[-1].content if messages else ""
        action_match = re.search(r"requests to (\w+)", last_msg, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).upper()
    elif tech_signal:
        # Infer from signal
        if tech_signal.signal in ["SELL", "STRONG_SELL"]:
            action = "SELL"

    # -------------------------------------------------------
    # SELL ORDERS: SIMPLIFIED VALIDATION (No Limit Checks)
    # -------------------------------------------------------
    if action == "SELL":
        # Parse quantity
        requested_qty = 0
        if is_manual and messages:
            last_msg = messages[-1].content
            match = re.search(r"requests to \w+ (\d+) shares", last_msg)
            if match:
                requested_qty = int(match.group(1))
            else:
                requested_qty = 10
        else:
            # For automated sells, use technician confidence
            base_size = 5000.0
            if tech_signal and tech_signal.confidence >= 0.8:
                base_size = 8000.0
            if live_price > 0:
                requested_qty = int(base_size / live_price)
        
        target_qty = max(requested_qty, 0)
        
        # Check if we have enough shares to sell
        portfolio = PortfolioService()
        state_data = portfolio.get_state()
        position = state_data.holdings.get(ticker)
        
        if not position:
            return {"risk_assessment": RiskAssessment(
                verdict="REJECTED",
                reason=f"HARD STOP: No position in {ticker} to sell.",
                approved_quantity=0,
                max_exposure_allowed=0.0,
                risk_score=10
            )}
        
        if position.quantity < target_qty:
            # Reduce to available quantity
            target_qty = position.quantity
        
        if target_qty <= 0:
            return {"risk_assessment": RiskAssessment(
                verdict="REJECTED",
                reason=f"HARD STOP: No shares available to sell.",
                approved_quantity=0,
                max_exposure_allowed=0.0,
                risk_score=10
            )}
        
        # SELL is approved - no sector/position limits apply
        proceeds = target_qty * live_price
        return {"risk_assessment": RiskAssessment(
            verdict="APPROVED",
            reason=f"SELL order approved for {target_qty} shares. Estimated proceeds: ${proceeds:,.2f}",
            approved_quantity=target_qty,
            max_exposure_allowed=proceeds,
            risk_score=2
        )}

    # -------------------------------------------------------
    # BUY ORDERS: FULL VALIDATION
    # -------------------------------------------------------
    
    # 0. HARD STOP: Daily Drawdown Check
    portfolio = PortfolioService()
    state_data = portfolio.get_state()
    daily_pnl = _calculate_daily_pnl(state_data, live_price)
    total_equity = state_data.cash_balance + sum(
        pos.quantity * _resolve_live_price(t) 
        for t, pos in state_data.holdings.items()
    )
    if total_equity > 0 and daily_pnl / total_equity < -DEFAULT_LIMITS.MAX_DAILY_DRAWDOWN:
        return {"risk_assessment": RiskAssessment(
            verdict="REJECTED",
            reason=f"HARD STOP: Daily drawdown limit ({DEFAULT_LIMITS.MAX_DAILY_DRAWDOWN:.0%}) exceeded. Current loss: {daily_pnl/total_equity:.1%}",
            approved_quantity=0, max_exposure_allowed=0.0, risk_score=10
        )}
    
    # 1. SOFT GATES (Strategy Quality) - APPLIES TO ALL TRADES
    # Rule A: Technician Confidence Threshold
    if tech_signal and tech_signal.confidence < 0.60:
        return {"risk_assessment": RiskAssessment(
            verdict="REJECTED",
            reason=f"SOFT STOP: Technician confidence too low ({tech_signal.confidence:.0%}). Consider waiting for stronger signal.",
            approved_quantity=0, max_exposure_allowed=0.0, risk_score=4
        )}
    
    # Rule B: Fundamental Veto
    if fund_signal and fund_signal.risk_level == "HIGH" and fund_signal.sentiment_score < -0.2:
        return {"risk_assessment": RiskAssessment(
            verdict="REJECTED",
            reason=f"SOFT STOP: High Sector Risk + Negative Sentiment ({fund_signal.sentiment_score:.2f}). Avoid new positions.",
            approved_quantity=0, max_exposure_allowed=0.0, risk_score=8
        )}

    # 2. PARSE TARGET QUANTITY
    requested_qty = 0
    if is_manual:
        last_msg = messages[-1].content if messages else ""
        match = re.search(r"requests to \w+ (\d+) shares", last_msg)
        if match:
            requested_qty = int(match.group(1))
        else:
            requested_qty = 10 
    else:
        # Automated Sizing Logic
        base_size = 5000.0
        if tech_signal.confidence >= 0.8:
            base_size = 8000.0
        
        # Apply Risk Haircut (Only if we passed the Soft Gate)
        if fund_signal and fund_signal.risk_level == "MEDIUM":
            base_size *= 0.8

        if live_price > 0:
            requested_qty = int(base_size / live_price)
        else:
            requested_qty = 0

    target_qty = max(requested_qty, 0)

    limits = compute_position_headroom(ticker, live_price)

    # 2b. HARD STOP: Reject if exceeds limits (with max_qty suggestion)
    if target_qty > limits["max_qty"]:
        max_allowed = limits["max_qty"]
        
        # Build detailed rejection reason
        limit_reasons = []
        if limits["cash_max_qty"] < target_qty:
            limit_reasons.append(
                f"Cash: {limits['cash_max_qty']} shares (after ${DEFAULT_LIMITS.MIN_CASH_BUFFER:,.0f} buffer)"
            )
        if limits["sector_max_qty"] < target_qty and limits["total_equity"] > 0:
            limit_reasons.append(
                f"{limits['sector']} sector: {limits['sector_max_qty']} shares ({limits['sector_limit_pct']:.0%} cap)"
            )
        if limits["single_position_max_qty"] < target_qty and limits["total_equity"] > 0:
            limit_reasons.append(
                f"Single position: {limits['single_position_max_qty']} shares ({DEFAULT_LIMITS.MAX_SINGLE_POSITION:.0%} cap)"
            )
        
        limits_detail = "; ".join(limit_reasons) if limit_reasons else "No capacity available"
        
        return {"risk_assessment": RiskAssessment(
            verdict="REJECTED",
            reason=f"LIMIT EXCEEDED: Requested {target_qty} shares. Maximum allowed: {max_allowed} shares. [{limits_detail}]",
            approved_quantity=0,
            max_exposure_allowed=max_allowed * live_price,
            risk_score=6
        )}

    total_equity = limits["total_equity"]
    sector_pct_current = (
        limits["current_sector_value"] / total_equity if total_equity > 0 else 0.0
    )
    sector_pct_post = (
        (limits["current_sector_value"] + target_qty * live_price) / total_equity
        if total_equity > 0
        else 0.0
    )
    position_pct_current = (
        limits["existing_position_value"] / total_equity if total_equity > 0 else 0.0
    )
    position_pct_post = (
        (limits["existing_position_value"] + target_qty * live_price) / total_equity
        if total_equity > 0
        else 0.0
    )

    cash_after_trade = limits["cash_balance"] - (target_qty * live_price)
    requested_cost = requested_qty * live_price
    final_cost = target_qty * live_price

    cap_candidates = [limits["cash_available_for_trade"]]
    if total_equity > 0:
        cap_candidates.extend(
            [limits["sector_value_room"], limits["single_position_value_room"]]
        )
    allowable_capital = max(0.0, min(cap_candidates)) if cap_candidates else 0.0

    # 3. THE HARD GATE (Solvency Check) - APPLIES TO ALL BUY ORDERS
    validation_msg = check_trade_risk.invoke(
        {"ticker": ticker, "action": "BUY", "quantity": target_qty}
    )

    if "REJECTED" in validation_msg:
        return {
            "risk_assessment": RiskAssessment(
                verdict="REJECTED",
                reason=f"HARD STOP: {validation_msg}",
                approved_quantity=0,
                max_exposure_allowed=allowable_capital,
                risk_score=10,
            )
        }

    # Trade passed all gates - prepare context for LLM approval
    context = f"""
    REQUEST: Validate BUY Trade for {ticker}
    MODE: {"MANUAL USER EXECUTION" if is_manual else "AUTOMATED STRATEGY"}
    PRICE: ${live_price:.2f}
    QUANTITY: {target_qty}
    TRADE_VALUE: ${final_cost:,.2f}
    CASH_BALANCE: ${limits['cash_balance']:,.2f}
    CASH_AFTER_TRADE: ${cash_after_trade:,.2f}
    SECTOR: {limits['sector']} | CURRENT: {sector_pct_current:.1%} | POST: {sector_pct_post:.1%}
    POSITION: CURRENT: {position_pct_current:.1%} | POST: {position_pct_post:.1%}

    *** VALIDATION RESULT: {validation_msg} ***
    """

    response = risk_runnable.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)] + [HumanMessage(content=context)]
    )

    if response.verdict == "APPROVED":
        response.approved_quantity = target_qty

    response.max_exposure_allowed = allowable_capital

    return {"risk_assessment": response}