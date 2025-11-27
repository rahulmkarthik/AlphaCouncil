from langchain_core.tools import tool
from alphacouncil.execution.portfolio import PortfolioService
from alphacouncil.execution.risk_rules import DEFAULT_LIMITS
from alphacouncil.data.live_feed import LiveMarketFeed

portfolio = PortfolioService()
market_feed = LiveMarketFeed.get_instance()

@tool
def get_portfolio_summary() -> str:
    """Returns cash, holdings, and total exposure using LIVE prices."""
    state = portfolio.get_state()
    lines = [f"💰 CASH: ${state.cash_balance:,.2f}"]
    
    total_exposure = 0.0
    lines.append("📊 HOLDINGS:")
    
    if not state.holdings:
        lines.append("  (Empty)")
    else:
        for ticker, pos in state.holdings.items():
            # USE REAL PRICE HERE
            live_price = market_feed.get_price(ticker) or pos.avg_price
            val = pos.quantity * live_price
            
            # Calculate Unrealized P&L
            pnl_pct = ((live_price / pos.avg_price) - 1) * 100
            
            total_exposure += val
            lines.append(f"  - {ticker}: {pos.quantity} shs @ ${live_price:.2f} (Avg: ${pos.avg_price:.2f}) [{pnl_pct:+.1f}%]")
            
    lines.append(f"📉 TOTAL EXPOSURE: ${total_exposure:,.2f}")
    return "\n".join(lines)

@tool
def check_trade_risk(ticker: str, action: str, quantity: int) -> str:
    """
    Validates trade risk using LIVE prices.
    Args:
        ticker: Symbol
        action: BUY/SELL
        quantity: Number of shares
    """
    # 1. Get Real Price
    price = market_feed.get_price(ticker)
    if price is None:
        return f"REJECTED: Could not fetch live price for {ticker}"
        
    state = portfolio.get_state()
    total_cost = quantity * price
    ticker = ticker.upper()
    
    # 2. Cash Check
    if action.upper() == "BUY":
        if state.cash_balance < total_cost:
            return f"REJECTED: Insufficient Cash. Needed ${total_cost:,.2f}, Have ${state.cash_balance:,.2f}"

    # 3. Concentration Check (using live equity value)
    current_holdings_val = sum(
        (market_feed.get_price(t) or p.avg_price) * p.quantity 
        for t, p in state.holdings.items()
    )
    total_equity = state.cash_balance + current_holdings_val
    
    if action.upper() == "BUY":
        # Projected value of THIS position after trade
        existing_val = 0
        if ticker in state.holdings:
            existing_val = state.holdings[ticker].quantity * price
            
        new_position_val = existing_val + total_cost
        concentration_pct = new_position_val / total_equity
        
        if concentration_pct > DEFAULT_LIMITS.MAX_SINGLE_POSITION:
             return f"REJECTED: Concentration Risk. {ticker} would be {concentration_pct:.1%} of portfolio (Limit: {DEFAULT_LIMITS.MAX_SINGLE_POSITION:.0%})"

    return f"APPROVED (Est. Price: ${price:.2f}, Total: ${total_cost:,.2f})"

@tool
def get_current_price(ticker: str) -> str: # <--- NEW TOOL FOR AGENT
    """Fetches the latest real-time price for a ticker."""
    price = market_feed.get_price(ticker)
    if price:
        return f"{price:.2f}"
    return "Unavailable"