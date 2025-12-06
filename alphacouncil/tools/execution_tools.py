from alphacouncil.utils.langchain_stub import tool
from alphacouncil.execution.portfolio import PortfolioService
from alphacouncil.execution.risk_rules import DEFAULT_LIMITS
from alphacouncil.data.live_feed import LiveMarketFeed
from alphacouncil.execution.limits import compute_position_headroom

# Global: Market Feed (Keep global because it manages the cache/singleton)
market_feed = LiveMarketFeed.get_instance()

# Global: PortfolioService REMOVED to prevent stale state.
# We now instantiate it fresh inside every tool.

@tool
def get_portfolio_summary() -> str:
    """Returns cash, holdings, and total exposure using LIVE prices."""
    # FIX: Instantiate fresh to read the latest JSON state from disk
    portfolio = PortfolioService()
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
            
            # Calculate Unrealized P&L (Preserved from your original file)
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
    # FIX: Instantiate fresh to see the updated cash/holdings from recent trades
    portfolio = PortfolioService()
    state = portfolio.get_state()

    import math
    
    # DEBUG: Log price lookup
    price = market_feed.get_price(ticker)
    print(f"[DEBUG check_trade_risk] Ticker: {ticker}, Raw price: {price}, Type: {type(price)}")
    
    if price is None or (isinstance(price, float) and math.isnan(price)):
        # Try refreshing the market feed
        print(f"[DEBUG] Price invalid for {ticker}, attempting refresh...")
        market_feed.refresh_snapshot()
        price = market_feed.get_price(ticker)
        print(f"[DEBUG] After refresh - Price: {price}")
        
        if price is None or (isinstance(price, float) and math.isnan(price)):
            return f"REJECTED: Could not fetch valid live price for {ticker}. Try clicking 'Refresh Market Data'."

    action = action.upper()
    ticker = ticker.upper()
    total_cost = quantity * price
    
    print(f"[DEBUG check_trade_risk] Total cost: {total_cost}, Cash: {state.cash_balance}")

    if action == "BUY":
        if state.cash_balance < total_cost:
            return (
                f"REJECTED: Insufficient Cash. Needed ${total_cost:,.2f}, "
                f"Have ${state.cash_balance:,.2f}"
            )

        headroom = compute_position_headroom(ticker, price, state=state)
        print(f"[DEBUG check_trade_risk] Headroom: cash_max_qty={headroom['cash_max_qty']}, cash_available={headroom['cash_available_for_trade']}")

        if headroom["cash_max_qty"] <= 0:
            return (
                f"REJECTED: Cash buffer of ${DEFAULT_LIMITS.MIN_CASH_BUFFER:,.0f} "
                f"would be violated (deployable ${headroom['cash_available_for_trade']:,.2f}). "
                f"Price: ${price:.2f}, Qty: {quantity}"
            )

        if quantity > headroom["cash_max_qty"]:
            return (
                f"REJECTED: Trade needs ${total_cost:,.2f} but only "
                f"${headroom['cash_available_for_trade']:,.2f} is deployable after the cash buffer "
                f"of ${DEFAULT_LIMITS.MIN_CASH_BUFFER:,.0f}."
            )

        total_equity = headroom["total_equity"]
        if total_equity > 0:
            current_sector_pct = headroom["current_sector_value"] / total_equity
            projected_sector_pct = (
                headroom["current_sector_value"] + quantity * price
            ) / total_equity

            if headroom["sector_max_qty"] <= 0:
                return (
                    f"REJECTED: {headroom['sector']} sector already at "
                    f"{current_sector_pct:.1%} (limit {headroom['sector_limit_pct']:.0%})."
                )

            if quantity > headroom["sector_max_qty"]:
                return (
                    f"REJECTED: {headroom['sector']} exposure would reach "
                    f"{projected_sector_pct:.1%} (limit {headroom['sector_limit_pct']:.0%})."
                )

            current_position_pct = headroom["existing_position_value"] / total_equity
            projected_position_pct = (
                headroom["existing_position_value"] + quantity * price
            ) / total_equity

            if headroom["single_position_max_qty"] <= 0:
                return (
                    f"REJECTED: {ticker} already at {current_position_pct:.1%} of "
                    f"portfolio (limit {DEFAULT_LIMITS.MAX_SINGLE_POSITION:.0%})."
                )

            if quantity > headroom["single_position_max_qty"]:
                return (
                    f"REJECTED: {ticker} would represent {projected_position_pct:.1%} of the "
                    f"portfolio (limit {DEFAULT_LIMITS.MAX_SINGLE_POSITION:.0%})."
                )

        projected_cash = state.cash_balance - total_cost
        projected_sector_pct = (
            headroom["current_sector_value"] + quantity * price
        ) / headroom["total_equity"] if headroom["total_equity"] > 0 else 0.0
        projected_position_pct = (
            headroom["existing_position_value"] + quantity * price
        ) / headroom["total_equity"] if headroom["total_equity"] > 0 else 0.0

        return (
            f"APPROVED (Est. Price: ${price:.2f}, Total: ${total_cost:,.2f}, "
            f"Cash After: ${projected_cash:,.2f}, Sector: {projected_sector_pct:.1%}, "
            f"Ticker: {projected_position_pct:.1%})"
        )

    if action == "SELL":
        projected_cash = state.cash_balance + total_cost
        return (
            f"APPROVED (Est. Price: ${price:.2f}, Total: ${total_cost:,.2f}, "
            f"Cash After: ${projected_cash:,.2f})"
        )

    return f"REJECTED: Unsupported action '{action}'"

@tool
def get_current_price(ticker: str) -> str:
    """Fetches the latest real-time price for a ticker."""
    price = market_feed.get_price(ticker)
    if price:
        return f"{price:.2f}"
    return "Unavailable"