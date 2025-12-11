from alphacouncil.execution.portfolio import PortfolioService

# 1. Initialize
pf = PortfolioService()
print(f"💰 Starting Cash: ${pf.get_cash():,.2f}")

# 2. Buy NVDA
msg = pf.execute_trade("NVDA", "BUY", 10, 135.50)
print(msg)

# 3. Buy SPY
msg = pf.execute_trade("SPY", "BUY", 5, 500.00)
print(msg)

# 4. Sell NVDA (Profit)
msg = pf.execute_trade("NVDA", "SELL", 5, 140.00)
print(msg)

# 5. Check State
print(f"\n📊 Final Holdings: {pf.get_state().holdings.keys()}")
print(f"💰 Final Cash: ${pf.get_cash():,.2f}")