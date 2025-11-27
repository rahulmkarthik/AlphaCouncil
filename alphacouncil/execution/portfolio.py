import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

# 1. Define Data Models (for type safety)
class Position(BaseModel):
    ticker: str
    quantity: int
    avg_price: float
    current_value: float = 0.0  # To be updated with live data

class TradeRecord(BaseModel):
    timestamp: str
    ticker: str
    action: str  # "BUY" or "SELL"
    quantity: int
    price: float
    total_cost: float
    pnl: Optional[float] = None  # Only for SELL trades

class PortfolioState(BaseModel):
    cash_balance: float
    holdings: Dict[str, Position]  # Ticker -> Position
    trade_history: List[TradeRecord]
    last_updated: str

# 2. The Persistence Service
class PortfolioService:
    def __init__(self, data_dir: str = "data", filename: str = "paper_portfolio.json"):
        self.file_path = os.path.join(os.getcwd(), data_dir, filename)
        self._ensure_dir_exists(data_dir)
        self.state = self._load_or_create()

    def _ensure_dir_exists(self, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _load_or_create(self) -> PortfolioState:
        """Loads the portfolio from disk or creates a fresh $100k account."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    data = json.load(f)
                return PortfolioState(**data)
            except Exception as e:
                print(f"⚠️ Failed to load portfolio: {e}. Starting fresh.")
        
        # Default State: $100,000 Cash
        return PortfolioState(
            cash_balance=100000.0,
            holdings={},
            trade_history=[],
            last_updated=datetime.now().isoformat()
        )

    def save(self):
        """Persists current state to JSON."""
        self.state.last_updated = datetime.now().isoformat()
        with open(self.file_path, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    # --- READ METHODS ---
    def get_state(self) -> PortfolioState:
        return self.state

    def get_cash(self) -> float:
        return self.state.cash_balance

    def get_holding(self, ticker: str) -> Optional[Position]:
        return self.state.holdings.get(ticker.upper())

    # --- WRITE METHODS ---
    def execute_trade(self, ticker: str, action: str, qty: int, price: float) -> str:
        """
        Executes a trade ONLY if valid (Risk Manager should check limits before calling this).
        Updates Cash, Holdings (Avg Cost), and History.
        """
        ticker = ticker.upper()
        action = action.upper()
        total_cost = qty * price

        if action == "BUY":
            if self.state.cash_balance < total_cost:
                return f"❌ REJECTED: Insufficient Cash (${self.state.cash_balance:.2f} < ${total_cost:.2f})"
            
            # Update Cash
            self.state.cash_balance -= total_cost
            
            # Update Position (Average Cost Basis Logic)
            if ticker in self.state.holdings:
                pos = self.state.holdings[ticker]
                new_total_qty = pos.quantity + qty
                # Weighted Average Price
                pos.avg_price = ((pos.quantity * pos.avg_price) + total_cost) / new_total_qty
                pos.quantity = new_total_qty
            else:
                self.state.holdings[ticker] = Position(ticker=ticker, quantity=qty, avg_price=price)

            self._log_trade(ticker, "BUY", qty, price, total_cost)
            self.save()
            return f"✅ FILLED: Bought {qty} {ticker} @ ${price:.2f}"

        elif action == "SELL":
            pos = self.state.holdings.get(ticker)
            if not pos or pos.quantity < qty:
                return f"❌ REJECTED: Insufficient Holdings ({pos.quantity if pos else 0} < {qty})"
            
            # Update Cash
            self.state.cash_balance += total_cost
            
            # Calculate Realized P&L (FIFO/Avg Cost assumes Avg Cost here)
            cost_basis = qty * pos.avg_price
            realized_pnl = total_cost - cost_basis
            
            # Update Position
            pos.quantity -= qty
            if pos.quantity == 0:
                del self.state.holdings[ticker]
            
            self._log_trade(ticker, "SELL", qty, price, total_cost, realized_pnl)
            self.save()
            return f"✅ FILLED: Sold {qty} {ticker} @ ${price:.2f} (P&L: ${realized_pnl:+.2f})"

        return "❌ INVALID ACTION"

    def _log_trade(self, ticker, action, qty, price, total, pnl=None):
        record = TradeRecord(
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            action=action,
            quantity=qty,
            price=price,
            total_cost=total,
            pnl=pnl
        )
        self.state.trade_history.append(record)