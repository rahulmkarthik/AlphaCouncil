from dataclasses import dataclass

@dataclass
class RiskLimits:
    MAX_SECTOR_EXPOSURE: float = 0.30  # Max 30% in one sector (e.g., Tech)
    MAX_SINGLE_POSITION: float = 0.10  # Max 10% in one ticker
    MIN_CASH_BUFFER: float = 5000.0    # Always keep $5k cash
    MAX_DAILY_DRAWDOWN: float = 0.02   # 2% portfolio stop-loss (future scope)

    # Allow specific exceptions (optional)
    SECTOR_EXCEPTIONS = {
        "Index/ETF": 0.50  # Allow 50% in SPY/QQQ
    }

# Global Singleton for easier access by agents
DEFAULT_LIMITS = RiskLimits()