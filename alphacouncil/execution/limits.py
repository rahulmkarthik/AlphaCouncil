"""Shared helpers for enforcing portfolio exposure limits."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Optional

from alphacouncil.execution.risk_rules import DEFAULT_LIMITS
from alphacouncil.data.live_feed import LiveMarketFeed
from volsense_inference.sector_mapping import get_sector_map  # type: ignore

PriceLookup = Callable[[str], Optional[float]]

_market_feed = LiveMarketFeed.get_instance()
_sector_map = get_sector_map("v507")


def _lookup_sector(ticker: str) -> str:
    return _sector_map.get(ticker.upper(), "Unknown")


def _portfolio_exposure_snapshot(state, price_lookup: PriceLookup):
    sector_totals = defaultdict(float)
    holdings_val = 0.0
    existing_values: Dict[str, float] = {}

    for symbol, pos in state.holdings.items():
        live_price = price_lookup(symbol) or pos.avg_price
        value = pos.quantity * live_price
        sector = _lookup_sector(symbol)
        sector_totals[sector] += value
        holdings_val += value
        existing_values[symbol.upper()] = value

    return holdings_val, sector_totals, existing_values


def compute_position_headroom(
    ticker: str,
    price: float,
    state=None,
    price_lookup: Optional[PriceLookup] = None,
):
    """Return sizing capacity respecting cash buffer, sector, and single-name caps."""
    if state is None:
        from alphacouncil.execution.portfolio import PortfolioService

        state = PortfolioService().get_state()

    price_lookup = price_lookup or _market_feed.get_price

    ticker = ticker.upper()
    holdings_val, sector_totals, existing_values = _portfolio_exposure_snapshot(
        state, price_lookup
    )

    total_equity = state.cash_balance + holdings_val
    sector = _lookup_sector(ticker)
    sector_limit_pct = DEFAULT_LIMITS.SECTOR_EXCEPTIONS.get(
        sector, DEFAULT_LIMITS.MAX_SECTOR_EXPOSURE
    )
    current_sector_value = sector_totals[sector]
    existing_position_value = existing_values.get(ticker, 0.0)

    cash_balance = state.cash_balance
    cash_available = max(cash_balance - DEFAULT_LIMITS.MIN_CASH_BUFFER, 0.0)

    # FIX: Check for NaN or invalid price before division
    import math
    if price <= 0 or math.isnan(price):
        cash_max_qty = 0
        sector_max_qty = 0
        single_max_qty = 0
        sector_value_room = 0.0
        single_value_room = 0.0
    else:
        cash_max_qty = int(cash_available / price)
        if total_equity > 0:
            sector_value_room = max(
                sector_limit_pct * total_equity - current_sector_value, 0.0
            )
            sector_max_qty = int(sector_value_room / price)
            single_value_room = max(
                DEFAULT_LIMITS.MAX_SINGLE_POSITION * total_equity
                - existing_position_value,
                0.0,
            )
            single_max_qty = int(single_value_room / price)
        else:
            sector_value_room = 0.0
            single_value_room = 0.0
            sector_max_qty = int(cash_available / price)
            single_max_qty = int(cash_available / price)

    max_qty = max(0, min(cash_max_qty, sector_max_qty, single_max_qty))

    return {
        "cash_balance": cash_balance,
        "cash_available_for_trade": cash_available,
        "cash_max_qty": max(cash_max_qty, 0),
        "sector": sector,
        "sector_limit_pct": sector_limit_pct,
        "sector_value_room": sector_value_room,
        "sector_max_qty": max(sector_max_qty, 0),
        "current_sector_value": current_sector_value,
        "total_equity": total_equity,
        "holdings_value": holdings_val,
        "existing_position_value": existing_position_value,
        "single_position_limit_pct": DEFAULT_LIMITS.MAX_SINGLE_POSITION,
        "single_position_value_room": single_value_room,
        "single_position_max_qty": max(single_max_qty, 0),
        "max_qty": max_qty,
    }
