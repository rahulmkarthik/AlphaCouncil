import os
import pandas as pd
import json
import builtins
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field

from alphacouncil.persistence import get_daily_cache
from alphacouncil.utils.langchain_stub import tool

# --- MONKEY PATCH ---
try:
    display
except NameError:
    def display(obj): pass
    builtins.display = display
# --------------------

from volsense_inference.forecast_engine import Forecast
from volsense_inference.signal_engine import SignalEngine
from volsense_inference.sector_mapping import get_sector_map, get_ticker_type_map


class VolSenseService:
    _instance = None
    _forecast_engine: Optional[Forecast] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VolSenseService()
        return cls._instance

    def __init__(self):
        self.model_version = "volnetx"
        # UPGRADE: Target the full 507-ticker universe
        self.universe_map = get_sector_map("v507") 
        self.checkpoints_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            "modules", "VolSense", "models"
        )
        
    def _ensure_loaded(self):
        if self._forecast_engine is None:
            print(f"🔄 VolSenseService: Loading model {self.model_version}...")
            self._forecast_engine = Forecast(
                model_version=self.model_version,
                checkpoints_dir=self.checkpoints_dir,
                start="2015-01-01"
            )

    def hydrate_market(self):
        """
        Runs batch inference on the full v507 universe and logs to today's file.
        """
        self._ensure_loaded()
        cache = get_daily_cache()
        tickers = list(self.universe_map.keys())
        
        print(f"🌊 HYDRATING MARKET: Batch inference on {len(tickers)} tickers (volnetx)...")
        
        # 1. BATCH PREDICTION (One big network call)
        preds = self._forecast_engine.run(tickers=tickers)
        
        # 2. GLOBAL SIGNALS (Z-scores now relative to 500+ peers)
        sig = SignalEngine(model_version=self.model_version)
        sig.set_data(preds)
        sig.compute_signals(enrich_with_sectors=True)
        
        type_map = get_ticker_type_map(self.model_version)
        full_hist = self._forecast_engine.df_recent
        
        print("💾 Serializing to daily log...")
        
        count = 0
        for ticker in tickers:
            # Data Quality Check
            row_slice = sig.signals[sig.signals["ticker"] == ticker]
            if row_slice.empty:
                continue
            
            # Get z-scores for each horizon
            z_scores_by_horizon = {}
            for h in [1, 5, 10]:
                h_row = row_slice[row_slice["horizon"] == h]
                if not h_row.empty:
                    z_scores_by_horizon[h] = round(float(h_row.iloc[0].get("vol_zscore", 0)), 2)
                else:
                    z_scores_by_horizon[h] = 0.0
            
            # Use horizon=5 as the primary row for other metrics (most common trading horizon)
            row = row_slice[row_slice["horizon"] == 5].iloc[0] if not row_slice[row_slice["horizon"] == 5].empty else row_slice.iloc[0]
            
            # Serialize History (180 days)
            t_hist = full_hist[full_hist["ticker"] == ticker].sort_values("date").tail(180)
            history_json = [
                {"date": r["date"].strftime("%Y-%m-%d"), 
                 "realized_vol": float(r["realized_vol"]) if pd.notna(r["realized_vol"]) else None}
                for _, r in t_hist.iterrows()
            ]
            
            # Serialize Forecasts
            forecast_levels = {}
            for h in [1, 5, 10]:
                col = f"pred_vol_{h}"
                if col in preds.columns:
                    val = preds.loc[preds["ticker"] == ticker, col].values[0]
                    forecast_levels[str(h)] = float(val) if pd.notna(val) else None

            payload = {
                "ticker": ticker,
                "type": type_map.get(ticker, "Equity"),
                "sector": str(row.get("sector", "Unknown")),
                
                # --- NEW: Explicit Signal Block ---
                "signal": {
                    "position": str(row.get("position", "NEUTRAL")),
                    "action": str(row.get("action", "Wait")),
                    "strength": float(row.get("signal_strength", 0.0))
                },
                # ----------------------------------

                "metrics": {
                    "current_vol": round(float(row.get("today_vol", 0)), 4),
                    # ... (keep existing 1d/5d/10d forecasts) ...
                    "forecast_1d": round(float(row.get("forecast_vol_1", preds.loc[preds["ticker"] == ticker, "pred_vol_1"].values[0] if "pred_vol_1" in preds.columns else 0)), 4),
                    "forecast_5d": round(float(row.get("forecast_vol", 0)), 4),
                    "forecast_10d": round(float(row.get("forecast_vol_10", preds.loc[preds["ticker"] == ticker, "pred_vol_10"].values[0] if "pred_vol_10" in preds.columns else 0)), 4),
                    "vol_spread_pct": round(float(row.get("vol_spread", 0)), 4),
                    "z_score": z_scores_by_horizon.get(5, 0.0),  # Default to 5d for backward compat
                    "z_score_1d": z_scores_by_horizon.get(1, 0.0),
                    "z_score_5d": z_scores_by_horizon.get(5, 0.0),
                    "z_score_10d": z_scores_by_horizon.get(10, 0.0),
                    "term_spread_10v5": round(float(row.get("term_spread_10v5", 0)), 4),
                    
                    # --- NEW: Add Momentum for Context ---
                    "momentum_5d": round(float(row.get("momentum_5d", 0)), 4),
                    "momentum_20d": round(float(row.get("momentum_20d", 0)), 4),
                },
                "context": {
                    "regime": str(row.get("regime_flag", "Normal")),
                    "sector_z_score": round(float(row.get("sector_z", 0)), 2),
                    "rank_in_sector": round(float(row.get("rank_sector", 0.5)), 2),
                    "heuristic_signal": str(row.get("position", "neutral"))
                },
                "plot_data": {
                    "history": history_json,
                    "forecasts": forecast_levels
                }
            }
            
            # Update memory cache
            cache._cache[ticker] = payload
            count += 1
            
        # One atomic write to disk
        cache._save_cache()
        print(f"✅ Market Hydration Complete. Logged {count} tickers.")


    def get_rich_data(self, ticker: str) -> dict:
        cache = get_daily_cache()
        
        # 1. CHECK TODAY'S LOG
        cached_result = cache.get_valid_entry(ticker)
        if cached_result:
            return cached_result

        # 2. CACHE MISS -> TRIGGER BATCH HYDRATION
        # Use the map to check validity before running a 1-minute job
        if ticker in self.universe_map:
            print(f"🐢 CACHE MISS for {ticker}. Triggering Market Hydration...")
            self.hydrate_market()
            
            # 3. RE-CHECK
            cached_result = cache.get_valid_entry(ticker)
            if cached_result:
                return cached_result
        
        # 4. FAILURE
        return {"error": f"Ticker {ticker} not in v507 universe or hydration failed."}


def _fetch_vol_payload(ticker: str) -> str:
    service = VolSenseService.get_instance()
    try:
        data = service.get_rich_data(ticker.upper())
        return json.dumps(data)
    except Exception as exc:  # pragma: no cover - defensive path
        return json.dumps({"error": str(exc)})

class TickerInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g. 'NVDA')")

@tool("get_vol_metrics", args_schema=TickerInput)
def get_vol_metrics(ticker: str) -> str:
    """
    Returns RICH volatility metrics including spreads, sector ranks, and heuristics.
    Use this to determine directional volatility exposure.
    """
    return _fetch_vol_payload(ticker)


@tool("get_volatility_forecast", args_schema=TickerInput)
def get_volatility_forecast(ticker: str) -> str:
    """Compatibility wrapper mirroring the original forecast tool name."""
    return _fetch_vol_payload(ticker)


@tool("get_sector_trends")
def get_sector_trends() -> str:
    """Summarize cached sector signals from the latest hydration run."""

    cache = get_daily_cache()
    if not getattr(cache, "_cache", {}):
        try:
            VolSenseService.get_instance().hydrate_market()
        except Exception as exc:  # pragma: no cover - defensive path
            return json.dumps({"error": str(exc)})

    if not cache._cache:
        return json.dumps({"error": "No volatility cache available for today."})

    sector_stats: dict[str, list[float]] = defaultdict(list)
    signal_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for payload in cache._cache.values():
        sector = payload.get("sector", "Unknown")
        signal_block = payload.get("signal", {}) or {}
        strength = signal_block.get("strength")
        position = (signal_block.get("position") or "NEUTRAL").upper()

        if strength is not None:
            try:
                sector_stats[sector].append(float(strength))
            except (TypeError, ValueError):
                pass

        signal_counts[sector][position] += 1

    summary = []
    for sector, strengths in sector_stats.items():
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
        counts = signal_counts.get(sector, {})
        summary.append(
            {
                "sector": sector,
                "avg_signal_strength": round(avg_strength, 4),
                "positions": counts,
                "observation_count": sum(counts.values()),
            }
        )

    # Sort sectors by descending signal strength for readability
    summary.sort(key=lambda item: item["avg_signal_strength"], reverse=True)

    return json.dumps({"sectors": summary})