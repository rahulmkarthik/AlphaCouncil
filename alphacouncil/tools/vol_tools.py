import os
import json
import builtins
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from alphacouncil.persistence import get_daily_cache

# --- MONKEY PATCH ---
try:
    display
except NameError:
    def display(obj): pass
    builtins.display = display
# --------------------

from volsense_inference.forecast_engine import Forecast
from volsense_inference.signal_engine import SignalEngine
from volsense_inference.sector_mapping import get_ticker_type_map

class VolSenseService:
    _instance = None
    _forecast_engine: Optional[Forecast] = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VolSenseService()
        return cls._instance

    def __init__(self):
        self.model_version = "v507"
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
                start="2010-01-01"
            )

    # UPDATED: No more @lru_cache. We use file persistence instead.
    def get_rich_data(self, ticker: str) -> dict:
        cache = get_daily_cache()
        
        # 1. CHECK CACHE FIRST (Speed: Instant)
        cached_result = cache.get_valid_entry(ticker)
        if cached_result:
            print(f"⚡ CACHE HIT: Returning saved data for {ticker}")
            return cached_result

        # 2. CACHE MISS: Run Heavy Inference (Speed: Slow)
        print(f"🐢 CACHE MISS: Running VolSense Inference for {ticker}...")
        self._ensure_loaded()
        
        # Inject Reference Universe
        universe = list(set([ticker, "SPY", "QQQ", "VXX", "GLD", "TLT"]))
        preds = self._forecast_engine.run(tickers=universe)
        
        sig = SignalEngine(model_version=self.model_version)
        sig.set_data(preds)
        sig.compute_signals(enrich_with_sectors=True)
        
        if ticker not in sig.signals["ticker"].values:
             return {"error": f"No signal generated for {ticker}"}

        row = sig.signals[sig.signals["ticker"] == ticker].iloc[0]
        type_map = get_ticker_type_map(self.model_version)
        asset_type = type_map.get(ticker, "Equity")

        # Build Payload
        result_payload = {
            "ticker": ticker,
            "type": asset_type,
            "sector": str(row.get("sector", "Unknown")),
            "metrics": {
                "current_vol": round(float(row.get("today_vol", 0)), 4),
                "forecast_5d": round(float(row.get("forecast_vol", 0)), 4),
                "vol_spread_pct": round(float(row.get("vol_spread", 0)), 4),
                "z_score": round(float(row.get("vol_zscore", 0)), 2),
                "term_spread_10v5": round(float(row.get("term_spread_10v5", 0)), 4),
            },
            "context": {
                "regime": str(row.get("regime_flag", "Normal")),
                "sector_z_score": round(float(row.get("sector_z", 0)), 2),
                "rank_in_sector": round(float(row.get("rank_sector", 0.5)), 2),
                "heuristic_signal": str(row.get("position", "neutral"))
            }
        }

        # 3. SAVE TO DISK
        cache.store_entry(ticker, result_payload)
        
        return result_payload

class TickerInput(BaseModel):
    ticker: str = Field(description="The stock ticker symbol (e.g. 'NVDA')")

@tool("get_vol_metrics", args_schema=TickerInput)
def get_vol_metrics(ticker: str) -> str:
    """
    Returns RICH volatility metrics including spreads, sector ranks, and heuristics.
    Use this to determine directional volatility exposure.
    """
    service = VolSenseService.get_instance()
    try:
        data = service.get_rich_data(ticker.upper())
        return json.dumps(data) 
    except Exception as e:
        return json.dumps({"error": str(e)})