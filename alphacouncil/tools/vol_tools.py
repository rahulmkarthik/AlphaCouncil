import os
import pandas as pd
import json
import builtins
from typing import Optional, List
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
        self.model_version = "v507" 
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
        
        print(f"🌊 HYDRATING MARKET: Batch inference on {len(tickers)} tickers (v507)...")
        
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
                
            row = row_slice.iloc[0]
            
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
                "metrics": {
                    "current_vol": round(float(row.get("today_vol", 0)), 4),
                    "forecast_1d": round(float(row.get("forecast_vol_1", 
                                     preds.loc[preds["ticker"] == ticker, "pred_vol_1"].values[0] 
                                     if "pred_vol_1" in preds.columns else 0)), 4),
                    "forecast_5d": round(float(row.get("forecast_vol", 0)), 4),
                    "forecast_10d": round(float(row.get("forecast_vol_10", 
                                      preds.loc[preds["ticker"] == ticker, "pred_vol_10"].values[0] 
                                      if "pred_vol_10" in preds.columns else 0)), 4),
                    "vol_spread_pct": round(float(row.get("vol_spread", 0)), 4),
                    "z_score": round(float(row.get("vol_zscore", 0)), 2),
                    "term_spread_10v5": round(float(row.get("term_spread_10v5", 0)), 4),
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
        print(f"🐢 CACHE MISS for {ticker}. Triggering Market Hydration...")
        self.hydrate_market()
        
        # 3. RE-CHECK
        cached_result = cache.get_valid_entry(ticker)
        if cached_result:
            return cached_result
        
        return {"error": f"Ticker {ticker} not in v507 universe."}

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