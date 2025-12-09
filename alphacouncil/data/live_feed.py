import os
import math
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict
from volsense_inference.sector_mapping import get_sector_map
CACHE_FILE = "data/market_cache.csv"
class LiveMarketFeed:
    _instance = None
    
    def __init__(self):
        self.universe = list(get_sector_map("v507").keys())
        self._price_cache: Dict[str, float] = {}
        self._last_update = None
        self._load_from_disk()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LiveMarketFeed()
        return cls._instance
    def _load_from_disk(self):
        """Loads cache from CSV if it exists and is recent (< 24 hours)."""
        if os.path.exists(CACHE_FILE):
            try:
                mtime = os.path.getmtime(CACHE_FILE)
                file_age = datetime.now() - datetime.fromtimestamp(mtime)
                
                if file_age < timedelta(hours=24):
                    df = pd.read_csv(CACHE_FILE)
                    
                    if "ticker" in df.columns and "price" in df.columns:
                        for _, row in df.iterrows():
                            ticker = str(row["ticker"])
                            price = row["price"]
                            if pd.notna(price) and price > 0:
                                self._price_cache[ticker] = float(price)
                    
                    self._last_update = datetime.fromtimestamp(mtime)
                    print(f"📂 Loaded {len(self._price_cache)} prices from disk cache.")
                else:
                    print("⚠️ Market cache expired. Refresh required.")
            except Exception as e:
                print(f"⚠️ Failed to load market cache: {e}")
    def refresh_snapshot(self):
        """Downloads latest data and saves to disk."""
        print(f"📡 Hydrating Market Data for {len(self.universe)} tickers...")
        try:
            # Use period='5d' to get data even on weekends, auto_adjust=False for standard columns
            df = yf.download(
                self.universe, 
                period="5d", 
                auto_adjust=False,  # Explicit to avoid warning
                threads=True, 
                progress=False
            )
            
            valid_count = 0
            
            # Multi-ticker download returns MultiIndex columns: ('Close', 'NVDA'), ('Close', 'AAPL'), etc.
            if isinstance(df.columns, pd.MultiIndex):
                # Get tickers from the column level
                available_tickers = df.columns.get_level_values(1).unique().tolist()
                
                for ticker in self.universe:
                    try:
                        if ticker in available_tickers:
                            # Access using tuple: ('Close', ticker) or ('Adj Close', ticker)
                            if ('Close', ticker) in df.columns:
                                series = df[('Close', ticker)].dropna()
                            elif ('Adj Close', ticker) in df.columns:
                                series = df[('Adj Close', ticker)].dropna()
                            else:
                                continue
                                
                            if not series.empty:
                                price = float(series.iloc[-1])
                                if not math.isnan(price) and price > 0:
                                    self._price_cache[ticker] = price
                                    valid_count += 1
                    except (KeyError, IndexError) as e:
                        continue
            else:
                # Single ticker: columns are just Close, Open, etc.
                if 'Close' in df.columns:
                    series = df['Close'].dropna()
                    if not series.empty:
                        price = float(series.iloc[-1])
                        if not math.isnan(price) and price > 0:
                            self._price_cache[self.universe[0]] = price
                            valid_count = 1
            
            # SAVE TO DISK
            cache_data = [
                {"ticker": t, "price": p} 
                for t, p in self._price_cache.items() 
                if p is not None and not math.isnan(p) and p > 0
            ]
            cache_df = pd.DataFrame(cache_data)
            
            os.makedirs("data", exist_ok=True)
            cache_df.to_csv(CACHE_FILE, index=False)
            
            self._last_update = datetime.now()
            print(f"✅ Market Data Hydrated & Saved. Cached {len(cache_data)} valid prices ({valid_count} new).")
            
        except Exception as e:
            print(f"⚠️ Market Fetch Failed: {e}")
            import traceback
            traceback.print_exc()
    def get_price(self, ticker: str) -> Optional[float]:
        ticker = ticker.upper()
        if not self._price_cache:
            self.refresh_snapshot()
        
        price = self._price_cache.get(ticker)
        
        if price is None:
            return None
        if isinstance(price, float) and (math.isnan(price) or price <= 0):
            return None
            
        return price