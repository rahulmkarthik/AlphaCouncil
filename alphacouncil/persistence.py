import os
import json
from datetime import datetime, date
from typing import Optional, Dict, Any

CACHE_FILE = "daily_vol_cache.json"

class DailyCacheManager:
    """
    Manages a local JSON file to ensure VolSense inference 
    runs exactly ONCE per ticker per day, persisting across restarts.
    """
    
    def __init__(self):
        self.file_path = os.path.join(os.getcwd(), CACHE_FILE)
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Loads the cache from disk, handling missing or corrupt files."""
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_cache(self):
        """Writes the current cache state to disk."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except IOError as e:
            print(f"⚠️ Warning: Failed to write to cache file: {e}")

    def get_valid_entry(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Returns cached data ONLY if it exists and matches TODAY'S date.
        """
        ticker = ticker.upper()
        if ticker not in self._cache:
            return None
        
        entry = self._cache[ticker]
        cached_date = entry.get("cache_date")
        today_str = date.today().isoformat()
        
        if cached_date == today_str:
            # HIT: Data is fresh
            return entry["data"]
        else:
            # MISS: Data is stale (from yesterday)
            return None

    def store_entry(self, ticker: str, data: Dict[str, Any]):
        """
        Saves the result with today's timestamp.
        """
        ticker = ticker.upper()
        self._cache[ticker] = {
            "cache_date": date.today().isoformat(),
            "data": data
        }
        self._save_cache()

# Global singleton instance
_daily_cache = DailyCacheManager()

def get_daily_cache():
    return _daily_cache