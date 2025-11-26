import os
import json
from datetime import date
from typing import Optional, Dict, Any

# FIX: Use the Current Working Directory (where you run the command from)
# This ensures the file appears right next to your .env and requirements.txt
ROOT_DIR = os.getcwd() 
CACHE_FILE = os.path.join(ROOT_DIR, "daily_vol_cache.json")

class DailyCacheManager:
    def __init__(self):
        self.file_path = CACHE_FILE
        self._ensure_file_exists()
        self._cache = self._load_cache()

    def _ensure_file_exists(self):
        """Forces the file to exist on disk immediately."""
        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, "w") as f:
                    json.dump({}, f)
            except Exception as e:
                print(f"[ERROR] ❌ Could not create cache file: {e}")

    def _load_cache(self) -> Dict[str, Any]:
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_cache(self):
        try:
            with open(self.file_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            print(f"[ERROR] ❌ Failed to write cache: {e}")

    def get_valid_entry(self, ticker: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(ticker.upper())
        if entry and entry.get("cache_date") == date.today().isoformat():
            return entry["data"]
        return None

    def store_entry(self, ticker: str, data: Dict[str, Any]):
        self._cache[ticker.upper()] = {
            "cache_date": date.today().isoformat(),
            "data": data
        }
        self._save_cache()

_daily_cache = DailyCacheManager()
def get_daily_cache(): return _daily_cache