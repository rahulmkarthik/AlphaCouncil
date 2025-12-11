import os
import json
from datetime import date
from typing import Optional, Dict, Any

# 1. Create a 'logs' directory to keep things tidy
ROOT_DIR = os.getcwd()
LOG_DIR = os.path.join(ROOT_DIR, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

class DailyCacheManager:
    def __init__(self):
        # Dynamic Filename: vol_cache_YYYY-MM-DD.json
        self._init_date = date.today().isoformat()
        self.filename = f"vol_cache_{self._init_date}.json"
        self.file_path = os.path.join(LOG_DIR, self.filename)
        
        self._ensure_file_exists()
        self._cache = self._load_cache()

    def _check_date_change(self) -> bool:
        """Check if the date has changed since initialization and refresh if needed."""
        current_date = date.today().isoformat()
        if current_date != self._init_date:
            print(f"📅 Date changed from {self._init_date} to {current_date} - refreshing cache")
            self._init_date = current_date
            self.filename = f"vol_cache_{self._init_date}.json"
            self.file_path = os.path.join(LOG_DIR, self.filename)
            self._ensure_file_exists()
            self._cache = self._load_cache()
            return True
        return False
    
    def is_stale(self) -> bool:
        """Check if cache is from a previous day."""
        return date.today().isoformat() != self._init_date
    
    def get_cache_date(self) -> str:
        """Return the date of the current cache."""
        return self._init_date

    def _ensure_file_exists(self):
        """Creates a fresh log file for the new day if missing."""
        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, "w") as f:
                    json.dump({}, f)
            except Exception as e:
                print(f"[ERROR] Could not create log file: {e}")

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
            print(f"[ERROR] Failed to write log: {e}")

    def get_valid_entry(self, ticker: str) -> Optional[Dict[str, Any]]:
        # Check for date change before returning cached data
        self._check_date_change()
        return self._cache.get(ticker.upper())

    def store_entry(self, ticker: str, data: Dict[str, Any]):
        # Check for date change before storing
        self._check_date_change()
        self._cache[ticker.upper()] = data
        self._save_cache()
    
    def clear(self):
        """Clear the in-memory cache (forces re-hydration on next access)."""
        self._cache = {}
        self._save_cache()

    def _today_str(self):
        return self._init_date

_daily_cache = DailyCacheManager()
def get_daily_cache(): return _daily_cache