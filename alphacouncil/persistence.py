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
        self.today = date.today().isoformat()
        self.filename = f"vol_cache_{self.today}.json"
        self.file_path = os.path.join(LOG_DIR, self.filename)
        
        print(f"[DEBUG] Active Cache File: {self.file_path}")
        self._ensure_file_exists()
        self._cache = self._load_cache()

    def _ensure_file_exists(self):
        """Creates a fresh log file for the new day if missing."""
        if not os.path.exists(self.file_path):
            print(f"[DEBUG] New day detected. Creating {self.filename}...")
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
            print(f"[DEBUG] ✅ Logged {len(self._cache)} entries to {self.filename}")
        except Exception as e:
            print(f"[ERROR] Failed to write log: {e}")

    def get_valid_entry(self, ticker: str) -> Optional[Dict[str, Any]]:
        # We only check the current day's file. 
        # If you need historical lookup later, we can add a specific method for that.
        return self._cache.get(ticker.upper())

    def store_entry(self, ticker: str, data: Dict[str, Any]):
        # No need to store "cache_date" inside the entry anymore
        # because the filename itself acts as the timestamp.
        self._cache[ticker.upper()] = data
        self._save_cache()

    def _today_str(self):
        return self.today

_daily_cache = DailyCacheManager()
def get_daily_cache(): return _daily_cache