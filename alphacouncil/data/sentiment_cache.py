"""
Sentiment Cache for Fundamentalist Agent
========================================
TTL-based caching for expensive LLM-derived sentiment analysis.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from alphacouncil.schema import SectorIntel


class SentimentCache:
    """
    Simple TTL-based cache for Fundamentalist sentiment results.
    
    - Key: ticker symbol (uppercase)
    - Value: SectorIntel object
    - TTL: 30 minutes default (news doesn't change that fast)
    """
    
    _instance = None
    
    def __init__(self, ttl_minutes: int = 90):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
    
    @classmethod
    def get_instance(cls) -> "SentimentCache":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = SentimentCache()
        return cls._instance
    
    def get(self, ticker: str) -> Optional[SectorIntel]:
        """
        Get cached sentiment for a ticker if not expired.
        Returns None if not in cache or expired.
        """
        ticker = ticker.upper()
        entry = self._cache.get(ticker)
        
        if entry is None:
            return None
        
        # Check expiry
        if datetime.now() > entry["expires_at"]:
            del self._cache[ticker]
            return None
        
        return entry["intel"]
    
    def set(self, ticker: str, intel: SectorIntel) -> None:
        """Store sentiment result with TTL."""
        ticker = ticker.upper()
        self._cache[ticker] = {
            "intel": intel,
            "cached_at": datetime.now(),
            "expires_at": datetime.now() + self._ttl
        }
    
    def clear(self, ticker: Optional[str] = None) -> None:
        """Clear cache for a ticker or all tickers."""
        if ticker:
            self._cache.pop(ticker.upper(), None)
        else:
            self._cache.clear()
    
    def get_cache_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get cache metadata (for debugging)."""
        ticker = ticker.upper()
        entry = self._cache.get(ticker)
        if entry:
            return {
                "cached_at": entry["cached_at"].isoformat(),
                "expires_at": entry["expires_at"].isoformat(),
                "ttl_remaining": (entry["expires_at"] - datetime.now()).total_seconds()
            }
        return None


# Convenience functions
def get_cached_sentiment(ticker: str) -> Optional[SectorIntel]:
    """Get cached sentiment for a ticker."""
    return SentimentCache.get_instance().get(ticker)


def cache_sentiment(ticker: str, intel: SectorIntel) -> None:
    """Cache sentiment for a ticker."""
    SentimentCache.get_instance().set(ticker, intel)


def clear_sentiment_cache(ticker: Optional[str] = None) -> None:
    """Clear sentiment cache."""
    SentimentCache.get_instance().clear(ticker)
