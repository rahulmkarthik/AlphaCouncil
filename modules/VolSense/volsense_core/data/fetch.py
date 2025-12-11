import yfinance as yf
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, date
from tqdm import tqdm
from typing import Union, List, Dict

# ============================================================
# 📦 Daily Cache Configuration
# ============================================================
_DAILY_CACHE_DIR = Path(os.environ.get("VOLSENSE_CACHE_DIR", ".volsense_cache"))


def _get_daily_cache_path(cache_key: str) -> Path:
    """Get the path to today's daily cache file."""
    today_str = date.today().strftime("%Y%m%d")
    return _DAILY_CACHE_DIR / f"{cache_key}_{today_str}.parquet"


def _load_daily_cache(cache_key: str) -> pd.DataFrame | None:
    """Load today's cached data if it exists."""
    cache_path = _get_daily_cache_path(cache_key)
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            return None
    return None


def _save_daily_cache(df: pd.DataFrame, cache_key: str) -> None:
    """Save data to today's daily cache."""
    _DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _get_daily_cache_path(cache_key)
    try:
        df.to_parquet(cache_path, index=False)
        # Clean up old cache files (keep only last 3 days)
        _cleanup_old_caches(cache_key, keep_days=3)
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")


def _cleanup_old_caches(cache_key: str, keep_days: int = 3) -> None:
    """Remove cache files older than keep_days."""
    if not _DAILY_CACHE_DIR.exists():
        return
    today = date.today()
    for f in _DAILY_CACHE_DIR.glob(f"{cache_key}_*.parquet"):
        try:
            # Extract date from filename
            date_str = f.stem.split("_")[-1]
            file_date = datetime.strptime(date_str, "%Y%m%d").date()
            if (today - file_date).days > keep_days:
                f.unlink()
        except Exception:
            pass


# ============================================================
# 🌍 Unified Fetch Function (Optimized with Batch Download)
# ============================================================
def fetch_ohlcv(
    tickers: Union[str, List[str]],
    start: str = "2000-01-01",
    end: str | None = None,
    show_progress: bool = True,
    cache_dir: str | None = None,
    use_daily_cache: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Fetch OHLCV data for one or more tickers from Yahoo Finance.
    
    Uses batch download for multiple tickers (10-20x faster than per-ticker).
    Optionally uses daily caching to avoid re-downloading within the same day.

    :param tickers: Single ticker symbol or list of tickers.
    :type tickers: str or list[str]
    :param start: Start date for the download (YYYY-MM-DD).
    :type start: str
    :param end: End date for the download (YYYY-MM-DD). If None, fetches up to the latest available date.
    :type end: str, optional
    :param show_progress: Whether to display progress for downloads.
    :type show_progress: bool
    :param cache_dir: Optional directory to cache each ticker's data as CSV (legacy).
    :type cache_dir: str, optional
    :param use_daily_cache: If True, cache the entire batch result for the day (default: True).
    :type use_daily_cache: bool
    :return: For a single ticker, a DataFrame. For multiple tickers, a dict mapping ticker to DataFrame.
    :rtype: pandas.DataFrame or dict[str, pandas.DataFrame]
    """
    # Normalize tickers
    single_mode = isinstance(tickers, str)
    tickers = (
        [tickers]
        if single_mode
        else list(dict.fromkeys([t.upper().strip() for t in tickers]))
    )

    # --- Daily Cache Check (for batch requests) ---
    if use_daily_cache and len(tickers) > 1:
        cache_key = f"ohlcv_batch_{len(tickers)}"
        cached_df = _load_daily_cache(cache_key)
        if cached_df is not None:
            print(f"✅ Loaded {len(tickers)} tickers from daily cache")
            # Convert back to dict format
            out_dict = {}
            for tkr in tickers:
                tkr_df = cached_df[cached_df["ticker"] == tkr].drop(columns=["ticker"])
                if not tkr_df.empty:
                    out_dict[tkr] = tkr_df.reset_index(drop=True)
            if single_mode:
                return next(iter(out_dict.values())) if out_dict else pd.DataFrame()
            return out_dict

    # --- Batch Download (for multiple tickers) ---
    if len(tickers) > 1:
        print(f"🌍 Batch downloading {len(tickers)} tickers...")
        try:
            raw_df = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=False,
                progress=show_progress,
                threads=True,  # Enable multi-threading
                group_by="ticker",
            )
            
            out_dict = {}
            for tkr in tickers:
                try:
                    if isinstance(raw_df.columns, pd.MultiIndex):
                        # Multi-ticker format: columns are (Ticker, Field)
                        if tkr in raw_df.columns.get_level_values(0):
                            df_tkr = raw_df[tkr].copy()
                        else:
                            continue
                    else:
                        # Single ticker fallback
                        df_tkr = raw_df.copy()
                    
                    if df_tkr.empty:
                        continue
                    
                    # Normalize columns
                    df_tkr = df_tkr.reset_index()
                    df_tkr.columns = [c.lower().replace(" ", "_") if isinstance(c, str) else c for c in df_tkr.columns]
                    
                    # Standardize column names
                    rename_map = {"adj_close": "adj_close", "date": "date"}
                    for old, new in [("Date", "date"), ("Adj Close", "adj_close"), ("Close", "close")]:
                        old_lower = old.lower().replace(" ", "_")
                        if old_lower in df_tkr.columns:
                            rename_map[old_lower] = new
                    df_tkr = df_tkr.rename(columns=rename_map)
                    
                    if "adj_close" not in df_tkr.columns:
                        df_tkr["adj_close"] = df_tkr.get("close", np.nan)
                    
                    df_tkr["date"] = pd.to_datetime(df_tkr["date"], errors="coerce")
                    df_tkr = df_tkr.dropna(subset=["date"])
                    
                    out_dict[tkr] = df_tkr
                    
                except Exception as e:
                    print(f"⚠️ Failed processing {tkr}: {e}")
                    continue
            
            # Save to daily cache
            if use_daily_cache and out_dict:
                cache_frames = []
                for tkr, df in out_dict.items():
                    df_copy = df.copy()
                    df_copy["ticker"] = tkr
                    cache_frames.append(df_copy)
                if cache_frames:
                    combined = pd.concat(cache_frames, ignore_index=True)
                    _save_daily_cache(combined, cache_key)
                    print(f"💾 Saved {len(out_dict)} tickers to daily cache")
            
            if single_mode:
                return next(iter(out_dict.values())) if out_dict else pd.DataFrame()
            return out_dict
            
        except Exception as e:
            print(f"⚠️ Batch download failed, falling back to per-ticker: {e}")
            # Fall through to per-ticker download

    # --- Per-Ticker Download (single ticker or fallback) ---
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    out_dict = {}
    iterator = (
        tqdm(tickers, desc="🌍 Fetching market data", unit="ticker")
        if show_progress and len(tickers) > 1
        else tickers
    )

    for tkr in iterator:
        try:
            cache_path = os.path.join(cache_dir, f"{tkr}.csv") if cache_dir else None
            if cache_path and os.path.exists(cache_path):
                df = pd.read_csv(cache_path, parse_dates=["date"])
                out_dict[tkr] = df
                continue

            df = yf.download(
                tkr,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df.empty:
                print(f"⚠️ {tkr}: no data returned.")
                continue

            # --- Normalize columns ---
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.reset_index().rename(
                columns={"Date": "date", "Adj Close": "adj_close", "Close": "close"}
            )
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col not in df.columns and col.capitalize() in df.columns:
                    df.rename(columns={col.capitalize(): col}, inplace=True)

            if "adj_close" not in df.columns:
                df["adj_close"] = df.get("close", np.nan)

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])

            # Cache save
            if cache_path:
                df.to_csv(cache_path, index=False)

            out_dict[tkr] = df

        except Exception as e:
            print(f"⚠️ Failed {tkr}: {e}")
            continue

    if single_mode:
        # Return single ticker DataFrame directly
        return next(iter(out_dict.values())) if out_dict else pd.DataFrame()
    return out_dict


# ============================================================
# 🌍 Fetch Macro Data
# ============================================================


def fetch_macro_series(start_date="2000-01-01", end_date=None):
    """
    Fetch global macro proxies including Credit, Curve, and USD.
    """
    proxies = {
        "Oil": "CL=F",       # Crude Oil
        "BTC": "BTC-USD",    # Bitcoin
        "VIX": "^VIX",       # Volatility Index
        "Rates10Y": "^TNX",  # 10Y Yield
        "Rates2Y": "^IRX",   # 2Y Yield (New)
        "CreditHY": "HYG",   # High Yield Bonds (New)
        "CreditGov": "IEF",  # 7-10Y Treasuries (New)
        "USD": "DX-Y.NYB"    # US Dollar Index (New)
    }
    
    macro_data = pd.DataFrame()
    print(f"🌍 Fetching Macro Proxies: {list(proxies.keys())}...")
    
    # Batch download is often cleaner, but looping handles errors better per ticker
    for name, ticker in proxies.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if not df.empty:
                col = "Adj Close" if "Adj Close" in df.columns else "Close"
                s = df[col]
                
                # Standardize index
                s.index = s.index.tz_localize(None)
                
                # Store raw levels first (we calculate spreads/returns later)
                macro_data[f"raw_{name}"] = s
                
        except Exception as e:
            print(f"⚠️ Failed to fetch {name} ({ticker}): {e}")

    # --- 🚀 Feature Engineering (Vectorized) ---
    if not macro_data.empty:
        # 1. Existing Legacy Features
        if "raw_Oil" in macro_data: macro_data["macro_Oil"] = macro_data["raw_Oil"].pct_change()
        if "raw_BTC" in macro_data: macro_data["macro_BTC"] = macro_data["raw_BTC"].pct_change()
        if "raw_VIX" in macro_data: macro_data["macro_VIX"] = macro_data["raw_VIX"] # VIX is already a level
        if "raw_Rates10Y" in macro_data: macro_data["macro_Rates"] = macro_data["raw_Rates10Y"] # Legacy support
        
        # 2. 🚀 NEW: Yield Curve (10Y - 2Y)
        if "raw_Rates10Y" in macro_data and "raw_Rates2Y" in macro_data:
            # Yields are in percent (e.g. 4.5), simple difference works
            macro_data["macro_Curve"] = macro_data["raw_Rates10Y"] - macro_data["raw_Rates2Y"]
            
        # 3. 🚀 NEW: Credit Spread (Log Return Divergence)
        if "raw_CreditHY" in macro_data and "raw_CreditGov" in macro_data:
            # We want the relative performance. If HYG drops faster than IEF, spread widens (stress).
            # Using returns difference: Ret(HYG) - Ret(IEF)
            # Negative value = Credit Stress.
            hy_ret = macro_data["raw_CreditHY"].pct_change()
            gov_ret = macro_data["raw_CreditGov"].pct_change()
            macro_data["macro_CreditSpread"] = hy_ret - gov_ret
            
        # 4. 🚀 NEW: USD Strength
        if "raw_USD" in macro_data: 
            macro_data["macro_USD"] = macro_data["raw_USD"].pct_change()

    # Drop the intermediate "raw_" columns to keep it clean
    cols_to_keep = [c for c in macro_data.columns if c.startswith("macro_")]
    return macro_data[cols_to_keep]


# ============================================================
# 📈 Unified Dataset Builder
# ============================================================
def build_dataset(
    tickers: Union[str, List[str]],
    start: str = "2000-01-01",
    end: str | None = None,
    window: int = 15,
    cache_dir: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Fetch raw OHLCV data and build a standardized volatility dataset.

    Computes daily returns from adjusted close (if available, else close) and realized volatility
    as a rolling standard deviation over the given window, annualized by sqrt(252).

    :param tickers: Single ticker symbol or list of tickers to include.
    :type tickers: str or list[str]
    :param start: Start date for the download (YYYY-MM-DD).
    :type start: str
    :param end: End date for the download (YYYY-MM-DD). If None, fetches up to the latest available date.
    :type end: str, optional
    :param window: Rolling window length (in trading days) used to compute realized volatility.
    :type window: int
    :param cache_dir: Optional directory for CSV caching during fetch.
    :type cache_dir: str, optional
    :param show_progress: Whether to display a tqdm progress bar for multi-ticker fetches.
    :type show_progress: bool
    :raises ValueError: If no valid data is returned for any ticker.
    :return: Tidy DataFrame sorted by ['ticker','date'] with columns ['date','ticker','return','realized_vol'].
    :rtype: pandas.DataFrame
    """
    data = fetch_ohlcv(
        tickers=tickers,
        start=start,
        end=end,
        cache_dir=cache_dir,
        show_progress=show_progress,
    )

    frames = []
    if isinstance(data, pd.DataFrame):  # single ticker
        data_dict = {tickers: data}
    else:
        data_dict = data

    for tkr, df in data_dict.items():
        # Handle yfinance MultiIndex columns (new yfinance format)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
        df = df.copy()
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        df["return"] = df[price_col].pct_change()
        df["realized_vol"] = df["return"].rolling(window).std() * np.sqrt(252)
        temp = df[["date", "return", "realized_vol"]].dropna().copy()
        temp["ticker"] = tkr
        frames.append(temp)

    if not frames:
        raise ValueError("No valid data returned for any ticker.")

    dataset = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return dataset


# ============================================================
# 🗓️ Earnings Events Fetcher (Optimized with Concurrency + Cache)
# ============================================================
def _fetch_single_earnings(ticker: str, ts_start, ts_end) -> pd.DataFrame | None:
    """Fetch earnings for a single ticker (for concurrent execution)."""
    import time
    
    for attempt in range(2):  # Retry once on failure
        try:
            ticker_obj = yf.Ticker(ticker)
            ed = ticker_obj.earnings_dates

            if ed is None or ed.empty:
                return None

            ed = ed.reset_index()
            # yfinance sometimes varies column names
            if "Earnings Date" in ed.columns:
                ed.rename(columns={"Earnings Date": "Date"}, inplace=True)
            elif "Event Date" in ed.columns:
                ed.rename(columns={"Event Date": "Date"}, inplace=True)

            ed["Date"] = pd.to_datetime(ed["Date"]).dt.tz_localize(None).dt.normalize()
            ed["Ticker"] = ticker
            
            mask = (ed["Date"] >= ts_start) & (ed["Date"] <= ts_end)
            result = ed.loc[mask, ["Date", "Ticker"]]
            return result if not result.empty else None
            
        except Exception as e:
            if attempt == 0:
                time.sleep(0.5)  # Wait before retry
                continue
            return None
    return None


def fetch_earnings_dates(
    tickers: List[str], 
    start_date: str, 
    end_date: str,
    use_daily_cache: bool = True,
    max_workers: int = 5,  # Reduced from 10 to avoid rate limiting
) -> pd.DataFrame:
    """
    Fetch earnings dates for multiple tickers with concurrent execution and caching.
    
    Uses ThreadPoolExecutor for parallel fetching (10x faster than sequential).
    Optionally caches results for the day to avoid re-fetching.
    
    :param tickers: List of ticker symbols.
    :param start_date: Start date (YYYY-MM-DD).
    :param end_date: End date (YYYY-MM-DD).
    :param use_daily_cache: If True, cache results for the day.
    :param max_workers: Number of concurrent threads (default: 10).
    :return: DataFrame with columns ['date', 'ticker'].
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # --- Daily Cache Check ---
    if use_daily_cache:
        cache_key = f"earnings_{len(tickers)}"
        cached_df = _load_daily_cache(cache_key)
        if cached_df is not None:
            print(f"✅ Loaded earnings from daily cache")
            return cached_df
    
    # Convert start/end to Naive Timestamps
    ts_start = pd.to_datetime(start_date).tz_localize(None)
    ts_end = pd.to_datetime(end_date).tz_localize(None)
    
    events = []
    failed_count = 0
    
    print(f"📅 Fetching earnings for {len(tickers)} tickers (concurrent)...")
    
    # --- Concurrent Fetching ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_earnings, t, ts_start, ts_end): t 
            for t in tickers
        }
        
        # Progress tracking
        completed = 0
        total = len(futures)
        
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"   Progress: {completed}/{total} tickers")
            
            result = future.result()
            if result is not None:
                events.append(result)
            else:
                failed_count += 1

    if failed_count > 0:
        print(f"⚠️ {failed_count} tickers had no earnings data")

    if not events:
        return pd.DataFrame(columns=["date", "ticker"])

    df = pd.concat(events, ignore_index=True)
    df = df.rename(columns={"Date": "date", "Ticker": "ticker"})
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # --- Save to Daily Cache ---
    if use_daily_cache:
        _save_daily_cache(df, cache_key)
        print(f"💾 Saved earnings to daily cache")
    
    return df

