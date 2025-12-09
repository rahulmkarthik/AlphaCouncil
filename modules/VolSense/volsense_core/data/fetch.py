import yfinance as yf
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from typing import Union, List, Dict


# ============================================================
# 🌍 Unified Fetch Function
# ============================================================
def fetch_ohlcv(
    tickers: Union[str, List[str]],
    start: str = "2000-01-01",
    end: str | None = None,
    show_progress: bool = True,
    cache_dir: str | None = None,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Fetch OHLCV data for one or more tickers from Yahoo Finance.

    :param tickers: Single ticker symbol or list of tickers.
    :type tickers: str or list[str]
    :param start: Start date for the download (YYYY-MM-DD).
    :type start: str
    :param end: End date for the download (YYYY-MM-DD). If None, fetches up to the latest available date.
    :type end: str, optional
    :param show_progress: Whether to display a tqdm progress bar for multi-ticker requests.
    :type show_progress: bool
    :param cache_dir: Optional directory to cache each ticker’s data as CSV. If provided and a cache exists, it will be reused.
    :type cache_dir: str, optional
    :return: For a single ticker, a DataFrame with columns ['date','open','high','low','close','adj_close','volume']. For multiple tickers, a dict mapping ticker to its DataFrame.
    :rtype: pandas.DataFrame or dict[str, pandas.DataFrame]
    """
    # Normalize tickers
    single_mode = isinstance(tickers, str)
    tickers = (
        [tickers]
        if single_mode
        else list(dict.fromkeys([t.upper().strip() for t in tickers]))
    )

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    iterator = (
        tqdm(tickers, desc="🌍 Fetching market data", unit="ticker")
        if show_progress and len(tickers) > 1
        else tickers
    )
    out_dict = {}

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
# 🗓️ Earnings Events Fetcher
# ============================================================
def fetch_earnings_dates(
    tickers: List[str], start_date: str, end_date: str
) -> pd.DataFrame:

    bad_tickers = []
    events = []
    
    # 1. Convert start/end to Naive Timestamps up front for safe comparison
    ts_start = pd.to_datetime(start_date).tz_localize(None)
    ts_end = pd.to_datetime(end_date).tz_localize(None)

    for t in tqdm(tickers, desc="📅 Fetching earnings", unit="ticker"):
        try:
            ticker_obj = yf.Ticker(t)
            ed = ticker_obj.earnings_dates

            if ed is None or ed.empty:
                bad_tickers.append(t)
                continue

            ed = ed.reset_index()
            # yfinance sometimes varies column names ("Earnings Date" vs "Date")
            # Rename specifically if needed, otherwise assume index reset put it in col 0 or named it 'Date'
            if "Earnings Date" in ed.columns:
                ed.rename(columns={"Earnings Date": "Date"}, inplace=True)
            elif "Event Date" in ed.columns: # Rare variation
                 ed.rename(columns={"Event Date": "Date"}, inplace=True)

            # 🚀 CRITICAL FIX: Strip Timezone (.tz_localize(None))
            ed["Date"] = pd.to_datetime(ed["Date"]).dt.tz_localize(None).dt.normalize()
            ed["Ticker"] = t
            
            # Filter rows immediately (safer than doing it at the end)
            mask = (ed["Date"] >= ts_start) & (ed["Date"] <= ts_end)
            events.append(ed.loc[mask, ["Date", "Ticker"]])
            
        except Exception:
            bad_tickers.append(t)
            continue

    if not events:
        return pd.DataFrame(columns=["date", "ticker"])

    df = pd.concat(events, ignore_index=True)
    
    # Rename for pipeline compatibility
    df = df.rename(columns={"Date": "date", "Ticker": "ticker"})

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
