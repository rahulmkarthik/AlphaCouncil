# volsense_pkg/data_fetching/data_utils.py
import os
import pandas as pd
from pathlib import Path
from .fetch import build_dataset

DATA_CACHE = Path(os.getenv("VOLSENSE_DATA", "./.volsense_cache"))
DATA_CACHE.mkdir(parents=True, exist_ok=True)


def save_to_cache(df: pd.DataFrame, name: str):
    """
    Save a DataFrame to local parquet cache.

    :param df: DataFrame to be saved.
    :type df: pandas.DataFrame
    :param name: Logical cache key (filename without extension).
    :type name: str
    :return: Filesystem path of the saved parquet file.
    :rtype: pathlib.Path
    """
    path = DATA_CACHE / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_from_cache(name: str) -> pd.DataFrame:
    """
    Load a DataFrame from parquet cache if it exists.

    :param name: Logical cache key (filename without extension).
    :type name: str
    :raises FileNotFoundError: If no cached parquet file exists for the given name.
    :return: Cached DataFrame loaded from parquet.
    :rtype: pandas.DataFrame
    """
    path = DATA_CACHE / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"No cache found for {name}")


def get_or_fetch_single(ticker: str, start="2000-01-01", end=None, use_cache=True):
    """
    Fetch OHLCV and realized volatility for a single ticker with optional caching.

    On cache hit, returns the cached parquet. Otherwise, downloads data and builds the dataset,
    then writes it to cache (if enabled) before returning.

    :param ticker: Ticker symbol to fetch.
    :type ticker: str
    :param start: Start date (YYYY-MM-DD) for the time series.
    :type start: str
    :param end: End date (YYYY-MM-DD). If None, fetches up to the latest available date.
    :type end: str, optional
    :param use_cache: Whether to read/write from local cache.
    :type use_cache: bool
    :raises ValueError: If the underlying fetch/build returns no valid data.
    :return: Long-form dataset with ['date','ticker','return','realized_vol'].
    :rtype: pandas.DataFrame
    """
    cache_name = f"{ticker}_{start}_{end}".replace(":", "-")
    if use_cache:
        try:
            return load_from_cache(cache_name)
        except FileNotFoundError:
            pass

    df = build_dataset(ticker, start=start, end=end)
    if use_cache:
        save_to_cache(df, cache_name)
    return df


def get_or_fetch_multi(
    tickers, start="2000-01-01", end=None, lookback=21, use_cache=True
):
    """
    Fetch OHLCV and realized volatility for multiple tickers with optional caching.

    On cache hit, returns the cached parquet. Otherwise, downloads data for all tickers and
    builds a unified dataset, then writes it to cache (if enabled) before returning.

    :param tickers: Collection of ticker symbols to fetch.
    :type tickers: list[str] or tuple[str, ...]
    :param start: Start date (YYYY-MM-DD) for the time series.
    :type start: str
    :param end: End date (YYYY-MM-DD). If None, fetches up to the latest available date.
    :type end: str, optional
    :param lookback: Rolling window length used to compute realized volatility.
    :type lookback: int
    :param use_cache: Whether to read/write from local cache.
    :type use_cache: bool
    :raises ValueError: If the underlying fetch/build returns no valid data.
    :return: Long-form dataset with ['date','ticker','return','realized_vol'] across all tickers.
    :rtype: pandas.DataFrame
    """
    cache_name = f"{'_'.join(tickers)}_{start}_{end}_{lookback}".replace(":", "-")
    if use_cache:
        try:
            return load_from_cache(cache_name)
        except FileNotFoundError:
            pass

    df = build_dataset(tickers=tickers, window=lookback)
    if use_cache:
        save_to_cache(df, cache_name)
    return df


# ============================================================
# 🔁 make_rolling_windows: Generate rolling subwindows
# ============================================================


def make_rolling_windows(df: pd.DataFrame, window: int = 30, stride: int = 5):
    """
    Generate rolling subwindows of a DataFrame for evaluation or backtesting.

    The function creates overlapping slices of length `window`, advancing the start
    index by `stride` each step. Assumes the DataFrame includes a 'date' column.

    :param df: Input time series; must include a 'date' column sorted ascending.
    :type df: pandas.DataFrame
    :param window: Length of each rolling window.
    :type window: int
    :param stride: Step size between window start indices.
    :type stride: int
    :raises ValueError: If 'date' column is missing from the input DataFrame.
    :return: List of rolling DataFrame segments, each of length `window` (except potentially the last if filtered elsewhere).
    :rtype: list[pandas.DataFrame]
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must include a 'date' column for rolling windows.")

    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    windows = []

    for start in range(0, n - window + 1, stride):
        sub = df.iloc[start : start + window].copy()
        windows.append(sub)

    return windows
