"""
volsense_inference.forecast_engine
=================================

High-level forecasting runtime for VolSense.

This module exposes the Forecast runtime class which orchestrates data fetching,
feature engineering, batch inference against a pretrained VolSense model, and
lightweight analytics and visualization helpers.

Primary class
-------------
Forecast
    - Loads a pretrained model and its artifacts (scalers, metadata, feature list)
      via volsense_inference.model_loader.load_model.
    - Prepares recent market data using volsense_core.data.fetch.build_dataset
      and volsense_core.data.feature_engineering.build_features to ensure the
      inference inputs match the training-time schema.
    - Runs batched inference using volsense_inference.predictor.predict_batch,
      attaches realized volatility values, and exposes an Analytics helper for
      cross-sectional signal computations and simple plots.

Usage example
-------------
>>> from volsense_inference.forecast_engine import Forecast
>>> f = Forecast(model_version="v507", checkpoints_dir="../models", start="2020-01-01")
>>> preds = f.run(["SPY", "QQQ"])
>>> f.plot("SPY")

Notes
-----
- The Forecast class expects model artifacts to be available in the checkpoints
  directory (meta + state or supported bundle format). See
  volsense_inference.model_loader for loading behavior and supported formats.
- Forecast.plot produces a matplotlib Figure when show=False, otherwise it
  renders and closes the figure to avoid duplicate captures in notebook contexts.

"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from volsense_inference.model_loader import load_model
from volsense_inference.predictor import predict_batch, attach_realized
from volsense_core.data.fetch import build_dataset
from volsense_core.data.feature_engineering import build_features
from volsense_inference.analytics import Analytics


class Forecast:
    """
    High-level runtime interface for volatility forecasting and visualization.

    Orchestrates:
      - recent data fetch and feature engineering aligned to the trained model,
      - batched inference for one or more tickers,
      - attachment of realized volatility for the snapshot date,
      - light analytics and plotting helpers.

    Attributes
    ----------
    model_version : str
        Version tag of the pretrained model assets to load.
    checkpoints_dir : str
        Directory containing model checkpoints and metadata.
    start : str
        Earliest date used when fetching recent data to build features.
    model : object
        Loaded model ready for batch prediction.
    meta : dict
        Model metadata (e.g., horizons, window).
    scalers : dict
        Per-ticker scalers used during training.
    ticker_to_id : dict
        Mapping of ticker symbols to integer IDs.
    features : list[str]
        Feature columns expected by the model.
    window : int
        Input window length used by the model.
    vol_window : int
        Lookback window for realized volatility computation in feature prep.
    horizons : list[int]
        Forecast horizons supported by the model.
    predictions : pandas.DataFrame | None
        Last forecast snapshot produced by run().
    signals : Analytics
        Analytics helper attached after run().
    """

    def __init__(
        self,
        model_version: str = "v109",
        checkpoints_dir: str = "models",
        start: str = "2005-01-01",
    ):
        """
        Initialize the Forecast runtime by loading pretrained assets.

        :param model_version: Model version tag to load (e.g., 'v109').
        :type model_version: str
        :param checkpoints_dir: Directory containing model checkpoints and metadata.
        :type checkpoints_dir: str
        :param start: Start date for fetching recent data to build features.
        :type start: str
        """
        print(f"🚀 Initializing VolSense.Forecast (model={model_version})")
        self.model_version = model_version
        self.checkpoints_dir = checkpoints_dir
        self.start = start

        # Load model and assets
        self.model, self.meta, self.scalers, self.ticker_to_id, self.features = (
            load_model(model_version=model_version, checkpoints_dir=checkpoints_dir)
        )

        self.window = self.meta.get("window", 40)
        self.vol_window = (
            self.meta.get("vol_window")
            or self.meta.get("rv_window")
            or self.meta.get("realized_window")
            or 15
        )
        self.horizons = self.meta.get("horizons", [1])
        
        # Validate VolNetX has correct window size
        if "volnetx" in self.model_version.lower() and self.window < 65:
            raise ValueError(
                f"VolNetX requires window >= 65, but got {self.window}. "
                f"Check model checkpoint metadata at {checkpoints_dir}/{model_version}.meta.json"
            )
        
        print(f"✔ Window={self.window}, Horizons={self.horizons}")

        self.predictions = None

    # ------------------------------------------------------------------
    # Data & Forecasting
    # ------------------------------------------------------------------
    def _prepare_data(self, tickers):
        """
        Build a recent, feature-engineered dataset aligned to the trained model.

        Steps:
          1) Fetch recent OHLCV and compute realized volatility with the model's vol_window.
          2) Apply feature engineering to match training-time feature schema.

        :param tickers: Single ticker or list of tickers to prepare data for.
        :type tickers: str or list[str]
        :return: Recent feature DataFrame used for inference.
        :rtype: pandas.DataFrame
        """
        end_date = datetime.today().date()
        # Increased to 200 days to ensure sufficient data for VolNetX (65-day window + 60-day rolling features)
        start_date = (end_date - timedelta(days=200)).strftime("%Y-%m-%d")
        
        # 1. Fetch Raw Data (Contains 'close')
        df_raw = build_dataset(
            tickers=tickers,
            start=start_date,
            end=end_date,
            window=self.vol_window,
            cache_dir=None,
            show_progress=True,
        )
        
        # 2. Engineer Features (Likely drops 'close')
        # Enable earnings features for VolNetX (trained with event_earnings_heat)
        df_recent = build_features(df_raw, include_macro=True, include_earnings=True)
        
        # 3. FIX: Merge 'close' back in if it was dropped
        if "close" in df_raw.columns and "close" not in df_recent.columns:
            # Merge on keys to ensure alignment
            df_recent = df_recent.merge(
                df_raw[["date", "ticker", "close"]], 
                on=["date", "ticker"], 
                how="left"
            )
            
        self.df_recent = df_recent
        return df_recent

    def run(self, tickers):
        """
        Generate multi-horizon forecasts for one or more tickers.

        Prepares recent data, runs batched predictions, attaches realized volatility,
        and computes cross-sectional analytics for convenience.

        :param tickers: Ticker symbol or list of symbols to forecast.
        :type tickers: str or list[str]
        :return: Forecast snapshot with columns like ['ticker','pred_vol_1','pred_vol_5',...,'realized_vol'].
        :rtype: pandas.DataFrame
        """
        tickers = [tickers] if isinstance(tickers, str) else tickers
        print(f"\n🌍 Running forecasts for {len(tickers)} tickers...\n")

        df_recent = self._prepare_data(tickers)
        preds = predict_batch(
            self.model,
            self.meta,
            df_recent,
            tickers,
            scalers=self.scalers,
            ticker_to_id=self.ticker_to_id,
            features=self.features,
        )
        preds = attach_realized(preds, df_recent)
        
        # --- NEW: Multi-Horizon Momentum ---
        if "close" in df_recent.columns:
            # Group once to avoid re-grouping 3 times
            grp = df_recent.sort_values("date").groupby("ticker")["close"]
            
            # Calculate 5d, 10d, 20d momentum (Snapshot of the LATEST value)
            mom_5 = grp.apply(lambda x: x.pct_change(5).iloc[-1]).rename("momentum_5d")
            mom_10 = grp.apply(lambda x: x.pct_change(10).iloc[-1]).rename("momentum_10d")
            mom_20 = grp.apply(lambda x: x.pct_change(20).iloc[-1]).rename("momentum_20d")
            
            # Merge all into preds
            preds = preds.merge(mom_5, on="ticker", how="left")
            preds = preds.merge(mom_10, on="ticker", how="left")
            preds = preds.merge(mom_20, on="ticker", how="left")
        else:
            preds["momentum_5d"] = 0.0
            preds["momentum_10d"] = 0.0
            preds["momentum_20d"] = 0.0

        self.predictions = preds
        print("✅ Forecast complete.")

        # Attach analytics object
        self.signals = Analytics(preds)
        self.signals.compute()

        return preds

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        ticker: str,
        show_vix: bool = False,
        vix_df: pd.DataFrame = None,
        show: bool = True,
    ):
        """
        Plot realized volatility history and overlay constant forecast levels by horizon.

        Draws:
          - Realized volatility time series for recent months.
          - Horizontal dashed lines for each available predicted horizon.
          - Optional VIX overlay if provided.

        :param ticker: Ticker symbol to visualize.
        :type ticker: str
        :param show_vix: Whether to overlay VIX time series (requires vix_df).
        :type show_vix: bool
        :param vix_df: DataFrame with columns ['date','close'] representing VIX levels.
        :type vix_df: pandas.DataFrame, optional
        :param show: If True, render the plot and return None; if False, return the Figure.
        :type show: bool
        :raises RuntimeError: If .run() has not been called prior to plotting.
        :raises ValueError: If no forecast results are available for the requested ticker.
        :return: Matplotlib Figure when show=False; otherwise None.
        :rtype: matplotlib.figure.Figure or None
        """
        if self.predictions is None:
            raise RuntimeError("No forecasts computed yet. Run .run(ticker) first.")

        preds = self.predictions[self.predictions["ticker"] == ticker]
        if preds.empty:
            raise ValueError(f"No forecast results for {ticker}")

        df_t = self.df_recent[self.df_recent["ticker"] == ticker].copy()
        df_t = df_t.sort_values("date").tail(180)  # last ~6 months

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            df_t["date"],
            df_t["realized_vol"],
            label="Realized Volatility",
            color="tab:blue",
        )

        # Collect horizons we actually have predictions for
        present_horizons = [
            h
            for h in sorted(set(getattr(self, "horizons", [])))
            if f"pred_vol_{h}" in preds.columns
        ]

        # Unique colors per horizon (skip index 0 in tab10 to avoid blue clash)
        palette = plt.get_cmap("tab10").colors
        start_idx = 1  # 0 is blue; we already used that for realized vol
        color_map = {
            h: palette[(start_idx + i) % len(palette)]
            for i, h in enumerate(present_horizons)
        }

        # Plot all horizons with distinct colors
        for h in present_horizons:
            col = f"pred_vol_{h}"
            y = preds[col].values[0]
            ax.axhline(
                y, color=color_map[h], linestyle="--", alpha=0.9, label=f"Pred {h}d"
            )

        if show_vix and vix_df is not None:
            vix_df = vix_df.copy()
            vix_df["date"] = pd.to_datetime(vix_df["date"], errors="coerce")
            ax.plot(
                vix_df["date"],
                vix_df["close"],
                color="tab:red",
                alpha=0.5,
                label="VIX Index",
            )

        ax.set_title(f"{ticker} — Forecast vs Realized Volatility")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility")

        # Deduplicate legend entries (defensive)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if show:
            plt.show()
            plt.close(fig)  # avoid a second capture by caller
            return None

        return fig
