"""
VolSense Analytics — Trader Snapshot Module
-------------------------------------------
Lightweight analytical layer for daily volatility forecasts.
Generates actionable metrics, relative signals, and visualizations
for multi-ticker forecasts without requiring historical data.

This module is intended for use within volsense_inference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


class Analytics:
    """
    Trader-facing analytics for VolSense forecasts.

    :param preds: DataFrame with columns like ['ticker','realized_vol','pred_vol_1', ...].
    :type preds: pandas.DataFrame
    :param vol_regime_quantiles: Quantile thresholds (low, high) for regime classification.
    :type vol_regime_quantiles: list[float], optional
    :raises ValueError: If an empty or None forecast DataFrame is provided.
    """

    def __init__(
        self, preds: pd.DataFrame, vol_regime_quantiles: List[float] = [0.2, 0.8]
    ):
        """
        Initialize the Analytics module with a single-snapshot forecast table.

        :param preds: Forecast snapshot containing realized and predicted vol columns.
        :type preds: pandas.DataFrame
        :param vol_regime_quantiles: Quantile thresholds (low, high) for regime classification.
        :type vol_regime_quantiles: list[float], optional
        :raises ValueError: If preds is None or empty.
        """
        if preds is None or preds.empty:
            raise ValueError("Empty forecast DataFrame passed to Analytics.")

        self.raw = preds.copy().reset_index(drop=True)
        self.vol_regime_quantiles = vol_regime_quantiles
        self.processed: Optional[pd.DataFrame] = None

    # ============================================================
    # 🔹 Core computation
    # ============================================================
    def compute(self) -> pd.DataFrame:
        """
        Compute cross-sectional volatility analytics for a single snapshot.

        Derives per-horizon metrics:
          - f/r ratio: forecast_vol / realized_vol
          - z-score: cross-sectional z-score vs realized_vol mean/std
          - signal_strength: clipped z-score in [-3, 3]
          - vol_regime: 'Low Vol' | 'Normal' | 'High Vol' via quantiles

        :return: Processed DataFrame with added per-horizon columns and 'vol_regime'.
        :rtype: pandas.DataFrame
        """
        df = self.raw.copy()
        horizons = [c for c in df.columns if c.startswith("pred_vol_")]

        # Cross-sectional mean/std (snapshot)
        mean = df["realized_vol"].mean()
        std = df["realized_vol"].std() or 1e-8

        for h in horizons:
            df[f"{h}_f_r_ratio"] = df[h] / (df["realized_vol"] + 1e-8)
            df[f"{h}_zscore"] = (df[h] - mean) / std
            df[f"{h}_signal_strength"] = df[f"{h}_zscore"].clip(-3, 3)

        # Volatility regimes
        low_q, high_q = self.vol_regime_quantiles
        low_thr = df["realized_vol"].quantile(low_q)
        high_thr = df["realized_vol"].quantile(high_q)

        def _regime(v):
            if v < low_thr:
                return "Low Vol"
            elif v > high_thr:
                return "High Vol"
            else:
                return "Normal"

        df["vol_regime"] = df["realized_vol"].apply(_regime)
        self.processed = df
        return df

    # ============================================================
    # 🔹 Quick trader summary
    # ============================================================
    def summary(self, horizon: str = "pred_vol_5") -> pd.DataFrame:
        """
        Summarize realized vs forecast metrics for each ticker.

        Returns a compact table with:
          ['ticker','realized_vol','forecast_vol','f/r_ratio','zscore','vol_regime'].

        If the requested horizon is absent, falls back to the first available 'pred_vol_*' column.

        :param horizon: Horizon column to summarize (e.g., 'pred_vol_5').
        :type horizon: str
        :return: Ticker-level summary sorted by descending z-score.
        :rtype: pandas.DataFrame
        """
        if self.processed is None:
            self.compute()
        df = self.processed
        if horizon not in df.columns:
            horizon = [c for c in df.columns if c.startswith("pred_vol_")][0]

        cols = [
            "ticker",
            "realized_vol",
            horizon,
            f"{horizon}_f_r_ratio",
            f"{horizon}_zscore",
            "vol_regime",
        ]
        df_summary = df[cols].copy()
        df_summary.rename(
            columns={
                horizon: "forecast_vol",
                f"{horizon}_f_r_ratio": "f/r_ratio",
                f"{horizon}_zscore": "zscore",
            },
            inplace=True,
        )
        df_summary.sort_values("zscore", ascending=False, inplace=True)
        return df_summary.reset_index(drop=True)

    # ============================================================
    # 🔹 Human-readable interpretation
    # ============================================================
    def describe(self, ticker: str, horizon: str = "pred_vol_5") -> str:
        """
        Produce a human-readable interpretation for a single ticker.

        The message encodes direction via f/r ratio (>1 rising, <1 easing), the
        cross-sectional z-score, and the volatility regime.

        :param ticker: Ticker to describe.
        :type ticker: str
        :param horizon: Horizon column to reference (e.g., 'pred_vol_5').
        :type horizon: str
        :return: Short interpretation string for trader consumption.
        :rtype: str
        """
        if self.processed is None:
            self.compute()
        df = self.processed
        if ticker not in df["ticker"].values:
            return f"{ticker}: no data available."

        row = df[df["ticker"] == ticker].iloc[-1]
        regime = row["vol_regime"]
        if horizon not in row:
            horizon = [c for c in df.columns if c.startswith("pred_vol_")][0]

        ratio = row.get(f"{horizon}_f_r_ratio", np.nan)
        z = row.get(f"{horizon}_zscore", np.nan)

        if np.isnan(ratio) or np.isnan(z):
            return f"{ticker}: insufficient data for signal generation."

        direction = "↑ rising" if ratio > 1 else "↓ easing"
        horizon_label = horizon.replace("pred_vol_", "")
        return (
            f"{ticker}: {horizon_label}-day vol {direction} "
            f"({z:+.2f}σ vs peers), regime: {regime}."
        )

    # ============================================================
    # 🔹 Visualization
    # ============================================================
    # ...existing code...
    def plot(self, horizon: str = "pred_vol_5", show: bool = False):
        """
        Scatter plot of forecast vs realized volatility at a selected horizon.

        Each ticker is plotted as a labeled point; includes a 45-degree reference line.

        :param horizon: Horizon column to plot (e.g., 'pred_vol_5').
        :type horizon: str
        :param show: If True, render and close the figure; if False, return the Figure.
        :type show: bool
        :return: Matplotlib Figure when show=False; otherwise None.
        :rtype: matplotlib.figure.Figure or None
        """
        if self.processed is None:
            self.compute()
        df = self.processed.copy()

        if horizon not in df.columns:
            horizon = [c for c in df.columns if c.startswith("pred_vol_")][0]

        # --- Generate unique color map per ticker
        tickers = sorted(df["ticker"].unique())
        cmap = plt.cm.get_cmap("tab10", len(tickers))
        colors = {t: cmap(i) for i, t in enumerate(tickers)}

        # --- Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for t in tickers:
            row = df[df["ticker"] == t].iloc[-1]
            ax.scatter(
                row["realized_vol"],
                row[horizon],
                s=100,
                color=colors[t],
                edgecolor="k",
                alpha=0.85,
                label=t,
            )

        # --- 45-degree reference line
        min_vol = df[["realized_vol", horizon]].min().min()
        max_vol = df[["realized_vol", horizon]].max().max()
        ax.plot([min_vol, max_vol], [min_vol, max_vol], "k--", lw=1)

        ax.set_xlabel("Realized Volatility", fontsize=11)
        ax.set_ylabel("Forecasted Volatility", fontsize=11)
        ax.set_title(
            f"Forecast Snapshot — {horizon.replace('pred_vol_', '')}-Day Horizon"
        )
        ax.legend(
            title="Ticker", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()

        # Notebook/Streamlit-safe: no implicit show unless requested
        if show:
            plt.show()
            plt.close(fig)
            return None

        fig_out = fig
        plt.close(fig)
        return fig_out
