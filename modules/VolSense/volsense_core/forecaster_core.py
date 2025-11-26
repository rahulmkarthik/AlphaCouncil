"""
volsense_core.forecaster_core
=============================

.. module:: volsense_core.forecaster_core
   :synopsis: Unified training and inference runtime for VolSense.

Overview
--------
This module provides a high-level wrapper and helpers for training and
evaluating VolSense forecasting backends including:

- BaseLSTM (ticker-specific recurrent forecaster)
- GlobalVolForecaster (multi-ticker shared model)
- ARCH-family forecasters (GARCH, EGARCH, GJR)

Public API
----------
Classes
  VolSenseForecaster
    Unified interface to train, evaluate and persist different forecasting
    backends. Produces standardized forecast DataFrames on the realized
    volatility scale.

Functions
  make_forecast_df(preds, actuals, dates, tickers, horizons, model_name)
    Build a tidy evaluation DataFrame from model predictions and realized
    values. Used internally by VolSenseForecaster for consistent output
    formatting.

Usage
-----
>>> from volsense_core.forecaster_core import VolSenseForecaster
>>> vf = VolSenseForecaster(method="global_lstm", device="cpu")
>>> vf.fit(training_df)
>>> df_results = vf.predict(evaluation_df)

Notes
-----
- All neural models expect inputs and features produced by the pipeline in
  volsense_core.data.feature_engineering.
- Forecast outputs use the column schema:
  ['asof_date','date','ticker','horizon','forecast_vol','realized_vol','model'].
"""

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from tqdm import tqdm

# --- Model Imports ---
from volsense_core.models.garch_methods import ARCHForecaster
from volsense_core.models.lstm_forecaster import (
    train_baselstm,
    evaluate_baselstm,
)
from volsense_core.models.global_vol_forecaster import (
    train_global_model,
    predict_next_day,
    make_last_windows,
)
from volsense_core.models.lstm_forecaster import TrainConfig as LSTMTrainConfig
from volsense_core.models.global_vol_forecaster import TrainConfig as GlobalTrainConfig


__all__ = ["VolSenseForecaster"]


# ============================================================
# 🔧 Utility: Unified forecast DataFrame builder
# ============================================================
def make_forecast_df(preds, actuals, dates, tickers, horizons, model_name):
    """
    Build a standardized forecast-evaluation DataFrame across horizons and tickers.

    :param preds: Predicted volatilities with shape (n_samples, n_horizons) on realized scale.
    :type preds: numpy.ndarray or pandas.DataFrame
    :param actuals: Realized future volatilities with shape (n_samples, n_horizons); use None if unavailable.
    :type actuals: numpy.ndarray or pandas.DataFrame or None
    :param dates: As-of dates aligned to rows in preds/actuals (length n_samples).
    :type dates: array-like of datetime-like
    :param tickers: Ticker identifiers aligned to rows in preds/actuals (length n_samples).
    :type tickers: array-like of str
    :param horizons: Forecast horizons corresponding to columns in preds/actuals (length n_horizons).
    :type horizons: list[int] or tuple[int, ...]
    :param model_name: Name of the model generating forecasts.
    :type model_name: str
    :return: Tidy DataFrame with columns ['asof_date','date','ticker','horizon','forecast_vol','realized_vol','model'].
    :rtype: pandas.DataFrame
    """
    records = []
    for h_idx, h in enumerate(horizons):
        df = pd.DataFrame(
            {
                "asof_date": pd.to_datetime(dates),
                "date": pd.to_datetime(dates) + BDay(h),
                "ticker": tickers,
                "horizon": h,
                "forecast_vol": preds[:, h_idx],
                "realized_vol": actuals[:, h_idx] if actuals is not None else np.nan,
                "model": model_name,
            }
        )
        records.append(df)
    return pd.concat(records, ignore_index=True).sort_values(
        ["ticker", "asof_date", "horizon"]
    )


# ============================================================
# 🌐 Unified Forecaster Wrapper
# ============================================================
class VolSenseForecaster:
    """
    Unified forecasting API for BaseLSTM, GlobalVolForecaster, and GARCH family models.

    All results are on realized (non-log) volatility scale.
    Output schema:
      ['asof_date','date','ticker','horizon','forecast_vol','realized_vol','model']

    :param method: One of 'lstm', 'global_lstm', 'garch', 'egarch', 'gjr'.
    :param device: 'cpu' or 'cuda'.
    :param mode: Default prediction mode ('eval' or 'inference').
    :param kwargs: method-specific configuration (window, horizons, epochs, extra_features, etc.)
    """

    def __init__(self, method="lstm", device="cpu", mode="eval", **kwargs):
        """
        Initialize a VolSense forecaster wrapper for the chosen method.

        :param method: Forecasting backend to use: 'lstm', 'global_lstm', 'garch', 'egarch', or 'gjr'.
        :type method: str
        :param device: Compute device for neural models, e.g., 'cpu' or 'cuda'.
        :type device: str
        :param mode: Default prediction mode, typically 'eval' (historical backtest).
        :type mode: str
        :param kwargs: Additional method-specific configuration:
            - LSTM/global: window, horizons, epochs, lr, dropout, hidden_dim, num_layers, val_start, extra_features
            - Global only: global_ckpt_path
            - GARCH family: p, q, o (GJR only), dist
            - Common: ticker (to pin a single ticker if needed)
        :type kwargs: dict
        """
        self.method = method.lower()
        self.device = device
        self.mode = mode
        self.kwargs = kwargs
        self.model = None
        self.cfg = None
        self.global_window = None
        self.global_ticker_to_id = None
        self.global_scalers = None
        self.ticker = kwargs.get("ticker", None)

        if self.method in ["garch", "egarch", "gjr"]:
            self.model = ARCHForecaster(
                model=self.method,
                p=kwargs.get("p", 1),
                q=kwargs.get("q", 1),
                o=kwargs.get("o", 1 if self.method == "gjr" else 0),
                dist=kwargs.get("dist", "t"),
            )

    # ============================================================
    # 🧠 Training
    # ============================================================
    def fit(self, data, **train_kwargs):
        """
        Fit the selected forecasting model.

        LSTM:
          - trains a per-ticker BaseLSTM.

        Global LSTM:
          - trains a single GlobalVolForecaster across all tickers.

        GARCH family:
          - fits a ticker-specific ARCH/GARCH variant.

        :param data: pandas.DataFrame containing required columns (date,ticker,return,realized_vol).
        :returns: self (trained forecaster)
        :rtype: VolSenseForecaster
        :raises KeyError: missing extra_features columns.
        :raises ValueError: unknown method.
        """
        extra_feats = self.kwargs.get("extra_features", None)

        # -------------------------------
        # LSTM (Ticker-specific)
        # -------------------------------
        if self.method == "lstm":
            print("🧩 Training BaseLSTM Forecaster...")
            self.ticker = data["ticker"].iloc[0]
            cfg = LSTMTrainConfig(
                window=self.kwargs.get("window", 30),
                horizons=self.kwargs.get("horizons", [1, 5, 10]),
                val_start=self.kwargs.get("val_start", "2023-01-01"),
                device=self.device,
                epochs=self.kwargs.get("epochs", 20),
                lr=self.kwargs.get("lr", 5e-4),
                dropout=self.kwargs.get("dropout", 0.2),
                hidden_dim=self.kwargs.get("hidden_dim", 128),
                num_layers=self.kwargs.get("num_layers", 3),
                output_activation="none",
                extra_features=extra_feats,
            )
            self.cfg = cfg

            # Validate extra features for LSTM if provided
            if extra_feats is not None:
                missing = [c for c in extra_feats if c not in data.columns]
                if missing:
                    raise KeyError(f"Missing columns for extra_features: {missing}")

            # Train regardless of whether extra_feats was provided
            self.model, self.hist, loaders = train_baselstm(data, cfg)
            self._val_loader = loaders[1]
            return self

        # -------------------------------
        # GARCH family (ticker-specific)
        # -------------------------------
        elif self.method in ["garch", "egarch", "gjr"]:
            print(f"📈 Fitting {self.method.upper()} Forecaster...")
            # Ensure single-ticker data
            if "ticker" in data.columns and data["ticker"].nunique() > 1:
                if not self.ticker:
                    self.ticker = data["ticker"].iloc[0]
                data = data[data["ticker"] == self.ticker].copy()

            if not self.ticker:
                self.ticker = data["ticker"].iloc[0]

            self.data = data.copy()
            ret_series = self.data.dropna(subset=["return"]).set_index("date")["return"]
            self.model.fit(ret_series)
            print(
                f"✅ {self.method.upper()} fit complete for {self.ticker} ({len(ret_series)} obs)."
            )
            return self

        # -------------------------------
        # Global LSTM (shared model)
        # -------------------------------
        elif self.method == "global_lstm":
            print("🌐 Training GlobalVolForecaster...")
            cfg = GlobalTrainConfig(
                window=self.kwargs.get("window", 40),
                horizons=self.kwargs.get("horizons", [1, 5, 10]),
                val_start=self.kwargs.get("val_start", "2023-01-01"),
                device=self.device,
                epochs=self.kwargs.get("epochs", 10),
                extra_features=extra_feats,
            )
            self.cfg = cfg

            # Validate extra features for Global LSTM if provided
            if extra_feats is not None:
                missing = [c for c in extra_feats if c not in data.columns]
                if missing:
                    raise KeyError(f"Missing columns for extra_features: {missing}")

            model, hist, val_loader, t2i, scalers, feats = train_global_model(data, cfg)
            self.model = model
            self.hist = hist
            self._val_loader = val_loader
            self.global_ticker_to_id = t2i
            self.global_scalers = scalers
            self.global_window = cfg.window
            if self.kwargs.get("global_ckpt_path"):
                save_checkpoint(self.kwargs["global_ckpt_path"], model, t2i, scalers)
            return self

        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ============================================================
    # 🔮 Prediction
    # ============================================================
    def predict(self, data=None, mode=None):
        """
        Generate forecasts (and realized-aligned evaluations when available).

        LSTM: returns eval-set predictions for the trained ticker on realized scale.
        Global LSTM: performs rolling realized-aligned evaluation across tickers; requires `data`.
        GARCH family: returns rolling 1-day forecasts with realized vol alignment.

        :param data: Input DataFrame required for 'global_lstm' evaluation; ignored for 'lstm' and GARCH.
        :type data: pandas.DataFrame or None
        :param mode: Optional override for prediction mode (unused in current implementations).
        :type mode: str or None
        :raises ValueError: If 'global_lstm' is used without providing `data`, or if method is unknown.
        :raises RuntimeError: If GARCH model has not been fitted prior to prediction.
        :return: Standardized forecast-evaluation DataFrame with columns ['asof_date','date','ticker','horizon','forecast_vol','realized_vol','model'].
        :rtype: pandas.DataFrame
        """
        mode = mode or self.mode
        horizons = getattr(self.cfg, "horizons", [1])
        model_name = (
            "GlobalVolForecaster"
            if self.method == "global_lstm"
            else ("BaseLSTM" if self.method == "lstm" else self.method.upper())
        )

        # -------------------------------
        # LSTM (Ticker-specific)
        # -------------------------------
        if self.method == "lstm":
            preds, actuals = evaluate_baselstm(
                self.model, self._val_loader, self.cfg, device=self.device
            )
            preds = np.asarray(preds)
            actuals = np.asarray(actuals)
            dates = getattr(
                self._val_loader.dataset, "sample_dates", [None] * len(preds)
            )
            tickers = np.repeat(self.ticker, len(preds))

            # convert from log-vol to realized scale
            preds_realized = np.exp(preds)
            actuals_realized = np.exp(actuals)

            return make_forecast_df(
                preds_realized, actuals_realized, dates, tickers, horizons, model_name
            )

        # -------------------------------
        # Global LSTM (Multi-Ticker, Realized-Aligned Evaluation)
        # -------------------------------
        elif self.method == "global_lstm":
            if data is None:
                raise ValueError(
                    "GlobalLSTM requires input DataFrame with multiple tickers."
                )

            data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
            horizons = self.cfg.horizons

            # Compute future realized vols for alignment
            for h in horizons:
                data[f"realized_shift_{h}d"] = data.groupby("ticker")[
                    "realized_vol"
                ].shift(-h)

            preds_all, actuals_all, dates_all, tickers_all = [], [], [], []

            # Rolling evaluation (vectorized within ticker)
            for ticker, df_t in tqdm(
                data.groupby("ticker"), desc="Rolling eval forecasts"
            ):
                df_t = df_t.dropna(subset=["realized_vol"]).reset_index(drop=True)

                # generate all valid windows
                windows = [
                    df_t.iloc[i - self.global_window : i]
                    for i in range(self.global_window, len(df_t))
                ]

                # skip short tickers
                if not windows:
                    continue

                for i, w in enumerate(windows):
                    df_win = make_last_windows(w, window=self.global_window)
                    preds_df = predict_next_day(
                        self.model,
                        df_win,
                        self.global_ticker_to_id,
                        self.global_scalers,
                        window=self.global_window,
                        device=self.device,
                        show_progress=False,
                    )
                    preds = np.stack(preds_df["forecast_vol_scaled"].values)
                    preds_realized = np.exp(preds)

                    # realized vol from actual data
                    realized_vals = [
                        df_t[f"realized_shift_{h}d"].iloc[self.global_window + i - 1]
                        for h in horizons
                    ]

                    if all(np.isnan(realized_vals)):
                        continue  # skip if no realized vols yet

                    preds_all.append(preds_realized)
                    actuals_all.append(realized_vals)
                    dates_all.append(df_t["date"].iloc[self.global_window + i - 1])
                    tickers_all.append(ticker)

            # Convert to arrays
            preds_all = np.array(preds_all)
            actuals_all = np.array(actuals_all)

            # Build standardized DataFrame
            df_out = make_forecast_df(
                preds=preds_all,
                actuals=actuals_all,
                dates=dates_all,
                tickers=tickers_all,
                horizons=horizons,
                model_name="GlobalVolForecaster",
            )

            df_out = df_out.dropna(subset=["realized_vol"]).reset_index(drop=True)
            print(
                f"✅ GlobalVolForecaster realized-aligned evaluation complete ({len(df_out)} rows)."
            )
            return df_out
        # -------------------------------
        # GARCH Family (Ticker-specific)
        # -------------------------------
        elif self.method in ("garch", "egarch", "gjr"):
            if self.model is None:
                raise RuntimeError("GARCH model not fitted yet. Call .fit() first.")
            ticker = self.ticker
            ret_series = self.data.dropna(subset=["return"]).set_index("date")["return"]

            realized_vol = ret_series.rolling(21).std() * np.sqrt(252)
            rolling_preds = self.model.rolling_forecast(ret_series, refit_every=5)
            df_eval = (
                pd.concat(
                    [
                        rolling_preds.rename("forecast_vol"),
                        realized_vol.rename("realized_vol"),
                    ],
                    axis=1,
                )
                .dropna()
                .reset_index()
            )
            df_eval["asof_date"] = pd.to_datetime(df_eval["date"])
            df_eval["date"] = df_eval["asof_date"] + BDay(1)
            df_eval["ticker"] = ticker
            df_eval["horizon"] = 1
            df_eval["model"] = self.method.upper()

            return df_eval[
                [
                    "asof_date",
                    "date",
                    "ticker",
                    "horizon",
                    "forecast_vol",
                    "realized_vol",
                    "model",
                ]
            ]

        else:
            raise ValueError(f"Unknown method: {self.method}")

    # ============================================================
    # 💾 Unified Checkpoint Saving (BaseLSTM + GlobalVolForecaster)
    # ============================================================
    def save(
        self, save_dir: str = "models", version: str = "latest", device: str = "cpu"
    ):
        """
        Save trained model and training artifacts in standardized VolSense formats.

        Produces:
        - <stem>_full.pkl
        - <stem>_bundle.pkl
        - <stem>.meta.json + <stem>.pt

        :param save_dir: Directory to store model artifacts.
        :param version: Version tag (e.g., 'v507').
        :param device: 'cpu' or 'cuda' (model will be moved to this device for serialization).
        :returns: meta dict produced by save_checkpoint utility.
        :rtype: dict
        """
        import os
        from volsense_core.utils.checkpoint_utils import save_checkpoint

        os.makedirs(save_dir, exist_ok=True)

        # --- Validate device ---
        device = device.lower()
        if device not in ("cpu", "cuda"):
            raise ValueError("Device must be 'cpu' or 'cuda'")

        # --- Move model to the requested device ---
        self.model.to(device)
        print(f"💾 Preparing to save model on device: {device.upper()}")

        # --- Identify model type ---
        model_class = self.model.__class__.__name__.lower()
        if "lstm" in model_class:
            arch_type = "baselstm" if "base" in model_class else "globalvolforecaster"
        else:
            arch_type = model_class

        # --- Compose version tag ---
        version_tag = f"{arch_type}_{version}"

        # --- Extract features and ticker mappings ---
        if arch_type == "globalvolforecaster":
            features = getattr(self, "global_features", None)
            ticker_to_id = getattr(self, "global_ticker_to_id", None)
        elif arch_type == "baselstm":
            features = getattr(self.cfg, "features", None)
            ticker_to_id = {getattr(self, "ticker", "TICKER"): 0}
        else:
            features = getattr(self.cfg, "features", [])
            ticker_to_id = {}

        # --- Call the centralized saver ---
        meta = save_checkpoint(
            model=self.model,
            cfg=self.cfg,
            version=version_tag,
            save_dir=save_dir,
            ticker_to_id=ticker_to_id,
            features=features,
        )

        print(f"\n✅ Model saved successfully in {save_dir}: {version_tag}")
        print("   Formats generated: .full.pkl, _bundle.pkl, .meta.json + .pth")
        print(f"   Serialized on device: {device.upper()}")
        print(
            "   🔁 Ready for reloading via: load_model(..., checkpoints_dir='models')"
        )
        return meta
