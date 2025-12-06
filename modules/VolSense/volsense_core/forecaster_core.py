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
- VolNetX (advanced LSTM + Attention + Transformer neural network forecaster)
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
import os
import torch
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
from volsense_core.models.volnetx import (VolNetXConfig, build_volnetx_dataset, train_volnetx)
from volsense_core.models.lstm_forecaster import TrainConfig as LSTMTrainConfig
from volsense_core.models.global_vol_forecaster import TrainConfig as GlobalTrainConfig
from volsense_core.utils.checkpoint_utils import save_checkpoint


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
    Unified forecasting API for BaseLSTM, GlobalVolForecaster, VolNetX, and GARCH family models.

    All results are on realized (non-log) volatility scale.
    Output schema:
      ['asof_date','date','ticker','horizon','forecast_vol','realized_vol','model']

    :param method: One of 'lstm', 'global_lstm', 'garch', 'egarch', 'gjr', 'volnetx'.
    :param device: 'cpu' or 'cuda'.
    :param mode: Default prediction mode ('eval' or 'inference').
    :param kwargs: method-specific configuration (window, horizons, epochs, extra_features, etc.)
        - For all methods:
            - method: "lstm", "global_lstm", or "volnetx"
            - extra_features: list of feature column names beyond "return"
            - device: "cpu" or "cuda"
            - global_ckpt_path: path for automatic checkpoint saving
            
        - LSTM-specific:
            - window, horizons, epochs, lr, dropout, hidden_dim, num_layers, val_start
            
        - GlobalVolForecaster (global_lstm):
            - Data: window, horizons, stride, val_start, val_end, val_mode, embargo_days, target_col
            - Training: epochs, lr, batch_size, early_stop, patience
            - Architecture: dropout, use_layernorm, separate_heads, attention, residual_head
            - Regularization: feat_dropout_p, variational_dropout_p
            - Strategy: cosine_schedule, cosine_restarts, grad_clip, weight_decay, loss_horizon_weights, loss_type
            - Data Loading: oversample_high_vol, dynamic_window_jitter, num_workers, pin_memory
            - EMA: use_ema, ema_decay
            
        - VolNetX:
            - window, horizons, epochs, lr, batch_size, hidden_dim, num_layers, dropout
            - feat_dropout, use_transformer, use_feature_attention, loss_horizon_weights
            - val_start, patience, grad_clip, weight_decay, cosine_schedule
    :type kwargs: dict
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
        :param kwargs: See class docstring for full parameter list.
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

        :param data: pandas.DataFrame containing required columns (date, ticker, return, realized_vol).
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
                # Data & Windowing
                window=self.kwargs.get("window", 75),
                horizons=self.kwargs.get("horizons", [1, 5, 10]),
                stride=self.kwargs.get("stride", 3),
                val_start=self.kwargs.get("val_start", "2023-01-01"),
                val_end=self.kwargs.get("val_end", None),
                val_mode=self.kwargs.get("val_mode", "causal"),
                embargo_days=self.kwargs.get("embargo_days", 0),
                target_col=self.kwargs.get("target_col", "realized_vol_log"),
                extra_features=extra_feats,
                
                # Training Dynamics
                epochs=self.kwargs.get("epochs", 45),
                lr=self.kwargs.get("lr", 3e-4),
                batch_size=self.kwargs.get("batch_size", 256),
                device=self.device,
                
                # Architecture & Regularization
                dropout=self.kwargs.get("dropout", 0.15),
                use_layernorm=self.kwargs.get("use_layernorm", True),
                separate_heads=self.kwargs.get("separate_heads", True),
                attention=self.kwargs.get("attention", True),
                residual_head=self.kwargs.get("residual_head", True),
                feat_dropout_p=self.kwargs.get("feat_dropout_p", 0.1),
                variational_dropout_p=self.kwargs.get("variational_dropout_p", 0.1),
                
                # Training Strategy
                early_stop=self.kwargs.get("early_stop", True),
                patience=self.kwargs.get("patience", 10),
                cosine_schedule=self.kwargs.get("cosine_schedule", True),
                cosine_restarts=self.kwargs.get("cosine_restarts", True),
                grad_clip=self.kwargs.get("grad_clip", 1.0),
                loss_horizon_weights=self.kwargs.get("loss_horizon_weights", None),
                
                # Data Loading & Augmentation
                oversample_high_vol=self.kwargs.get("oversample_high_vol", False),
                dynamic_window_jitter=self.kwargs.get("dynamic_window_jitter", 0),
                num_workers=self.kwargs.get("num_workers", 2),
                pin_memory=self.kwargs.get("pin_memory", True),
                
                # EMA
                use_ema=self.kwargs.get("use_ema", True),
                ema_decay=self.kwargs.get("ema_decay", 0.995),
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
                # Split path into dir and version
                ckpt_path = self.kwargs["global_ckpt_path"]
                save_dir = os.path.dirname(ckpt_path)
                version = os.path.basename(ckpt_path)
                save_checkpoint(model, cfg, version, save_dir, t2i, feats, scalers)
            return self

        # -------------------------------
        # 🧠 VolNetX (Transformer-Hybrid)
        # -------------------------------
        elif self.method == "volnetx":
            print("🧠 Training VolNetX Hybrid Model...")
            
            # 1. Config
            cfg = VolNetXConfig(
                window=self.kwargs.get("window", 65),
                horizons=self.kwargs.get("horizons", [1, 5, 10]),
                input_size=self.kwargs.get("input_size", 16),
                hidden_dim=self.kwargs.get("hidden_dim", 160),
                num_layers=self.kwargs.get("num_layers", 2),
                dropout=self.kwargs.get("dropout", 0.2),
                feat_dropout=self.kwargs.get("feat_dropout", 0.0),
                lr=self.kwargs.get("lr", 5e-4),
                epochs=self.kwargs.get("epochs", 20),
                batch_size=self.kwargs.get("batch_size", 64),
                device=self.device,
                use_transformer=self.kwargs.get("use_transformer", True),
                use_feature_attention=self.kwargs.get("use_feature_attention", True),
                val_mode=self.kwargs.get("val_mode", "causal"),
                val_start=self.kwargs.get("val_start", None),
                val_end=self.kwargs.get("val_end", None),
                loss_horizon_weights=self.kwargs.get("loss_horizon_weights", (0.55, 0.25, 0.20)),
                loss_type=self.kwargs.get("loss_type", "mse"),
                cosine_schedule=self.kwargs.get("cosine_schedule", False),
                grad_clip=self.kwargs.get("grad_clip", 0.5),
                weight_decay=self.kwargs.get("weight_decay", 1e-4),
                patience=self.kwargs.get("patience", 5),
                early_stop=self.kwargs.get("early_stop", True),
            )
            cfg.extra_features = extra_feats 
            self.cfg = cfg

            # 2. Prepare Data
            print(f"   ↳ Building VolNetX dataset ({cfg.val_mode} mode)...")
            features = ["return"] + (extra_feats or [])
            cfg.input_size = len(features)
            
            # 🚀 UPDATED: unpack the scaler returned by build_volnetx_dataset
            t2i, train_loader, val_loader, _, _, scaler = build_volnetx_dataset(
                data, 
                features=features,
                target_col="realized_vol_log",
                config=cfg
            )
            
            self.global_ticker_to_id = t2i
            self.global_window = cfg.window
            
            # 🚀 Store the scaler dictionary directly
            # build_volnetx_dataset now returns a dict {ticker: scaler}
            if isinstance(scaler, dict):
                self.global_scalers = scaler
            else:
                # Fallback for legacy behavior (should not happen with new volnetx.py)
                self.global_scalers = {t: scaler for t in t2i.keys()}

            # 3. Train
            print("   ↳ Starting training loop...")
            model = train_volnetx(cfg, train_loader, val_loader, n_tickers=len(t2i))
            self.model = model

            # 4. Save
            if self.kwargs.get("global_ckpt_path"):
                 self.save(
                    save_dir=os.path.dirname(self.kwargs["global_ckpt_path"]),
                    version=os.path.basename(self.kwargs["global_ckpt_path"]),
                    device=self.device
                )
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
        VolNetX: returns rolling forecasts with realized vol alignment.
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
                
                # Skip tickers with insufficient data
                if len(df_t) <= self.global_window:
                    continue
                
                ticker_id = self.global_ticker_to_id.get(ticker)
                if ticker_id is None:
                    continue
                
                # Get scaler for this ticker
                scaler = self.global_scalers.get(ticker)
                if scaler is None:
                    continue
                
                # 🚀 VECTORIZED: Build feature matrix
                features = ["return"] + (getattr(self.cfg, "extra_features", []) or [])
                feat_data = df_t[features].values.astype(np.float32)
                
                # Apply per-ticker scaling
                feat_scaled = scaler.transform(feat_data)
                
                # 🚀 VECTORIZED: Sliding window construction using stride tricks
                n_windows = len(df_t) - self.global_window
                if n_windows <= 0:
                    continue
                
                shape = (n_windows, self.global_window, len(features))
                strides = (feat_scaled.strides[0], feat_scaled.strides[0], feat_scaled.strides[1])
                windows = np.lib.stride_tricks.as_strided(feat_scaled, shape=shape, strides=strides)
                
                # 🚀 VECTORIZED: Batch inference (all windows at once)
                x_batch = torch.tensor(windows, dtype=torch.float32).to(self.device)
                t_batch = torch.full((n_windows,), ticker_id, dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    preds_batch = self.model(t_batch, x_batch).cpu().numpy()  # Shape: (n_windows, n_horizons)
                
                # Convert from log-vol to realized scale
                preds_realized = np.exp(preds_batch)
                
                # Align with dates and realized values
                window_dates = df_t["date"].iloc[self.global_window:].values
                
                # Build realized values matrix
                realized_matrix = np.column_stack([
                    df_t[f"realized_shift_{h}d"].iloc[self.global_window:].values
                    for h in horizons
                ])
                
                # Filter out rows with all NaN realized values
                valid_mask = ~np.all(np.isnan(realized_matrix), axis=1)
                
                if np.any(valid_mask):
                    preds_all.append(preds_realized[valid_mask])
                    actuals_all.append(realized_matrix[valid_mask])
                    dates_all.extend(window_dates[valid_mask])
                    tickers_all.extend([ticker] * np.sum(valid_mask))


            # Convert to arrays
            if not preds_all:
                return pd.DataFrame()  # No valid predictions
                
            preds_all = np.vstack(preds_all)
            actuals_all = np.vstack(actuals_all)

            # Build standardized DataFrame
            df_out = make_forecast_df(
                preds=preds_all,
                actuals=actuals_all,
                dates=dates_all,
                tickers=tickers_all,
                horizons=horizons,
                model_name="GlobalVolForecaster"
            )

            df_out = df_out.dropna(subset=["realized_vol"]).reset_index(drop=True)
            print(
                f"✅ GlobalVolForecaster realized-aligned evaluation complete ({len(df_out)} rows)."
            )
            return df_out
        
        # 3. VolNetX (Isolated Path)
        elif self.method == "volnetx":
            if data is None: raise ValueError("VolNetX requires input DataFrame.")
            return self._predict_volnetx(data, model_name, horizons)
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
    # 🧪 VolNetX Specific Prediction Loop
    # ============================================================
    def _predict_volnetx(self, data, model_name, horizons):
        """
        Isolated prediction loop for VolNetX.
        Handles per-ticker scaling using the stored scaler dictionary.
        """
        data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
        features = ["return"] + (getattr(self.cfg, "extra_features", []) or [])
        
        # Compute targets for alignment (shifted realized vol)
        for h in horizons:
            data[f"realized_shift_{h}d"] = data.groupby("ticker")["realized_vol"].shift(-h)

        preds_all, actuals_all, dates_all, tickers_all = [], [], [], []
        self.model.eval()
        
        # Validation: Ensure we have scalers
        if not self.global_scalers:
            raise RuntimeError("Model has no scalers. Was it trained correctly?")

        # 🚀 CRITICAL FIX: Iterate tickers and look up specific scaler
        for ticker, df_t in tqdm(data.groupby("ticker"), desc="VolNetX Forecasts"):
            # Clean data
            df_t = df_t.dropna(subset=["realized_vol"] + features).reset_index(drop=True)
            
            # 1. Check Ticker Existence (Model & Scaler)
            if self.global_ticker_to_id and ticker not in self.global_ticker_to_id:
                continue
            
            # 2. Retrieve Per-Ticker Scaler
            if ticker not in self.global_scalers:
                # If we didn't see this ticker during training, we can't scale it.
                continue

            tid = self.global_ticker_to_id.get(ticker, 0)
            scaler = self.global_scalers[ticker]
            
            # 3. Apply Scaling
            # transform() returns numpy array, fillna(0.0) prevents NaN propagation
            input_data = scaler.transform(df_t[features].fillna(0.0))
            input_data = input_data.astype(np.float32)
            
            dates = df_t["date"].values
            realized_map = {h: df_t[f"realized_shift_{h}d"].values for h in horizons}
            
            if len(df_t) <= self.global_window: continue
            
            # Vectorized Sliding Window construction
            shape = (len(input_data) - self.global_window, self.global_window, len(features))
            strides = (input_data.strides[0], input_data.strides[0], input_data.strides[1])
            windows = np.lib.stride_tricks.as_strided(input_data, shape=shape, strides=strides)
            
            # To Tensor
            x_batch = torch.tensor(windows, dtype=torch.float32).to(self.device) 
            t_batch = torch.full((len(windows),), tid, dtype=torch.long).to(self.device) 
            
            # Inference
            with torch.no_grad():
                preds = self.model(t_batch, x_batch).cpu().numpy()
            
            # Inverse Log Transform (Model predicts log_vol)
            preds_realized = np.exp(preds)
            
            # Alignment Logic
            valid_indices = np.arange(self.global_window - 1, len(df_t) - 1)
            current_dates = dates[valid_indices]
            realized_matrix = np.stack([realized_map[h][valid_indices] for h in horizons], axis=1)
            
            # Filter valid targets
            mask = ~np.all(np.isnan(realized_matrix), axis=1)
            
            if np.sum(mask) > 0:
                preds_all.append(preds_realized[mask])
                actuals_all.append(realized_matrix[mask])
                dates_all.append(current_dates[mask])
                tickers_all.append(np.full(np.sum(mask), ticker))

        if not preds_all: return pd.DataFrame()
        
        return make_forecast_df(
            np.concatenate(preds_all), 
            np.concatenate(actuals_all), 
            np.concatenate(dates_all), 
            np.concatenate(tickers_all), 
            horizons, model_name
        ).dropna().reset_index(drop=True)

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
        # Explicitly construct the full feature list to match training behavior
        if arch_type == "globalvolforecaster":
            # GlobalVolForecaster uses ["return"] + extra_features
            extra_feats = getattr(self.cfg, "extra_features", None) or []
            features = ["return"] + extra_feats
            ticker_to_id = getattr(self, "global_ticker_to_id", None)
        elif arch_type == "volnetx":
            # VolNetX also uses ["return"] + extra_features
            extra_feats = getattr(self.cfg, "extra_features", None) or []
            features = ["return"] + extra_feats
            ticker_to_id = getattr(self, "global_ticker_to_id", None)
        elif arch_type == "baselstm":
            # BaseLSTM may have explicit features or use extra_features
            features = getattr(self.cfg, "features", None)
            if features is None:
                extra_feats = getattr(self.cfg, "extra_features", None) or []
                features = ["return"] + extra_feats if extra_feats else ["return"]
            ticker_to_id = {getattr(self, "ticker", "TICKER"): 0}
        else:
            # Fallback for unknown architectures
            features = getattr(self.cfg, "features", ["return"])
            ticker_to_id = {}

        # --- Call the centralized saver ---
        meta = save_checkpoint(
            model=self.model,
            cfg=self.cfg,
            version=version_tag,
            save_dir=save_dir,
            ticker_to_id=ticker_to_id,
            features=features,
            scalers=self.global_scalers,
        )

        print(f"\n✅ Model saved successfully in {save_dir}: {version_tag}")
        print("   Formats generated: .full.pkl, _bundle.pkl, .meta.json + .pth")
        print(f"   Serialized on device: {device.upper()}")
        print(
            "   🔁 Ready for reloading via: load_model(..., checkpoints_dir='models')"
        )
        return meta
