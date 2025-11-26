"""
VolSense Inference — Predictor
------------------------------
Helper utilities to run single and batched predictions with pretrained VolSense models.

Responsibilities:
- Resolve the expected feature list from model metadata or user override.
- Apply per-ticker scaling using training-time scalers, with sensible fallbacks.
- Execute model forward passes with optional ticker ID conditioning.
- Produce per-ticker predictions and aggregate batch results.
- Attach latest realized volatility for comparison and derive simple diffs.

Used by volsense_inference.forecast_engine.Forecast.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 🔧 Helper functions
# ---------------------------------------------------------------------------


def _get_feature_list(
    meta: Dict[str, Any], user_feats: Optional[List[str]] = None
) -> List[str]:
    """
    Determine the feature list to use for inference.

    Precedence:
      1) user_feats when provided,
      2) meta["features"] when present,
      3) ['return'] + meta['extra_features'] if defined,
      4) fallback to ['return'].

    :param meta: Model metadata/config dictionary.
    :type meta: dict
    :param user_feats: Optional explicit list of feature names to use.
    :type user_feats: list[str], optional
    :return: Ordered feature names expected by the model.
    :rtype: list[str]
    """
    if user_feats:
        return user_feats
    if "features" in meta and meta["features"]:
        return meta["features"]
    if "extra_features" in meta and meta["extra_features"]:
        return ["return"] + meta["extra_features"]
    return ["return"]


def _scale_features(
    df_t: pd.DataFrame, feats: List[str], ticker: str, scalers: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Apply per-ticker feature scaling with training scalers, with fallback.

    If a scaler is available for the ticker, applies it. Otherwise, falls back to
    in-sample mean-std normalization over the recent window. Any transform errors
    (e.g., shape mismatch) will silently fallback to raw features.

    :param df_t: Ticker-filtered DataFrame to be scaled.
    :type df_t: pandas.DataFrame
    :param feats: Feature columns to scale.
    :type feats: list[str]
    :param ticker: Ticker symbol used to select the appropriate scaler.
    :type ticker: str
    :param scalers: Optional dict mapping ticker -> fitted StandardScaler-like object.
    :type scalers: dict[str, Any], optional
    :return: DataFrame with scaled feature columns.
    :rtype: pandas.DataFrame
    """
    df_scaled = df_t.copy()
    if scalers is None:
        return df_scaled
    sc = scalers.get(ticker)
    if sc is None:
        # fallback: mean-std normalize with recent window
        df_scaled[feats] = (df_scaled[feats] - df_scaled[feats].mean()) / (
            df_scaled[feats].std() + 1e-8
        )
        return df_scaled
    try:
        df_scaled[feats] = sc.transform(df_scaled[feats].astype(float).fillna(0.0))
    except Exception:
        # if scaler incompatible, fallback to raw
        pass
    return df_scaled


# --- replace _forward with this robust variant ---
def _forward(model, X: torch.Tensor, tid_tensor: Optional[torch.Tensor]) -> np.ndarray:
    """
    Execute a forward pass, optionally providing ticker IDs to the model.

    Tries model(tid, X) first; if the signature does not accept ticker IDs,
    falls back to model(X).

    :param model: Loaded inference model (PyTorch module or compatible callable).
    :type model: Any
    :param X: Input tensor of shape [B, W, F].
    :type X: torch.Tensor
    :param tid_tensor: Optional tensor of ticker IDs of shape [B].
    :type tid_tensor: torch.Tensor, optional
    :return: Model outputs as a NumPy array (flattened to 1D when appropriate).
    :rtype: numpy.ndarray
    """
    model.eval()
    with torch.no_grad():
        # Try (tid, X) if we have ids
        if tid_tensor is not None:
            try:
                out = model(tid_tensor, X)
                return out.cpu().numpy().reshape(-1)
            except TypeError:
                pass
            try:
                out = model(X, tid_tensor)
                return out.cpu().numpy().reshape(-1)
            except TypeError:
                pass
        # Try classic single-arg
        try:
            out = model(X)
            return out.cpu().numpy().reshape(-1)
        except TypeError:
            pass
        # Last resort for models that accept (None, X)
        out = model(None, X)
        return out.cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# 🎯 Core prediction logic
# ---------------------------------------------------------------------------

# TODO fix flat outputs in volatility forecsting. Likely a problem with feature parsing


def predict_single(
    model: Any,
    meta: Dict[str, Any],
    df: pd.DataFrame,
    ticker: str,
    scalers: Optional[Dict[str, Any]] = None,
    ticker_to_id: Optional[Dict[str, int]] = None,
    features: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Predict multi-horizon volatility for a single ticker.

    Aligns features to the model schema (adding missing features as zeros, dropping extras),
    scales inputs with the appropriate per-ticker scaler, runs the model, and formats
    outputs as {'ticker', 'pred_vol_1', 'pred_vol_5', ...} using horizons from meta.

    :param model: Loaded model used for inference.
    :type model: Any
    :param meta: Model metadata/config including 'window', 'horizons', and optional 'config.target_col'.
    :type meta: dict
    :param df: Feature DataFrame containing rows for multiple tickers and dates.
    :type df: pandas.DataFrame
    :param ticker: Ticker symbol to predict.
    :type ticker: str
    :param scalers: Optional dict mapping ticker -> fitted scaler for features.
    :type scalers: dict[str, Any], optional
    :param ticker_to_id: Optional mapping from ticker to integer ID expected by the model.
    :type ticker_to_id: dict[str, int], optional
    :param features: Optional explicit feature list to override meta-derived list.
    :type features: list[str], optional
    :raises ValueError: If there is insufficient data to form the input window.
    :return: Dictionary with 'ticker' and predicted volatility per horizon.
    :rtype: dict[str, float]
    """
    feats = _get_feature_list(meta, features)
    # 🔹 Align DataFrame columns with the model’s expected features
    missing = [f for f in feats if f not in df.columns]
    if missing:
        print(f"⚠️ {ticker}: missing features {missing}, filling with 0.")
        for f in missing:
            df[f] = 0.0

    extra = [
        f for f in df.columns if f not in feats + ["date", "ticker", "realized_vol"]
    ]
    if extra:
        df = df.drop(columns=extra)

    window = int(meta.get("window", meta.get("lookback", 30)))
    df_t = df[df["ticker"] == ticker].sort_values("date")

    if len(df_t) < window:
        raise ValueError(
            f"{ticker}: insufficient data ({len(df_t)} rows, need {window})"
        )

    df_t = _scale_features(df_t, feats, ticker, scalers)
    X_df = df_t.iloc[-window:][feats]
    X = torch.tensor(X_df.values, dtype=torch.float32).unsqueeze(0)

    # default to 0 if ticker not in dict
    tid_tensor = torch.tensor(
        [ticker_to_id.get(ticker, 0)] if ticker_to_id else [0], dtype=torch.long
    )

    yhat = _forward(model, X, tid_tensor)

    # --- EXPONENTIATION SAFETY LOGIC ---
    # Priority order:
    #   1. meta["config"]["target_col"]
    #   2. meta["target_col"]
    #   3. default assume log if model name smells like global / v5xx
    tc_from_config = meta.get("config", {}).get("target_col")
    tc_fallback = meta.get("target_col", None)

    target_col_name = tc_from_config or tc_fallback or ""

    # heuristic: many of our models forecast log-vol
    looks_like_log = (
        "_log" in target_col_name.lower()
        or target_col_name.lower().endswith("log")
        or "vol_log" in target_col_name.lower()
    )

    # ultimate fallback for older global models where we forgot to store target_col:
    # global models (v5xx) *always* train on realized_vol_log right now.
    if not looks_like_log:
        arch_name = str(meta.get("arch", "")).lower()
        if "globalvolforecaster" in arch_name:
            looks_like_log = True

    if looks_like_log:
        yhat = np.exp(yhat)

    horizons = meta.get("horizons", [1])
    preds = {
        f"pred_vol_{h}": float(yhat[i]) if i < len(yhat) else np.nan
        for i, h in enumerate(horizons)
    }

    return {"ticker": ticker, **preds}


def predict_batch(
    model: Any,
    meta: Dict[str, Any],
    df: pd.DataFrame,
    tickers: List[str],
    scalers: Optional[Dict[str, Any]] = None,
    ticker_to_id: Optional[Dict[str, int]] = None,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run predictions for a list of tickers and return a tidy DataFrame.

    Exceptions for individual tickers are caught and printed; failed tickers are skipped.

    :param model: Loaded model used for inference.
    :type model: Any
    :param meta: Model metadata/config dictionary.
    :type meta: dict
    :param df: Feature DataFrame containing multiple tickers.
    :type df: pandas.DataFrame
    :param tickers: List of ticker symbols to predict.
    :type tickers: list[str]
    :param scalers: Optional dict mapping ticker -> fitted scaler.
    :type scalers: dict[str, Any], optional
    :param ticker_to_id: Optional mapping from ticker to integer ID expected by the model.
    :type ticker_to_id: dict[str, int], optional
    :param features: Optional explicit feature list to override meta-derived list.
    :type features: list[str], optional
    :return: DataFrame with columns ['ticker','pred_vol_...'] for available horizons.
    :rtype: pandas.DataFrame
    """
    rows = []
    for t in tqdm(tickers, desc="Forecasting"):
        try:
            rows.append(
                predict_single(model, meta, df, t, scalers, ticker_to_id, features)
            )
        except Exception as e:
            print(f"⚠️ {t}: {e}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 📊 Post-processing utilities
# ---------------------------------------------------------------------------


def attach_realized(df_preds: pd.DataFrame, df_recent: pd.DataFrame) -> pd.DataFrame:
    """
    Attach the latest realized volatility to each forecast row and compute diffs.

    When df_recent contains 'realized_vol', merges the most recent value per ticker
    into df_preds, then computes simple diagnostics:
      - vol_diff = pred_vol_1 - realized_vol
      - vol_direction = sign(vol_diff)

    If 'realized_vol' is not available in df_recent, the input df_preds is returned unchanged.

    :param df_preds: DataFrame of predictions with one row per ticker.
    :type df_preds: pandas.DataFrame
    :param df_recent: Recent feature DataFrame containing realized volatility history.
    :type df_recent: pandas.DataFrame
    :return: DataFrame augmented with 'realized_vol' and derived diagnostics when available.
    :rtype: pandas.DataFrame
    """

    if df_preds is None or not hasattr(df_preds, "columns"):
        print("⚠️ df_preds is None or not a DataFrame")
        return pd.DataFrame()
    if "realized_vol" not in df_recent.columns:
        return df_preds

    # Get most recent realized vol per ticker
    last = (
        df_recent.sort_values("date")
        .groupby("ticker")["realized_vol"]
        .last()
        .rename("realized_vol")
        .reset_index()
    )

    out = df_preds.merge(last, on="ticker", how="left")

    # Compute diffs
    if "pred_vol_1" in out.columns:
        out["vol_diff"] = out["pred_vol_1"] - out["realized_vol"]
        out["vol_direction"] = np.sign(out["vol_diff"])

    # Reorder columns so realized_vol is first
    base_cols = ["ticker", "realized_vol"]
    pred_cols = [c for c in out.columns if c.startswith("pred_vol_")]
    extra_cols = [c for c in ["vol_diff", "vol_direction"] if c in out.columns]
    out = out[base_cols + pred_cols + extra_cols]

    return out
