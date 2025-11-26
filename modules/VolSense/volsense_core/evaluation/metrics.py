# volsense_pkg/utils/metrics.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf


# =====================================================
# --- Existing Low-Level GARCH Utilities (KEEP THESE) ---
# =====================================================
def rolling_garch_forecast(model, returns, horizon=1):
    """
    Generate rolling 1-step-ahead volatility forecasts using an ARCH/GARCH model.

    At each time t, fits the provided model on returns[:t] and produces a 1-step
    ahead forecast for t+1. The output aligns to returns[horizon:].

    :param model: An unfitted arch_model specification (e.g., arch.univariate.arch_model instance).
    :type model: object
    :param returns: 1D array-like of returns used to fit and roll the forecast.
    :type returns: array-like
    :param horizon: Starting offset before rolling begins; forecasts are always 1-step ahead.
    :type horizon: int
    :return: Array of forecasted conditional volatilities (same units as returns' std).
    :rtype: numpy.ndarray
    """
    forecasts = []
    n = len(returns)
    for i in range(horizon, n):
        window_data = returns[:i]
        res = model.fit(disp="off")
        fcast = res.forecast(horizon=1, reindex=False)
        vol = np.sqrt(fcast.variance.iloc[-1, 0])
        forecasts.append(vol)
    return np.array(forecasts)


# =====================================================
# --- Universal Evaluation Metrics (REPLACE OLD ONES) ---
# =====================================================
def rmse(y_true, y_pred):
    """
    Root Mean Squared Error between true and predicted values.

    :param y_true: Ground-truth values.
    :type y_true: array-like
    :param y_pred: Predicted values.
    :type y_pred: array-like
    :return: RMSE value.
    :rtype: float
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Mean Absolute Error between true and predicted values.

    :param y_true: Ground-truth values.
    :type y_true: array-like
    :param y_pred: Predicted values.
    :type y_pred: array-like
    :return: MAE value.
    :rtype: float
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error between true and predicted values (in percent).

    Uses a small epsilon in the denominator for numerical stability.

    :param y_true: Ground-truth values.
    :type y_true: array-like
    :param y_pred: Predicted values.
    :type y_pred: array-like
    :return: MAPE value in percentage.
    :rtype: float
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100


def r2_score(y_true, y_pred):
    """
    Coefficient of determination R².

    :param y_true: Ground-truth values.
    :type y_true: array-like
    :param y_pred: Predicted values.
    :type y_pred: array-like
    :return: R² score in [-inf, 1].
    :rtype: float
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def acf_sum_k10(resid, m=10):
    """
    Sum of squared autocorrelations up to lag m.

    Stable alternative to Ljung–Box p-values. Smaller values imply weaker serial correlation.

    :param resid: Residual series to test for autocorrelation.
    :type resid: array-like
    :param m: Maximum lag to include in the sum (default 10).
    :type m: int
    :return: Sum of squared ACF values from lags 1..m; NaN if too few points or near-constant series.
    :rtype: float
    """
    resid = np.asarray(resid)
    if len(resid) < m + 2 or np.std(resid) < 1e-8:
        return np.nan
    vals = acf(resid, nlags=m, fft=True)[1:]  # skip lag 0
    return float(np.sum(vals**2))


# =====================================================
# --- Unified Forecast Evaluation Interface ---
# =====================================================
def evaluate_forecasts(df):
    """
    Compute core evaluation metrics for volatility forecasts.

    Expects a DataFrame with columns ['realized_vol', 'forecast_vol'] and returns
    a dictionary of RMSE, MAE, MAPE, and R².

    :param df: Input table containing realized and forecast volatility.
    :type df: pandas.DataFrame
    :raises TypeError: If df is not a pandas DataFrame.
    :raises ValueError: If required columns are missing.
    :return: Mapping of metric name to value.
    :rtype: dict[str, float]
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if not all(c in df.columns for c in ["realized_vol", "forecast_vol"]):
        raise ValueError(
            "DataFrame must contain 'realized_vol' and 'forecast_vol' columns."
        )

    metrics = {
        "RMSE": rmse(df["realized_vol"], df["forecast_vol"]),
        "MAE": mae(df["realized_vol"], df["forecast_vol"]),
        "MAPE": mape(df["realized_vol"], df["forecast_vol"]),
        "R2": r2_score(df["realized_vol"], df["forecast_vol"]),
    }
    return metrics


# =====================================================
# --- Backward Compatibility: old evaluate_forecast() ---
# =====================================================
def evaluate_forecast(y_true, y_pred):
    """
    Legacy wrapper around evaluate_forecasts for arrays.

    :param y_true: Ground-truth values.
    :type y_true: array-like
    :param y_pred: Predicted values.
    :type y_pred: array-like
    :return: Mapping of metric name to value.
    :rtype: dict[str, float]
    """
    df = pd.DataFrame({"realized_vol": y_true, "forecast_vol": y_pred})
    return evaluate_forecasts(df)
