# volsense_core/models/garch_methods.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Iterable, Optional, Tuple, Union, Dict, Any

from arch import arch_model


ArrayLike = Union[np.ndarray, pd.Series, Iterable[float]]


# ============================================================
# 🧩 Configuration
# ============================================================
@dataclass
class GARCHConfig:
    """
    Configuration for ARCH-family forecasters.

    :param model: Model family to use: 'garch' (GARCH), 'egarch' (EGARCH), or 'gjr' (GJR/TGARCH).
    :type model: str
    :param p: ARCH order.
    :type p: int
    :param q: GARCH order.
    :type q: int
    :param o: Asymmetry order (used for GJR when > 0). Ignored for EGARCH.
    :type o: int
    :param dist: Innovation distribution, e.g., 'normal', 't', 'skewt', or 'ged'.
    :type dist: str
    :param mean: Mean model, typically 'Zero' for daily returns.
    :type mean: str
    :param scale: Factor to scale returns before fitting (e.g., 100.0 for percent returns).
    :type scale: float
    :param annualize: If True, convert daily sigma to annualized volatility using sqrt(252).
    :type annualize: bool
    :param fit_kwargs: Extra keyword arguments passed to arch's .fit() (e.g., {'disp': 'off'}).
    :type fit_kwargs: dict, optional
    :raises ValueError: If an unsupported model name is provided.
    """

    model: str = "garch"
    p: int = 1
    q: int = 1
    o: int = 0
    dist: str = "normal"
    mean: str = "Zero"
    scale: float = 100.0
    annualize: bool = True
    fit_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        self.model = str(self.model).lower()
        if self.model not in {"garch", "egarch", "gjr"}:
            raise ValueError("model must be one of {'garch','egarch','gjr'}")
        if self.fit_kwargs is None:
            # quiet + deterministic defaults
            self.fit_kwargs = dict(disp="off", update_freq=0)


# ============================================================
# 🧠 Forecaster
# ============================================================
class ARCHForecaster:
    """
    Thin, safe wrapper around `arch.arch_model` supporting GARCH, EGARCH, and GJR.

    Usage
    -----
    >>> cfg = GARCHConfig(model="gjr", p=1, o=1, q=1, dist="t")
    >>> f = ARCHForecaster(cfg).fit(ret)          # ret: pd.Series or np.array
    >>> sigma_next = f.predict(horizon=5)         # np.ndarray length 5 (annualized by default)
    >>> roll = f.rolling_forecast(ret)            # DataFrame with one-step-ahead forecasts
    """

    def __init__(self, config: Optional[GARCHConfig] = None, **kwargs):
        # Allow dict-like construction via kwargs
        self.cfg = config or GARCHConfig(**kwargs)
        self._fitted_ret_scaled: Optional[np.ndarray] = None
        self._result = None

    # --------------- internal helpers ----------------
    def _to_series(self, x: ArrayLike) -> Tuple[np.ndarray, Optional[pd.Index]]:
        """
        Convert input returns to a clean numpy array and preserve index if Series.

        :param x: Input returns as array-like or pandas Series.
        :type x: ArrayLike
        :return: Tuple of (values array without NaNs, original index if provided as Series else None).
        :rtype: tuple[numpy.ndarray, pandas.Index | None]
        """
        if isinstance(x, pd.Series):
            vals = x.dropna().astype(float).values
            idx = x.dropna().index
            return vals, idx
        x = np.asarray(list(x), dtype=float)
        x = x[~np.isnan(x)]
        return x, None

    def _build_model(self, y_scaled: np.ndarray):
        """
        Construct the underlying arch model based on configuration.

        :param y_scaled: Scaled return series (already multiplied by cfg.scale).
        :type y_scaled: numpy.ndarray
        :return: Configured ARCH model specification ready to fit.
        :rtype: arch.univariate.base.ARCHModel
        """
        if self.cfg.model == "egarch":
            vol = "EGARCH"
            p, q, o = self.cfg.p, self.cfg.q, 0
        elif self.cfg.model == "gjr":
            # GJR/TGARCH is specified in arch as GARCH with o > 0
            vol = "GARCH"
            p, q, o = self.cfg.p, self.cfg.q, max(1, self.cfg.o or 1)
        else:
            vol = "GARCH"
            p, q, o = self.cfg.p, self.cfg.q, 0

        return arch_model(
            y_scaled, vol=vol, p=p, o=o, q=q, dist=self.cfg.dist, mean=self.cfg.mean
        )

    def _postprocess_sigma(self, sigma: np.ndarray) -> np.ndarray:
        """
        Undo scaling and optionally annualize sigma.

        :param sigma: Sigma values on the scaled, daily basis.
        :type sigma: numpy.ndarray
        :return: Sigma on original scale; annualized if configured.
        :rtype: numpy.ndarray
        """
        sigma = sigma / self.cfg.scale
        if self.cfg.annualize:
            sigma = sigma * np.sqrt(252.0)
        return sigma

    # --------------- public API ----------------
    def fit(self, returns: ArrayLike) -> "ARCHForecaster":
        """
        Fit the selected ARCH-family model to a return series.

        :param returns: Daily simple returns (e.g., price.pct_change()).
        :type returns: ArrayLike
        :return: Self, with fitted result accessible via .result.
        :rtype: ARCHForecaster
        """
        y, _ = self._to_series(returns)
        y_scaled = y * self.cfg.scale
        self._fitted_ret_scaled = y_scaled

        model = self._build_model(y_scaled)
        self._result = model.fit(**self.cfg.fit_kwargs)
        return self

    @property
    def result(self):
        """
        Access the underlying ARCHModelResult after fit().

        :raises RuntimeError: If the model has not been fitted yet.
        :return: Fitted result object from arch.
        :rtype: object
        """
        if self._result is None:
            raise RuntimeError("Model not fitted yet. Call .fit() first.")
        return self._result

    def predict(self, horizon: int | list[int] = 1) -> np.ndarray:
        """
        Generate multi-horizon volatility forecasts.

        :param horizon: Single horizon or list of horizons to forecast (e.g., 1 or [1, 5, 10]).
        :type horizon: int or list[int]
        :raises RuntimeError: If called before fit() (via the .result accessor).
        :return: Forecasted sigma values (annualized if configured), shape (len(horizons),).
        :rtype: numpy.ndarray
        """
        res = self.result
        horizons = horizon if isinstance(horizon, (list, tuple)) else [horizon]
        forecasts = []

        for h in horizons:
            # Use analytic or simulation forecast depending on model type
            if self.cfg.model == "egarch" and h > 1:
                f = res.forecast(
                    horizon=h, method="simulation", reindex=False, simulations=500
                )
            else:
                f = res.forecast(horizon=h, reindex=False)

            # Extract the last-step variance forecast
            try:
                sigma = np.sqrt(np.asarray(f.variance.iloc[-1].values, dtype=float))
                sigma = self._postprocess_sigma(sigma)
                forecasts.append(float(sigma[-1]))
            except Exception as e:
                print(f"⚠️ GARCH forecast failed for horizon={h}: {e}")
                forecasts.append(np.nan)

        forecasts = np.array(forecasts, dtype=float)
        self._last_forecast_ = forecasts  # cache for reference
        return forecasts

    def rolling_forecast(
        self,
        returns: ArrayLike,
        start: Optional[int] = None,
        min_obs: Optional[int] = None,
        refit_every: int = 1,
    ) -> Union[pd.Series, np.ndarray]:
        """
        Produce rolling one-step-ahead forecasts using an expanding window.

        :param returns: Daily simple returns. If a Series is provided, output is a Series aligned to input dates.
        :type returns: ArrayLike
        :param start: First index to start forecasting. Defaults to max(p+q+o, 30) if None.
        :type start: int, optional
        :param min_obs: Minimum observations before first fit; overrides start when provided.
        :type min_obs: int, optional
        :param refit_every: Refit frequency in steps (1 = refit each step).
        :type refit_every: int
        :raises ValueError: If there are fewer than ~50 observations.
        :return: One-step-ahead sigma forecasts (annualized if configured), as Series or ndarray.
        :rtype: pandas.Series or numpy.ndarray
        """
        y, idx = self._to_series(returns)
        n = len(y)
        if n < 50:
            raise ValueError(
                "Not enough observations for rolling forecast (need ~50+)."
            )

        warm = max(self.cfg.p + self.cfg.q + max(0, self.cfg.o), 30)
        if min_obs is not None:
            warm = max(warm, int(min_obs))
        if start is not None:
            warm = max(warm, int(start))

        sigmas = np.full(n, np.nan, dtype=float)
        last_res = None

        i = warm
        while i < n:
            # Refit on expanding window y[:i]
            if (i == warm) or ((i - warm) % refit_every == 0) or (last_res is None):
                m = self._build_model(y[:i] * self.cfg.scale)
                last_res = m.fit(**self.cfg.fit_kwargs)

            f = last_res.forecast(horizon=1, reindex=False)
            sigma1 = np.sqrt(float(f.variance.iloc[-1, 0]))
            sigmas[i] = sigma1
            i += 1

        sigmas = self._postprocess_sigma(sigmas)
        if idx is not None:
            # forecast for time t uses info up to t-1 → align to original index
            out = pd.Series(sigmas, index=idx, name="sigma_forecast")
            return out
        return sigmas

    # convenience
    def get_config(self) -> Dict[str, Any]:
        """
        Get a dictionary copy of the current configuration.

        :return: Configuration as a plain dictionary.
        :rtype: dict
        """
        return asdict(self.cfg)


# ============================================================
# 🔧 Convenience top-level helpers (backward-friendly)
# ============================================================
def fit_arch(
    returns: ArrayLike,
    model_type: str = "garch",
    dist: str = "normal",
    p: int = 1,
    q: int = 1,
    o: int = 0,
    mean: str = "Zero",
    scale: float = 100.0,
    annualize: bool = True,
    **fit_kwargs,
):
    """
    Quick-fit a GARCH/EGARCH/GJR model and return the fitted result.

    :param returns: Daily simple returns to fit on.
    :type returns: ArrayLike
    :param model_type: Model family: 'garch', 'egarch', or 'gjr'.
    :type model_type: str
    :param dist: Innovation distribution, e.g., 'normal', 't', 'skewt', 'ged'.
    :type dist: str
    :param p: ARCH order.
    :type p: int
    :param q: GARCH order.
    :type q: int
    :param o: Asymmetry order (GJR) when > 0; ignored for EGARCH.
    :type o: int
    :param mean: Mean model, typically 'Zero' for daily returns.
    :type mean: str
    :param scale: Factor to scale returns before fitting.
    :type scale: float
    :param annualize: If True, annualize sigma via sqrt(252).
    :type annualize: bool
    :param fit_kwargs: Extra keyword args forwarded to arch .fit().
    :type fit_kwargs: dict
    :return: Fitted ARCH model result object.
    :rtype: object
    """
    cfg = GARCHConfig(
        model=model_type,
        p=p,
        q=q,
        o=o,
        dist=dist,
        mean=mean,
        scale=scale,
        annualize=annualize,
        fit_kwargs=fit_kwargs or dict(disp="off", update_freq=0),
    )
    return ARCHForecaster(cfg).fit(returns).result


def forecast_arch(
    fitted_model, horizon: int = 1, scale: float = 100.0, annualize: bool = True
):
    """
    Generate volatility forecasts from a fitted ARCHModelResult.

    :param fitted_model: Result object returned by arch after fitting.
    :type fitted_model: object
    :param horizon: Number of steps to forecast ahead.
    :type horizon: int
    :param scale: Scale factor used during fitting to unscale sigma.
    :type scale: float
    :param annualize: If True, annualize sigma via sqrt(252).
    :type annualize: bool
    :return: Array of forecasted sigma values of length `horizon`.
    :rtype: numpy.ndarray
    """
    f = fitted_model.forecast(horizon=horizon, reindex=False)
    sigma = np.sqrt(np.asarray(f.variance.iloc[-1].values, dtype=float)) / scale
    if annualize:
        sigma = sigma * np.sqrt(252.0)
    return sigma


# ============================================================
# 🔄 Unified Output Standardizer
# ============================================================
def standardize_outputs(
    dates,
    tickers,
    forecast_vols,
    realized_vols=None,
    model_name="UnknownModel",
    horizons=None,
):
    """
    Standardize model outputs for evaluation/backtesting.

    :param dates: Forecast or validation dates.
    :type dates: list or numpy.ndarray
    :param tickers: Corresponding tickers for each row.
    :type tickers: list or numpy.ndarray
    :param forecast_vols: Predicted volatilities, shape (N, H) or (N,).
    :type forecast_vols: numpy.ndarray
    :param realized_vols: True realized volatilities, same shape as forecast_vols.
    :type realized_vols: numpy.ndarray or list, optional
    :param model_name: Model identifier, e.g., 'BaseLSTM', 'GlobalVolForecaster', 'ARCHForecaster'.
    :type model_name: str
    :param horizons: Forecast horizons, e.g., [1, 5, 10]. Defaults to 1..H if None.
    :type horizons: list[int], optional
    :return: DataFrame with columns ['date','ticker','horizon','forecast_vol','realized_vol','model'].
    :rtype: pandas.DataFrame
    """
    dates = np.asarray(dates)
    tickers = np.asarray(tickers)
    forecast_vols = np.atleast_2d(forecast_vols)
    n, h = forecast_vols.shape
    if realized_vols is None:
        realized_vols = np.full_like(forecast_vols, np.nan)
    else:
        realized_vols = np.atleast_2d(realized_vols)

    if horizons is None:
        horizons = list(range(1, h + 1))

    records = []
    for i in range(n):
        for j, horizon in enumerate(horizons):
            records.append(
                dict(
                    date=pd.to_datetime(dates[i]),
                    ticker=tickers[i],
                    horizon=horizon,
                    forecast_vol=forecast_vols[i, j],
                    realized_vol=realized_vols[i, j],
                    model=model_name,
                )
            )
    return pd.DataFrame.from_records(records)
