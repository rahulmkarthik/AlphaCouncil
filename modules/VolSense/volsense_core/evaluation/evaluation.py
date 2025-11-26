# ============================================================
# 📈 VolSense Unified Evaluation Framework
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.stattools import durbin_watson
from statsmodels.api import qqplot
from volsense_core.evaluation.metrics import rmse, mae, mape, r2_score, acf_sum_k10


class ModelEvaluator:
    """
    Unified evaluation object for VolSense models.

    Input: standardized forecast DataFrame with columns:
        ['asof_date','date','ticker','horizon','forecast_vol','realized_vol','model']

    Provides:
        - ticker × horizon metrics
        - horizon-level summary
        - residual diagnostics
        - regime/time-slice performance
        - visualization suite
    """

    def __init__(self, eval_df: pd.DataFrame, model_name="UnknownModel"):
        """
        Initialize the evaluator with a standardized forecast-evaluation DataFrame.

        :param eval_df: Evaluation DataFrame containing forecasts and realized values.
        :type eval_df: pandas.DataFrame
        :param model_name: Label used in titles/exports for identification.
        :type model_name: str
        :return: None
        :rtype: None
        """
        self.df = eval_df.copy()
        self.model_name = model_name
        self.metrics_df = None
        self.summary_df = None

    # --------------------------------------------------------
    # 🧮 Core Metrics
    # --------------------------------------------------------
    def compute_metrics(self):
        """
        Compute per-ticker, per-horizon performance metrics.

        Calculates RMSE, MAE, MAPE, R², correlation, Durbin–Watson (DW),
        and sum of autocorrelation up to lag-10 for residuals.

        :return: Ticker × horizon metrics table.
        :rtype: pandas.DataFrame
        """
        metrics = []
        for (t, h), g in self.df.groupby(["ticker", "horizon"]):
            g = g.dropna(subset=["forecast_vol", "realized_vol"])
            if len(g) < 5:
                continue
            y_true, y_pred = g["realized_vol"].values, g["forecast_vol"].values
            resid = y_true - y_pred

            metrics.append(
                {
                    "ticker": t,
                    "horizon": h,
                    "RMSE": rmse(y_true, y_pred),
                    "MAE": mae(y_true, y_pred),
                    "MAPE": mape(y_true, y_pred),
                    "R2": r2_score(y_true, y_pred),
                    "Corr": np.corrcoef(y_true, y_pred)[0, 1],
                    "DW": durbin_watson(resid),
                    "ACF_SumSq": acf_sum_k10(resid),
                }
            )
        self.metrics_df = pd.DataFrame(metrics)
        return self.metrics_df

    # --------------------------------------------------------
    # 📊 Summary by Horizon
    # --------------------------------------------------------
    def summarize(self):
        """
        Aggregate metrics across tickers for each horizon.

        If metrics are not yet computed, runs compute_metrics() first.

        :return: Horizon-level summary with mean RMSE, MAE, MAPE, R², Corr, and DW.
        :rtype: pandas.DataFrame
        """
        if self.metrics_df is None:
            self.compute_metrics()
        self.summary_df = (
            self.metrics_df.groupby("horizon")
            .agg(
                {
                    "RMSE": "mean",
                    "MAE": "mean",
                    "MAPE": "mean",
                    "R2": "mean",
                    "Corr": "mean",
                    "DW": "mean",
                }
            )
            .reset_index()
        )
        print(f"\n📈 Horizon-Level Summary for {self.model_name}")
        display(self.summary_df.round(4))
        return self.summary_df

    # --------------------------------------------------------
    # ⏳ Regime/Time-Slice Evaluation
    # --------------------------------------------------------
    def regime_summary(self, freq="M"):
        """
        Evaluate performance over time slices (e.g., monthly, quarterly).

        :param freq: Pandas offset alias for period grouping (e.g., 'M', 'Q', 'Y').
        :type freq: str
        :return: Time-slice metrics per horizon with columns ['horizon','period','R2','Corr','RMSE'].
        :rtype: pandas.DataFrame
        """
        df = self.df.copy()
        df["period"] = pd.to_datetime(df["date"]).dt.to_period(freq).dt.to_timestamp()
        slices = []
        for (h, p), g in df.groupby(["horizon", "period"]):
            if len(g) < 10:
                continue
            slices.append(
                {
                    "horizon": h,
                    "period": p,
                    "R2": r2_score(g["realized_vol"], g["forecast_vol"]),
                    "Corr": np.corrcoef(g["realized_vol"], g["forecast_vol"])[0, 1],
                    "RMSE": rmse(g["realized_vol"], g["forecast_vol"]),
                }
            )
        regime_df = pd.DataFrame(slices)
        plt.figure(figsize=(10, 5))
        sns.lineplot(x="period", y="R2", hue="horizon", data=regime_df, marker="o")
        plt.title(f"{self.model_name}: R² Over Time (Regime Robustness)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return regime_df

    # --------------------------------------------------------
    # 📈 Plotting Utilities
    # --------------------------------------------------------
    def plot_true_vs_pred(self, horizon):
        """
        Scatter plot of realized vs forecast volatility for a given horizon.

        :param horizon: Forecast horizon to visualize.
        :type horizon: int
        :return: None
        :rtype: None
        """
        g = self.df[self.df["horizon"] == horizon].dropna(
            subset=["forecast_vol", "realized_vol"]
        )
        plt.figure(figsize=(5, 5))
        sns.scatterplot(x="realized_vol", y="forecast_vol", data=g, s=15, alpha=0.6)
        lims = [
            min(g["realized_vol"].min(), g["forecast_vol"].min()),
            max(g["realized_vol"].max(), g["forecast_vol"].max()),
        ]
        plt.plot(lims, lims, "r--")
        plt.title(f"True vs Predicted Vol – {horizon}d")
        plt.xlabel("Realized Vol")
        plt.ylabel("Forecast Vol")
        plt.show()

    def plot_residual_distribution(self, horizon):
        """
        Plot histogram and KDE of residuals for a given horizon.

        Residual is defined as forecast_vol - realized_vol.

        :param horizon: Forecast horizon to visualize.
        :type horizon: int
        :return: None
        :rtype: None
        """
        g = self.df[self.df["horizon"] == horizon]
        resid = g["forecast_vol"] - g["realized_vol"]
        plt.figure(figsize=(8, 4))
        sns.histplot(resid, bins=40, kde=True)
        plt.title(f"Residual Distribution – {horizon}d")
        plt.show()

    def plot_qq(self, horizon):
        """
        QQ plot of residuals against the normal distribution for a given horizon.

        Residual is defined as forecast_vol - realized_vol.

        :param horizon: Forecast horizon to visualize.
        :type horizon: int
        :return: None
        :rtype: None
        """
        g = self.df[self.df["horizon"] == horizon]
        resid = g["forecast_vol"] - g["realized_vol"]
        qqplot(resid, line="45", fit=True)
        plt.title(f"QQ Plot – {horizon}d")
        plt.show()

    def plot_best_worst(self, horizon, top_n=10):
        """
        Horizontal bar charts for top and bottom tickers by R² for a given horizon.

        If metrics are not computed yet, runs compute_metrics() first.

        :param horizon: Forecast horizon to rank.
        :type horizon: int
        :param top_n: Number of best and worst tickers to display.
        :type top_n: int
        :return: None
        :rtype: None
        """
        if self.metrics_df is None:
            self.compute_metrics()
        hdf = self.metrics_df[self.metrics_df["horizon"] == horizon]
        top = hdf.nlargest(top_n, "R2")
        worst = hdf.nsmallest(top_n, "R2")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.barplot(y="ticker", x="R2", data=top, ax=ax[0], color="green")
        sns.barplot(y="ticker", x="R2", data=worst, ax=ax[1], color="red")
        ax[0].set_title(f"Top {top_n} by R² – {horizon}d")
        ax[1].set_title(f"Worst {top_n} by R² – {horizon}d")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 💾 Save / Export
    # --------------------------------------------------------
    def save_metrics(self, save_path):
        """
        Save computed ticker × horizon metrics to CSV.

        If metrics are not yet computed, prints a warning instead of raising.

        :param save_path: Filesystem path for the CSV export.
        :type save_path: str
        :return: None
        :rtype: None
        """
        if self.metrics_df is not None:
            self.metrics_df.to_csv(save_path, index=False)
            print(f"💾 Saved tickerwise metrics to {save_path}")
        else:
            print("⚠️ Metrics not yet computed. Run .compute_metrics() first.")

    # --------------------------------------------------------
    # 🚀 Quick Evaluation Workflow
    # --------------------------------------------------------
    def run_full_evaluation(self, save_dir=None):
        """
        Run the full evaluation workflow: compute, summarize, visualize, and optionally save.

        Produces scatter, residual, QQ, and best/worst plots per horizon, computes a regime
        time-series summary, and optionally writes metrics to disk.

        :param save_dir: Directory to save metrics CSV; if None, no files are written.
        :type save_dir: str, optional
        :return: Tuple of (metrics_df, summary_df, regime_df).
        :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        """
        print(f"\n🚀 Running full evaluation for {self.model_name}")
        self.compute_metrics()
        self.summarize()
        horizons = sorted(self.df["horizon"].unique())

        for h in horizons:
            self.plot_true_vs_pred(h)
            self.plot_residual_distribution(h)
            self.plot_qq(h)
            self.plot_best_worst(h)

        regime_df = self.regime_summary()
        if save_dir:
            self.save_metrics(f"{save_dir}/{self.model_name}_tickerwise_metrics.csv")
        print("✅ Evaluation complete.")
        return self.metrics_df, self.summary_df, regime_df
