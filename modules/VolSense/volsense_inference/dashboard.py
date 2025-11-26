"""
Streamlit dashboard for VolSense.

Provides a small interactive UI to run forecasts, inspect per-ticker plots,
sector heatmaps and cross-sectional signals.
---------------------------------------

This module exposes the Streamlit application and several helper functions:

- :func:`_parse_tickers` - parse user comma-separated tickers
- :func:`_safe_style_format` - style numeric columns for display
- :func:`_detect_horizons` - detect forecast horizons from dataframe columns
- :func:`_load_forecast_model` - cached model loader
- :func:`run_volsense` - run forecasts + signals (cached)
- :func:`export_csv_button` - download button for DataFrame CSV export

Typical usage
~~~~~~~~~~~~~
Run the Streamlit app:

.. code-block:: bash

   streamlit run volsense_inference/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from typing import List

# VolSense imports
from volsense_inference.forecast_engine import Forecast  # runs models + features  📦
from volsense_inference.signal_engine import SignalEngine  # sector-aware signals
from volsense_inference.sector_mapping import get_sector_map, get_color

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VolSense — Volatility Insights",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# ⚙️ Session state setup
# ─────────────────────────────────────────────────────────────
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Overview"
if "tabs_list" not in st.session_state:
    st.session_state.tabs_list = [
        "Overview",
        "Ticker Analytics",
        "Sector View",
        "Signal Table",
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parse_tickers(s: str) -> List[str]:
    """
    Parse a comma-separated ticker string into a sorted list of unique tickers.

    :param s: Comma-separated tickers entered by the user.
    :type s: str
    :return: Sorted list of uppercase, de-duplicated ticker symbols.
    :rtype: List[str]
    """
    return sorted({t.strip().upper() for t in s.split(",") if t.strip()})


def _safe_style_format(df: pd.DataFrame, formatter):
    """
    Apply a number formatter to numeric columns of a DataFrame and return a Styler.

    This avoids formatting attempts on non-numeric columns (which would raise).

    :param df: DataFrame to format.
    :type df: pandas.DataFrame
    :param formatter: Callable or format string to apply to numeric columns.
    :return: pandas Styler with applied formatting.
    :rtype: pandas.io.formats.style.Styler
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    format_dict = {c: formatter for c in num_cols}
    return df.style.format(format_dict)


def _detect_horizons(df: pd.DataFrame) -> List[int]:
    """
    Detect integer horizon values from column names like 'pred_vol_5'.

    :param df: Forecast DataFrame containing horizon columns.
    :type df: pandas.DataFrame
    :return: Sorted list of unique horizon integers discovered.
    :rtype: List[int]
    """
    cols = [c for c in df.columns if c.startswith("pred_vol_")]
    hs = []
    for c in cols:
        try:
            hs.append(int(c.split("_")[-1].replace("d", "")))
        except Exception:
            pass
    return sorted(set(hs))


@st.cache_resource(show_spinner=False)
def _load_forecast_model(model_version: str, checkpoints_dir: str):
    """
    Cached loader for Forecast object.

    Uses Streamlit's cache_resource to avoid re-loading heavy models on each interaction.

    :param model_version: Model version/stem to load (e.g., "v109").
    :type model_version: str
    :param checkpoints_dir: Directory containing model artifacts.
    :type checkpoints_dir: str
    :return: Instantiated Forecast object ready to run.
    :rtype: volsense_inference.forecast_engine.Forecast
    """
    fcast = Forecast(
        model_version=model_version, checkpoints_dir=checkpoints_dir, start="2005-01-01"
    )
    return fcast


@st.cache_data(show_spinner=False, ttl=60 * 30)
def run_volsense(
    model_version: str, checkpoints_dir: str, start: str, tickers: List[str]
):
    """
    Run VolSense forecasts and compute analytics and signals.

    This function is cached for 30 minutes to avoid repeated expensive runs.

    :param model_version: Model version to use for forecasting.
    :type model_version: str
    :param checkpoints_dir: Directory with model artifacts.
    :type checkpoints_dir: str
    :param start: Start date string for feature fetch.
    :type start: str
    :param tickers: List of tickers to forecast.
    :type tickers: List[str]
    :return: Tuple (Forecast instance, predictions DataFrame, analytics object,
             analytics summary DataFrame, SignalEngine instance, signals DataFrame)
    :rtype: tuple
    """
    fcast = _load_forecast_model(model_version, checkpoints_dir)
    preds = fcast.run(
        tickers
    )  # attaches fcast.signals (Analytics) internally  :contentReference[oaicite:4]{index=4}
    analytics = (
        fcast.signals
    )  # Analytics(preds).compute() already called        :contentReference[oaicite:5]{index=5}
    ae_summary = (
        analytics.summary(horizon="pred_vol_5")
        if "pred_vol_5" in preds.columns
        else analytics.summary()
    )

    # Signal engine (sector-aware)
    se = SignalEngine(model_version=model_version)
    se.set_data(
        preds
    )  # coerce wide → long if needed  :contentReference[oaicite:6]{index=6}
    sig_df = se.compute_signals(enrich_with_sectors=True)

    return fcast, preds, analytics, ae_summary, se, sig_df


def _pretty_number(x, nd=3):
    """
    Human-friendly numeric formatter used by the UI.

    :param x: Numeric value (or NaN).
    :param nd: Number of decimal places to format.
    :type nd: int
    :return: Formatted string or empty string for NaN.
    :rtype: str
    """
    if pd.isna(x):
        return ""
    return f"{x:.{nd}f}"


# ──────────────────────────────────────────────────────────────────────────────
# CSV Export helper
# ──────────────────────────────────────────────────────────────────────────────


def export_csv_button(df: pd.DataFrame, filename: str, label: str = "📥 Export CSV"):
    """
    Create a Streamlit download button that exports a DataFrame as CSV.

    :param df: DataFrame to export.
    :type df: pandas.DataFrame
    :param filename: Suggested filename for the downloaded CSV.
    :type filename: str
    :param label: Button label text.
    :type label: str
    :return: None
    """
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("VolSense Dashboard")
st.sidebar.caption("Trader-facing Volatility Analytics & Signals")

TICKERS = [
    # 100 cross-sector tickers to initialize the dashboard with
    # ----- Index / ETF -----
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "GLD",
    "SLV",
    "TLT",
    "HYG",
    "EEM",
    # ----- Technology (17) -----
    "AAPL",
    "MSFT",
    "GOOG",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "AVGO",
    "AMD",
    "INTC",
    "ORCL",
    "TXN",
    "QCOM",
    "ADBE",
    "CSCO",
    "NOW",
    "INTU",
    # ----- Financials (13) -----
    "JPM",
    "BAC",
    "C",
    "WFC",
    "GS",
    "MS",
    "V",
    "MA",
    "BLK",
    "PNC",
    "USB",
    "TFC",
    "COF",
    # ----- Healthcare (13) -----
    "JNJ",
    "PFE",
    "MRK",
    "UNH",
    "ABBV",
    "ABT",
    "LLY",
    "BMY",
    "TMO",
    "CVS",
    "AMGN",
    "REGN",
    "MDT",
    # ----- Energy / Materials (12) -----
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "HAL",
    "EOG",
    "BHP",
    "RIO",
    "FCX",
    "LIN",
    "APD",
    "NUE",
    # ----- Consumer Discretionary (11) -----
    "TSLA",
    "HD",
    "MCD",
    "NKE",
    "SBUX",
    "TGT",
    "BKNG",
    "CMG",
    "LOW",
    "MAR",
    "EBAY",
    # ----- Industrials (8) -----
    "CAT",
    "BA",
    "HON",
    "UPS",
    "FDX",
    "LMT",
    "GE",
    "DE",
    # ----- Consumer Staples (8) -----
    "PG",
    "KO",
    "PEP",
    "COST",
    "WMT",
    "MDLZ",
    "CL",
    "KHC",
    # ----- Communication Services (9) -----
    "NFLX",
    "DIS",
    "T",
    "VZ",
    "TMUS",
    "CMCSA",
    "CHTR",
    "EA",
    "TTWO",
]

with st.sidebar:
    model_version = st.text_input(
        "Model version", value="v507"
    )  # matches your checkpoints naming
    checkpoints_dir = st.text_input("Checkpoints directory", value="models")
    default_tickers = ", ".join(TICKERS)
    tickers_str = st.text_area(
        "Tickers (comma-separated)", value=default_tickers, height=90
    )
    start_date = st.date_input("Start fetch (for features)", value=date(2005, 1, 1))
    run_btn = st.button("🚀 Run Forecasts", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("📈 VolSense — Sector & Ticker Volatility Insights")
st.markdown(
    "Real-time cross-sectional analytics and sector-aware signals. "
    "Use the sidebar to select model/tickers and generate insights."
)

# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────

if run_btn or st.session_state.forecast_data is not None:
    try:
        # 1️⃣ Load cached or run new forecast
        if not run_btn:
            fcast = st.session_state.forecast_data["fcast"]
            preds = st.session_state.forecast_data["preds"]
            analytics = st.session_state.forecast_data["analytics"]
            ae_summary = st.session_state.forecast_data["ae_summary"]
            se = st.session_state.forecast_data["se"]
            sig_df = st.session_state.forecast_data["sig_df"]
        else:
            tickers = _parse_tickers(tickers_str)
            if not tickers:
                st.warning("Please provide at least one ticker.")
                st.stop()

            with st.spinner("Running VolSense forecasts and computing signals…"):
                fcast, preds, analytics, ae_summary, se, sig_df = run_volsense(
                    model_version=model_version,
                    checkpoints_dir=checkpoints_dir,
                    start=str(start_date),
                    tickers=tickers,
                )
            st.session_state.forecast_data = {
                "fcast": fcast,
                "preds": preds,
                "analytics": analytics,
                "ae_summary": ae_summary,
                "se": se,
                "sig_df": sig_df,
            }

        st.success("✅ Forecasts & signals ready!")

        # 2️⃣ Remember which tab is active between reruns
        tabs = st.session_state.tabs_list
        selected_tab = st.session_state.active_tab
        tab_objs = st.tabs(tabs)
        selected_index = tabs.index(selected_tab) if selected_tab in tabs else 0

        # 3️⃣ Map tabs to content
        tab_overview, tab_tickers, tab_sector, tab_signals = tab_objs

        # 4️⃣ After rendering, update active tab
        st.session_state.active_tab = tabs[selected_index]

        # 5️⃣ Tab contents
        with tab_overview:
            st.subheader("Snapshot (Realized vs Forecast)")
            ordered_cols = (
                ["ticker", "realized_vol"]
                + [c for c in preds.columns if c.startswith("pred_vol_")]
                + [c for c in ["vol_diff", "vol_direction"] if c in preds.columns]
            )
            st.dataframe(
                _safe_style_format(preds[ordered_cols], _pretty_number),
                use_container_width=True,
                height=380,
            )
            # Export CSV button for the overview table
            export_csv_button(
                preds[ordered_cols],
                f"volsense_forecast_{date.today()}.csv",
                "📥 Export Predictions",
            )
            # export_csv_button(preds, f"volsense_forecast_{date.today()}.csv", "Export Predictions")

        with tab_tickers:
            st.subheader("Per-Ticker Forecast vs Realized")
            horizons = _detect_horizons(preds)
            selected_ticker = st.selectbox(
                "Ticker", preds["ticker"].unique(), key="ticker_ta"
            )
            chosen_horizon = st.selectbox("Horizon", horizons, key="horizon_ta")

            fig = fcast.plot(selected_ticker, show=False)
            st.pyplot(fig, use_container_width=True)
            st.info(analytics.describe(selected_ticker, f"pred_vol_{chosen_horizon}"))

        # ──────────────────────────────────────────────────────────────────────
        # SECTOR VIEW
        # ──────────────────────────────────────────────────────────────────────
        with tab_sector:
            st.subheader("Sector Heatmap & Leaders")

            # Controls
            all_horizons = sorted(sig_df["horizon"].unique().tolist())
            h_sel = st.select_slider(
                "Horizon", options=all_horizons, value=all_horizons[0]
            )
            sector_map = get_sector_map(model_version)

            # Build sector pivot for heatmap (sector z per horizon)
            dsub = sig_df.copy()
            # latest date snapshot (SignalEngine standardizes date)   :contentReference[oaicite:8]{index=8}
            latest_date = dsub["date"].max()
            dsub = dsub[dsub["date"] == latest_date]

            pivot = (
                dsub.pivot_table(
                    index="sector", columns="horizon", values="sector_z", aggfunc="mean"
                )
                .fillna(0.0)
                .sort_index()
            )

            # Plot with st.pyplot (simple heatmap using matplotlib)
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig_hm, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(
                pivot,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".2f",
                cbar_kws={"label": "Sector Z-score"},
                ax=ax,
            )
            ax.set_title(f"Sector Volatility Heatmap  —  {latest_date.date()}")
            st.pyplot(fig_hm, use_container_width=True)

            # Top sectors by mean z at selected horizon
            top_n = st.slider("Top sectors to display", 3, 12, 8)
            sec_slice = (
                dsub[dsub["horizon"] == h_sel]
                .groupby("sector")["sector_z"]
                .mean()
                .sort_values(ascending=False)
            )
            top = sec_slice.head(top_n)

            # Horizontal bar with sector colors
            colors = [get_color(s) for s in top.index]
            fig_bar, axb = plt.subplots(figsize=(6, 0.45 * len(top) + 1.5))
            axb.barh(top.index, top.values, color=colors)
            axb.invert_yaxis()
            axb.set_xlabel("Mean Sector Z-score")
            axb.set_title(f"Top {top_n} Sectors (H={h_sel}d)")
            st.pyplot(fig_bar, use_container_width=True)

        # ──────────────────────────────────────────────────────────────────────
        # SIGNAL TABLE
        # ──────────────────────────────────────────────────────────────────────
        with tab_signals:
            st.subheader("Cross-Sectional Signal Table")
            # ───────────────────────────────────────────────
            # Filters
            # ───────────────────────────────────────────────
            sectors = ["All"] + sorted(sig_df["sector"].dropna().unique().tolist())
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
            with c1:
                sector_filter = st.selectbox(
                    "Sector", sectors, index=0, key="sig_sector"
                )
            with c2:
                horizon_filter = st.selectbox(
                    "Horizon",
                    sorted(sig_df["horizon"].unique().tolist()),
                    key="sig_horizon",
                )
            with c3:
                regime = st.selectbox(
                    "Regime",
                    ["All", "calm", "normal", "spike"],
                    index=0,
                    key="sig_regime",
                )
            with c4:
                position_filter = st.selectbox(
                    "Position",
                    ["All", "long", "neutral", "short"],
                    index=0,
                    key="sig_position",
                )
            with c5:
                sort_by = st.selectbox(
                    "Sort by",
                    [
                        "vol_zscore",
                        "vol_spread",
                        "term_spread_10v5",
                        "rank_universe",
                        "rank_sector",
                        "ticker",
                    ],
                    key="sig_sort",
                )

            # ───────────────────────────────────────────────
            # Safe filtering logic
            # ───────────────────────────────────────────────
            table = sig_df.copy()
            if "horizon" in table.columns:
                table["horizon"] = table["horizon"].astype(str)
            horizon_filter = str(horizon_filter)
            if "sector" in table.columns:
                table["sector"] = table["sector"].astype(str)

            if sector_filter != "All" and "sector" in table.columns:
                table = table[table["sector"] == sector_filter]
            if horizon_filter != "All" and "horizon" in table.columns:
                table = table[table["horizon"] == horizon_filter]
            if regime != "All" and "regime_flag" in table.columns:
                table = table[table["regime_flag"] == regime]
            if position_filter != "All" and "position" in table.columns:
                table = table[table["position"] == position_filter]

            # ───────────────────────────────────────────────
            # Subset and display
            # ───────────────────────────────────────────────
            cols = [
                "ticker",
                "sector",
                "horizon",
                "forecast_vol",
                "today_vol",
                "vol_spread",
                "term_spread_10v5",
                "vol_zscore",
                "rank_universe",
                "rank_sector",
                "position",
                "regime_flag",
            ]
            cols = [c for c in cols if c in table.columns]
            table = table[cols].sort_values(sort_by, ascending=False)

            if table.empty:
                st.warning("⚠️ No signals available for the selected filters.")
            else:
                st.caption(
                    f"📊 Positions: {table['position'].value_counts().to_dict()}"
                )
                st.dataframe(
                    table.style.format(
                        {
                            "forecast_vol": "{:.4f}".format,
                            "today_vol": "{:.4f}".format,
                            "vol_spread": "{:+.2%}".format,
                            "term_spread_10v5": "{:+.2%}".format,
                            "vol_zscore": "{:+.2f}".format,
                            "rank_universe": lambda x: f"{x:.2f}",
                            "rank_sector": lambda x: f"{x:.2f}",
                        }
                    ),
                    use_container_width=True,
                    height=520,
                )

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info(
        "Set model, tickers and press **Run Forecasts** to generate analytics & signals."
    )
