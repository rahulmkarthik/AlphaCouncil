# **VolSense: Explainable Volatility Forecasting & Signal Intelligence**

**VolSense** is an end-to-end, explainable volatility forecasting and signal generation framework — blending **econometric foundations** (GARCH, EGARCH, GJR-GARCH) with **deep neural architectures** (LSTM, attention, residual heads) to deliver multi-horizon, interpretable volatility insights across global markets.

---

## Overview

VolSense answers a central quant research question:

> _Can deep learning architectures outperform classical GARCH-style baselines **without losing interpretability** for traders and risk managers?_

To explore this, VolSense introduces a **two-tiered architecture**:

1. **`volsense_core`** — the research and training pipeline for feature engineering, multi-model learning, evaluation, and explainability.
2. **`volsense_inference`** — the deployment layer for real-time forecasting, analytics, and signal generation, complete with an interactive Streamlit dashboard.

---

## Core Components

### **1️⃣ VolSense_Core — Research & Model Training**

A modular experimentation framework supporting both classical and neural volatility models:

| Model | Description |
|-------|-------------|
| **ARCHFamily (GARCH, EGARCH, GJR-GARCH)** | Econometric baselines for regime calibration and interpretability. |
| **BaseLSTM** | Single-ticker LSTM with residual & attention heads, designed for temporal precision. |
| **GlobalVolForecaster** | Multi-ticker shared LSTM with ticker embeddings, horizon-specific heads, and dropout regularization. |

**Highlights:**
- Configurable `TrainConfig` dataclass for all hyperparameters.  
- Robust checkpointing and bundle serialization.  
- Multi-horizon support (`1d`, `5d`, `10d`).  
- Regime-wise validation on calm (2013-14), spike (2020-21), and forward (2023-present) periods.  
- Integrated evaluation layer (RMSE, MAE, correlation, Durbin–Watson, feature importance).

Two flagship models — **v109** (109 tickers) and **v507** (507 tickers) — are trained and released as production-ready baselines.

---

### **2️⃣ VolSense_Inference — Operational Forecasting & Signals**

A portable, production-grade inference layer enabling researchers and traders to run forecasts, analyze analytics, and extract signals on demand.

**Modules:**
- `forecast_engine.py` — unified inference entrypoint.  
- `model_loader.py` — loads serialized `.pth` / `.pkl` checkpoints.  
- `predictor.py` — manages scaling, batching, and forecast generation.  
- `analytics.py` — computes cross-ticker analytics, realized vs predicted correlation, and feature diagnostics.  
- `signal_engine.py` — transforms forecasts into actionable long/short/neutral signals with sector and regime awareness.  
- `dashboard.py` — a four-tab Streamlit UI for traders and researchers.

**Dashboard Tabs:**
| Tab | Function |
|------|-----------|
| **Overview** | Realized vs predicted volatility snapshot. |
| **Ticker Analytics** | Forecast curves, realized overlays, error distributions. |
| **Sector View** | Heatmaps and top/bottom sector rankings by Z-score. |
| **Signal Table** | Filterable cross-sectional signal matrix by sector, regime, horizon, or position. |

---

## Data & Feature Engineering

- Automated OHLCV ingestion via **Yahoo Finance** and `fetch.py`.  
- Unified feature generation with `build_features()` ensuring identical transformations for training and inference.  
- Core features include:
  - `vol_3d`, `vol_20d`, `vol_60d`, `vol_ratio`, `vol_vol`, `vol_chg`
  - `return`, `ret_sq`, `abs_return`, `ewma_return_5d`
  - `beta_20d`, `market_stress`, `skew_5d`
  - Calendar features: `day_of_week`, `month_sin`, `month_cos`

---

## Explainability

VolSense incorporates a multi-layered **explainability suite** purpose-built for sequence-based volatility forecasting, ensuring transparency across both features and time:

- **Attention heatmaps** — visualize where the model “looks” across the past 40-day window, revealing temporal focus patterns for each forecast horizon.  
- **Feature sensitivity analysis** — quantifies how changes in key drivers (e.g., `abs_return`, `vol_20d`, `beta_20d`) influence log-volatility forecasts at different horizons (`1d`, `5d`, `10d`).  
- **Temporal sensitivity curves** — highlight how recency affects predictive power, showing that recent shocks dominate short-term forecasts while older volatility events fade naturally.  
- **Cross-horizon explainability** — compare how feature importance and attention dynamics evolve from reactive (1-day) to smoothed (10-day) predictions.  
- **Human-aligned interpretation** — findings align with financial intuition: volatility clusters, decays over time, and remains influenced by recent market shocks.  

Together, these tools make VolSense not only **accurate**, but also **interpretable** — bridging quantitative modeling with human reasoning about market dynamics.

## Evaluation

Comprehensive evaluation framework (`evaluation.py`, `metrics.py`) featuring:
- RMSE, MAE, correlation, and horizon-specific error tracking.  
- Volatility clustering and Durbin–Watson diagnostics.  
- Residual distribution and QQ plots for calibration.  
- Best/worst ticker identification across horizons.  
- Feature importance and prediction–truth correlation analytics.

---

## Deployment & CLI

VolSense provides a lightweight CLI for both training and forecasting:

```bash
# Train models
volsense-train --config configs/train_v109.yaml

# Run forecasts
volsense-forecast --model v109 --tickers AAPL,MSFT,SPY
```

## Streamlit Dashboard Preview

Launches an intuitive dashboard providing:
- Real-time forecasts and realized vol tracking.
- Sector heatmaps and Z-score distributions.
- Cross-sectional signal visualization with position, regime, and horizon filters.
- One-click CSV export for further analysis.
 
## Documentation

Full technical documentation is generated via Sphinx and Napoleon, including:

- API references for every core/inference module.
- Notebook-integrated tutorials (`01_data_pipeline` → `06_explainability`).
- Architectural diagrams and configuration examples.

## Repository Structure

```

VolSense/
│
├── volsense_core/
│   ├── cli/                  # Training CLI
│   ├── data/                 # Data fetch + feature engineering
│   ├── evaluation/           # Metrics + evaluation
│   ├── explainability/       # SHAP & interpretability notebooks
│   ├── models/               # LSTM, GARCH, GlobalVolForecaster
│   ├── utils/                # Checkpoint utils, scalers
│   └── forecaster_core.py    # Core trainer entrypoint
│
├── volsense_inference/
│   ├── cli/                  # Forecast CLI
│   ├── notebooks/            # Dashboard & explainability demos
│   ├── analytics.py
│   ├── forecast_engine.py
│   ├── model_loader.py
│   ├── predictor.py
│   ├── signal_engine.py
│   ├── dashboard.py
│   └── sector_mapping.py
│
├── models/                   # Trained model bundles (v109, v509)
│
├── notebooks/                # Jupyter notebooks (01–06)
│   ├── 01_data_pipeline.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_inference.ipynb
│   ├── 04_signals_and_dashboard.ipynb
│   ├── 05_v109_training.ipynb
│   └── 06_v109_explainability.ipynb
│
├── docs/                     # Sphinx documentation
├── pyproject.toml
├── README.md
├── requirements.txt
└── .gitignore

```

## Credits & License

Developed by Rahul Karthik as part of the _VolSense Research Framework_.
Inspired by hybrid econometric-deep learning models in quantitative volatility research.

Licensed under the MIT License.
Pull requests, collaborations, and discussions are welcome!

## Connect

- 📫 **LinkedIn:** [Rahul Karthik](https://www.linkedin.com/rahulmkarthik/)
- 💻 **GitHub:** [github.com/rahulmkarthik](https://github.com/rahulmkarthik)  
- 🧠 **Research Interests:** Quantitative Finance · Quantitative Systems Design · Volatility Modeling · Explainable AI
- 🧰 **Tech Stack:** Python · Matplotlib · NumPy · PyTorch · SHAP · Pandas · Seaborn · Streamlit · Sphinx · yFinance
