#!/usr/bin/env python3
"""
VolSense CLI — Train Model Utility
----------------------------------
Command-line interface for training VolSense models
(LSTM, Global LSTM, or GARCH family) directly from datasets.

Example:
    volsense-train --method global_lstm --data data/volsense_dataset.csv \
        --epochs 15 --window 40 --horizons 1 5 10 --val_start 2023-01-01 \
        --save_dir models --device cuda
"""

import argparse
import sys
import os
import pandas as pd
from datetime import datetime
from volsense_core.forecaster_core import VolSenseForecaster


# ============================================================
# 🧩 Argument Parser
# ============================================================
def parse_args():
    """
    Parse CLI arguments for training VolSense models.

    Defines flags for data path, model method, window, horizons, extra features,
    validation start, epochs, learning rate, dropout, device, save directory, and version tag.

    :return: Parsed arguments namespace with attributes matching CLI flags.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Train VolSense volatility forecasting models."
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV dataset with columns ['date','ticker','return','realized_vol', ...]",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="global_lstm",
        choices=["lstm", "global_lstm", "garch", "egarch", "gjr"],
        help="Model architecture to train.",
    )
    parser.add_argument("--window", type=int, default=30, help="Lookback window size.")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Forecast horizons to train for.",
    )
    parser.add_argument(
        "--extra_features",
        nargs="+",
        default=None,
        help="Optional list of extra feature column names (e.g. vol_3d vol_10d vol_ratio).",
    )
    parser.add_argument(
        "--val_start",
        type=str,
        default="2023-01-01",
        help="Validation start date (YYYY-MM-DD).",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=datetime.now().strftime("%Y%m%d"),
        help="Optional model version tag (default: date).",
    )

    return parser.parse_args()


# ============================================================
# 🚀 Main Entry
# ============================================================
def main():
    """
    Main entry point for the volsense-train CLI.

    Loads a dataset CSV, initializes a VolSenseForecaster according to CLI args,
    trains the selected model (LSTM, Global LSTM, or GARCH family), and saves
    a checkpoint to the specified directory.

    :raises SystemExit: If the dataset path is missing/invalid or required columns are absent.
    :return: None
    :rtype: None
    """
    args = parse_args()
    print(
        f"\n🚀 Starting VolSense training (method={args.method}, device={args.device})"
    )

    # --- Load dataset ---
    if not os.path.exists(args.data):
        print(f"❌ Dataset not found: {args.data}")
        sys.exit(1)

    df = pd.read_csv(args.data)
    if "date" not in df.columns or "ticker" not in df.columns:
        print("❌ Dataset must include 'date' and 'ticker' columns.")
        sys.exit(1)

    # --- Initialize Forecaster ---
    forecaster = VolSenseForecaster(
        method=args.method,
        device=args.device,
        window=args.window,
        horizons=args.horizons,
        val_start=args.val_start,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        extra_features=args.extra_features,
    )

    # --- Train ---
    model = forecaster.fit(df)
    print(f"✅ Training complete for method={args.method}")

    # --- Save checkpoint ---
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = f"volsense_{args.method}_{args.version}"
    save_path = os.path.join(args.save_dir, ckpt_name)

    try:
        if args.method in ["lstm", "global_lstm"]:
            # torch save or bundle pickle handled in forecaster.fit internally
            import torch

            torch.save(model.model.state_dict(), save_path + ".pth")
        else:
            # GARCH models saved as pickle
            import pickle

            with open(save_path + ".pkl", "wb") as f:
                pickle.dump(model.model, f)
        print(f"💾 Model saved to {save_path}")
    except Exception as e:
        print(f"⚠️ Save failed: {e}")

    print("\n🎯 All done! Ready for inference via volsense-forecast CLI.\n")


if __name__ == "__main__":
    main()
