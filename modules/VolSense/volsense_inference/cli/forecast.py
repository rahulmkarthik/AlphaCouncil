#!/usr/bin/env python3
"""
VolSense CLI — Forecast Interface
----------------------------------
Command-line tool for running volatility forecasts
using pretrained global VolSense models.

Example:
    volsense-forecast --tickers AAPL MSFT --model v3 --horizon 5 --plot
"""

import argparse
import sys
from volsense_inference.forecast_engine import Forecast


def parse_args():
    """
    Parse CLI arguments for VolSense forecasts.

    Defines flags for tickers, model version, checkpoints directory, display horizon,
    plotting, and optional CSV save path.

    :return: Parsed arguments namespace with keys: tickers, model, checkpoints_dir, horizon, plot, save.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Run volatility forecasts using pretrained VolSense models."
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="List of ticker symbols to forecast (e.g. AAPL MSFT TSLA).",
    )
    parser.add_argument(
        "--model",
        default="v5c",
        help="Model version identifier (matches .pth / .pkl / .json files).",
    )
    parser.add_argument(
        "--checkpoints_dir",
        default="models",
        help="Path to directory containing trained checkpoints.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Forecast horizon to display (1, 5, 10, etc.).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, generates forecast visualization(s).",
    )
    parser.add_argument(
        "--save", default=None, help="Optional path to save forecast results as CSV."
    )

    return parser.parse_args()


def main():
    """
    Run the volsense-forecast CLI.

    Initializes the Forecast engine, generates forecasts for the requested tickers,
    optionally saves results to CSV, and renders per-ticker plots.

    :raises SystemExit: If forecasting fails or no results are produced.
    :return: None
    :rtype: None
    """
    args = parse_args()

    print(f"\n🚀 Initializing VolSense Forecast (model={args.model})")
    fcast = Forecast(model_version=args.model, checkpoints_dir=args.checkpoints_dir)

    try:
        preds = fcast.run(args.tickers)
        if preds.empty:
            print("⚠️ No valid forecasts were generated.")
            sys.exit(1)

        # Display results
        print("\n📊 Forecast Results (realized first):")
        ordered_cols = (
            ["ticker", "realized_vol"]
            + [c for c in preds.columns if c.startswith("pred_vol_")]
            + [c for c in ["vol_diff", "vol_direction"] if c in preds.columns]
        )
        print(preds[ordered_cols].to_string(index=False))

        # Optional: Save to CSV
        if args.save:
            preds.to_csv(args.save, index=False)
            print(f"\n💾 Results saved to {args.save}")

        # Optional: Plot
        if args.plot:
            for tkr in args.tickers:
                fcast.plot(tkr)

    except Exception as e:
        print(f"\n❌ Forecast failed due to: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
