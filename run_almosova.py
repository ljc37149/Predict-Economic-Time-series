"""
Reproduction driver: Almosova & Andresen (2023), Journal of Forecasting,
DOI 10.1002/for.2901 — US CPI inflation, expanding-window forecasts from 1990.

Example:
  python run_almosova.py --quick
  python run_almosova.py --data nsa --max-origins 24 --epochs 300 --models RW AR NN LSTM SARIMA
  python run_almosova.py --epochs 500 --progress --progress-every 5
  python run_almosova.py --state rolling.pkl --progress
  python run_almosova.py --state rolling.pkl --resume --progress
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pandas as pd

from data_fred import load_inflation
from rolling_forecast import RollConfig, msfe_matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=("sa", "nsa"), default="nsa", help="FRED CPI series (paper Table 1 vs 2)")
    ap.add_argument("--test-start", default="1990-01-01", help="First forecast origin month")
    ap.add_argument("--max-origins", type=int, default=None, help="Cap rolling origins (default: all test months)")
    ap.add_argument("--epochs", type=int, default=500, help="Adam epochs for AR/NN/LSTM (paper: often 500–2000)")
    ap.add_argument("--hidden", type=int, default=50, help="Hidden units for NN/LSTM (paper Table 5 LSTM: 50)")
    ap.add_argument("--lr", type=float, default=0.001, help="Adam learning rate")
    ap.add_argument("--max-lag", type=int, default=24, help="Max lags; LSTM uses this fixed if --no-bic")
    ap.add_argument("--bic", action="store_true", help="Use BIC lag for AR/NN (LSTM still uses --max-lag)")
    ap.add_argument("--quick", action="store_true", help="3 origins, 80 epochs, subset of models")
    ap.add_argument(
        "--models",
        nargs="*",
        default=["RW", "AR", "NN", "LSTM", "SARIMA", "MS"],
        help="Models to run",
    )
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Print one line after each rolling origin (helps when stdout is redirected to a log)",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
        metavar="N",
        help="With --progress, emit a line every N origins (still logs first & last); default 1 = all origins",
    )
    ap.add_argument(
        "--state",
        default=None,
        metavar="PATH",
        help="Pickle checkpoint path: save partial sums for resume after each origin (or see --save-every)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="With --state, load checkpoint if the file exists (metadata must match this run)",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=1,
        metavar="N",
        help="With --state, write checkpoint every N completed origins (default: 1)",
    )
    args = ap.parse_args()

    if args.resume and not args.state:
        ap.error("--resume requires --state PATH")

    if args.quick:
        args.max_origins = 3
        args.epochs = 80
        args.models = ["RW", "AR", "LSTM"]

    y = load_inflation(kind=args.data)
    cfg = RollConfig(
        max_lag=args.max_lag,
        use_bic=args.bic,
        hidden=args.hidden,
        lr=args.lr,
        epochs=args.epochs,
    )

    test_start = pd.Timestamp(args.test_start)
    df = msfe_matrix(
        y,
        test_start=test_start,
        cfg=cfg,
        models=args.models,
        max_test_origins=args.max_origins,
        progress=args.progress,
        progress_every=max(1, args.progress_every),
        state_path=args.state,
        resume=args.resume,
        save_every=max(1, args.save_every),
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
