"""
Build a Table-5 style ranking for multivariate LSTM settings.

Example:
  python lstm_multivar_table.py --use-macro --use-month-features --max-origins 36
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_fred import load_inflation, load_macro_features, load_month_features
from rolling_forecast import RollConfig, msfe_matrix


def _parse_list_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_list_float(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_horizons(s: str) -> list[int]:
    hs = [int(x.strip()) for x in s.split(",") if x.strip()]
    hs = [h for h in hs if 1 <= h <= 12]
    if not hs:
        raise ValueError("--rank-horizons must contain values in 1..12")
    return hs


def _build_exog(y: pd.Series, use_macro: bool, use_month_features: bool) -> pd.DataFrame | None:
    parts: list[pd.DataFrame] = []
    if use_macro:
        macro = load_macro_features(start=str(y.index.min().date()), end=str(y.index.max().date()))
        parts.append(macro.reindex(y.index).ffill().bfill())
    if use_month_features:
        parts.append(load_month_features(y.index))
    return pd.concat(parts, axis=1) if parts else None


def _plot_table(df: pd.DataFrame, out_png: Path) -> None:
    show_cols = ["rank", "test_error", "h2", "h3", "h6", "h12", "n", "infc", "p", "Lag", "LR", "Epochs"]
    disp = df[show_cols].copy()
    for c in ["test_error", "h2", "h3", "h6", "h12", "LR"]:
        disp[c] = disp[c].map(lambda x: f"{x:.3f}")
    for c in ["rank", "n", "p", "Lag", "Epochs"]:
        disp[c] = disp[c].astype(int).astype(str)

    fig_h = max(3.0, 0.48 * (len(disp) + 2))
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.axis("off")
    ax.set_title("Top multivariate LSTM settings", fontsize=13, pad=12)
    tbl = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create Table-5 style ranking for multivariate LSTM")
    ap.add_argument("--data", choices=("sa", "nsa"), default="nsa")
    ap.add_argument("--test-start", default="1990-01-01")
    ap.add_argument("--max-origins", type=int, default=None, help="Use a smaller value for a faster sweep")
    ap.add_argument("--max-lag", type=int, default=24)
    ap.add_argument("--hidden-list", default="20,50,100")
    ap.add_argument("--lr-list", default="0.001,0.01,0.05")
    ap.add_argument("--epochs-list", default="500,1000,1500")
    ap.add_argument("--rank-horizons", default="2,3,6,12", help="Horizons used to compute test_error")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-macro", action="store_true")
    ap.add_argument("--use-month-features", action="store_true")
    ap.add_argument("--out-csv", type=Path, default=Path("lstm_multivar_ranking.csv"))
    ap.add_argument("--out-png", type=Path, default=Path("lstm_multivar_table.png"))
    args = ap.parse_args()

    hidden_list = _parse_list_int(args.hidden_list)
    lr_list = _parse_list_float(args.lr_list)
    epochs_list = _parse_list_int(args.epochs_list)
    rank_h = _parse_horizons(args.rank_horizons)
    rank_cols = [f"h{h}" for h in rank_h]

    y = load_inflation(kind=args.data)
    exog = _build_exog(y, args.use_macro, args.use_month_features)

    rows: list[dict[str, float | int | str]] = []
    combos = list(itertools.product(hidden_list, lr_list, epochs_list))
    for i, (hidden, lr, epochs) in enumerate(combos, start=1):
        cfg = RollConfig(
            max_lag=args.max_lag,
            use_bic=False,
            hidden=hidden,
            lr=lr,
            epochs=epochs,
            seed=args.seed,
        )
        df = msfe_matrix(
            y,
            exog=exog,
            test_start=pd.Timestamp(args.test_start),
            cfg=cfg,
            models=["LSTM"],
            max_test_origins=args.max_origins,
            progress=False,
        )
        r = df.iloc[0].to_dict()
        test_error = float(np.nanmean([float(r[c]) for c in rank_cols]))
        row: dict[str, float | int | str] = {
            "test_error": test_error,
            "h2": float(r["h2"]),
            "h3": float(r["h3"]),
            "h6": float(r["h6"]),
            "h12": float(r["h12"]),
            "n": hidden,
            "infc": "None",
            "p": args.max_lag,
            "Lag": args.max_lag,
            "LR": lr,
            "Epochs": epochs,
        }
        rows.append(row)
        print(f"[{i}/{len(combos)}] done n={hidden} lr={lr} epochs={epochs} test_error={test_error:.4f}", flush=True)

    out_df = pd.DataFrame(rows).sort_values("test_error", ascending=True).reset_index(drop=True)
    out_df["rank"] = np.arange(1, len(out_df) + 1)
    out_df = out_df[["rank", "test_error", "h2", "h3", "h6", "h12", "n", "infc", "p", "Lag", "LR", "Epochs"]]
    top_df = out_df.head(max(1, args.top_k)).copy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    top_df.to_csv(args.out_csv, index=False)
    _plot_table(top_df, args.out_png)
    print(f"Wrote ranking CSV: {args.out_csv.resolve()}")
    print(f"Wrote table PNG: {args.out_png.resolve()}")


if __name__ == "__main__":
    main()

