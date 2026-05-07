"""
Plot evaluation outputs: MSFE table (from a log tail or CSV) and CPI inflation series.

Examples:
  python plot_results.py
  python plot_results.py --log train_out.log --out-dir figures
  python plot_results.py --msfe-csv results_msfe.csv --out-dir figures
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_fred import load_inflation


def _extract_msfe_block_from_log(text: str) -> str:
    lines = text.splitlines()
    header_i: int | None = None
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if s.startswith("model") and "h1" in s and "h12" in s:
            header_i = i
            break
    if header_i is None:
        raise ValueError("no MSFE table header (line starting with 'model' plus h1..h12) found in log")

    block: list[str] = []
    for j in range(header_i, len(lines)):
        row = lines[j].rstrip()
        if not row.strip():
            break
        if row.strip().startswith("[progress]"):
            break
        block.append(row)
    if len(block) < 2:
        raise ValueError("MSFE table in log appears empty")
    return "\n".join(block)


def load_msfe_table(*, log_path: Path | None, csv_path: Path | None) -> pd.DataFrame:
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    elif log_path is not None:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
        block = _extract_msfe_block_from_log(raw)
        df = pd.read_csv(io.StringIO(block), sep=r"\s+")
    else:
        raise ValueError("pass --log or --msfe-csv")

    df.columns = [str(c).strip() for c in df.columns]
    if "model" not in df.columns:
        raise ValueError("MSFE table must have a 'model' column")

    hcols = [c for c in df.columns if re.match(r"^h\d+$", c)]
    for c in hcols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_msfe_curves(df: pd.DataFrame, path: Path) -> None:
    hcols = [c for c in df.columns if re.match(r"^h\d+$", c)]
    if not hcols:
        raise ValueError("no h1..h12 columns found")
    x = np.array([int(c[1:]) for c in hcols])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, row in df.iterrows():
        m = str(row["model"]).strip()
        y = row[hcols].to_numpy(dtype=float)
        if np.all(np.isnan(y)):
            ax.plot(x, y, linestyle=":", linewidth=1.5, label=f"{m} (all NaN)", alpha=0.7)
            continue
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.5, label=m)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("MSFE")
    ax.set_title("Mean squared forecast error by horizon")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_msfe_heatmap(df: pd.DataFrame, path: Path) -> None:
    hcols = [c for c in df.columns if re.match(r"^h\d+$", c)]
    if not hcols:
        raise ValueError("no h1..h12 columns found")
    mat = df.set_index("model")[hcols].to_numpy(dtype=float)
    mat = np.ma.masked_invalid(mat)
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(df))))
    vmax = np.nanmax(mat)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = None
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"].astype(str).tolist())
    ax.set_xticks(range(len(hcols)))
    ax.set_xticklabels(hcols)
    ax.set_xlabel("Horizon")
    ax.set_title("MSFE heatmap (NaN shown as blank)")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="MSFE")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_msfe_bars_mean(df: pd.DataFrame, path: Path) -> None:
    hcols = [c for c in df.columns if re.match(r"^h\d+$", c)]
    means: list[float] = []
    labels: list[str] = []
    for _, row in df.iterrows():
        v = row[hcols].to_numpy(dtype=float)
        labels.append(str(row["model"]).strip())
        means.append(float(np.nanmean(v)) if np.any(np.isfinite(v)) else float("nan"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    colors = ["#4c72b0" if np.isfinite(m) else "#c44e52" for m in means]
    ax.bar(x, [m if np.isfinite(m) else 0.0 for m in means], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean MSFE (h1–h12)")
    ax.set_title("Average MSFE across horizons (ignores NaNs in mean)")
    for i, m in enumerate(means):
        if not np.isfinite(m):
            ax.text(i, 0.02 * (ax.get_ylim()[1] or 1), "NaN", ha="center", fontsize=8, rotation=0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_inflation_series(kind: str, test_start: str, path: Path) -> None:
    y = load_inflation(kind=kind)
    ts = pd.Timestamp(test_start)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(y.index, y.values, color="black", linewidth=0.9, label=f"CPI inflation ({kind.upper()})")
    if ts in y.index or ts < y.index.max():
        ax.axvline(ts, color="#c44e52", linestyle="--", linewidth=1.2, label=f"Test start {test_start[:10]}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly % change")
    ax.set_title("US CPI inflation (FRED)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot MSFE tables and inflation data")
    ap.add_argument("--log", type=Path, default=Path("train_out.log"), help="Log file containing final MSFE table")
    ap.add_argument("--msfe-csv", type=Path, default=None, help="Optional CSV (model,h1,...) instead of --log")
    ap.add_argument("--out-dir", type=Path, default=Path("figures"), help="Directory for PNG figures")
    ap.add_argument("--data", choices=("sa", "nsa"), default="nsa", help="FRED series for inflation plot")
    ap.add_argument("--test-start", default="1990-01-01", help="Vertical line on inflation plot")
    ap.add_argument("--no-inflation", action="store_true", help="Skip CPI time series figure")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_msfe_table(log_path=None if args.msfe_csv else args.log, csv_path=args.msfe_csv)
    plot_msfe_curves(df, out / "msfe_by_horizon.png")
    plot_msfe_heatmap(df, out / "msfe_heatmap.png")
    plot_msfe_bars_mean(df, out / "msfe_mean_bar.png")

    if not args.no_inflation:
        plot_inflation_series(args.data, args.test_start, out / "inflation_series.png")

    print(f"Wrote figures to {out.resolve()}")


if __name__ == "__main__":
    main()
