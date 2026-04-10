"""US CPI from FRED (Almosova & Andresen 2023, Journal of Forecasting)."""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import pandas as pd
import requests

# SA: CPI index (compute m/m %); NSA: FRED already reports monthly % change (paper Table 1)
FRED_SERIES = {
    "sa": "CPALTT01USM661S",
    "nsa": "CPALTT01USM657N",
}


def fetch_cpi(series_id: str, start: str = "1960-01-01", end: str = "2020-06-01") -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=[0])
    df = df.iloc[:, :2]
    df.columns = ["date", "cpi"]
    df = df.sort_values("date").set_index("date")
    df = df.loc[start:end]
    df = df[df["cpi"].astype(str) != "."]
    df["cpi"] = pd.to_numeric(df["cpi"], errors="coerce")
    df = df.dropna(subset=["cpi"])
    df = df[np.isfinite(df["cpi"])]
    return df["cpi"].astype(float)


def monthly_inflation_pct(cpi: pd.Series) -> pd.Series:
    """100 * (P_t / P_{t-1} - 1), first observation is dropped."""
    prev = cpi.shift(1)
    ok = (prev > 0) & np.isfinite(prev) & np.isfinite(cpi)
    pi = pd.Series(index=cpi.index, dtype=float)
    pi.loc[ok] = 100.0 * (cpi.loc[ok] / prev.loc[ok] - 1.0)
    return pi.dropna()


def load_inflation(
    kind: Literal["sa", "nsa"] = "sa",
    start: str = "1960-01-01",
    end: str = "2020-06-01",
) -> pd.Series:
    s = fetch_cpi(FRED_SERIES[kind], start=start, end=end)
    if kind == "sa":
        y = monthly_inflation_pct(s)
    else:
        # FRED "All Items Consumer Price Index for All Urban Consumers: All Items" monthly change series
        y = s.replace([np.inf, -np.inf], np.nan).dropna()
    y.name = f"inflation_{kind}"
    return y


def naive_seasonal_adjust(y: pd.Series) -> pd.Series:
    """Subtract historic monthly mean (calendar month)."""
    m = y.groupby(y.index.month).transform("mean")
    return y - m + y.mean()
