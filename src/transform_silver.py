from __future__ import annotations

import numpy as np
import pandas as pd

from utils import ROOT, ensure_dirs, load_params


def main() -> None:
    ensure_dirs()
    params = load_params()
    date_col = params["data"]["date_col"]
    w = int(params["silver"]["rolling_window_days"])
    lags = list(params["silver"]["lags"])

    bronze_path = ROOT / "data/bronze/bronze.parquet"
    silver_path = ROOT / "data/silver/silver.parquet"

    df = pd.read_parquet(bronze_path).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    # Keep only relevant raw columns + batch metadata if present
    keep = [date_col, "meantemp", "humidity", "wind_speed", "meanpressure"]
    meta = [c for c in ["__batch_name"] if c in df.columns]
    df = df[keep + meta]

    # Drop rows with invalid date
    df = df.dropna(subset=[date_col])

    # Deduplicate by date (silver requires no duplicated days)
    # Strategy: aggregate duplicates by mean (stable and explainable)
    agg_cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]
    df = df.groupby(date_col, as_index=False)[agg_cols].mean()

    # Reindex to continuous daily frequency (makes missing days explicit)
    df = df.sort_values(date_col).reset_index(drop=True)
    full_idx = pd.date_range(df[date_col].min(), df[date_col].max(), freq="D")
    df = df.set_index(date_col).reindex(full_idx).rename_axis(date_col).reset_index()

    # Treat out-of-range meanpressure as corrupted -> set to NaN (then impute)
    df.loc[(df["meanpressure"] < 800) | (df["meanpressure"] > 1200), "meanpressure"] = np.nan

    # Imputation (explain in report as “time-series consistent”)
    # - forward fill then back fill to cover leading gaps
    for c in agg_cols:
        df[c] = df[c].astype("float64")
        df[c] = df[c].ffill().bfill()

    # Clamp humidity to [0, 100] (physical bounds)
    df["humidity"] = df["humidity"].clip(0, 100)

    # Basic calendar features (seasonality)
    df["month"] = df[date_col].dt.month.astype(int)
    df["dayofyear"] = df[date_col].dt.dayofyear.astype(int)

    # Rolling mean + lag features (forecasting-friendly)
    df = df.sort_values(date_col).reset_index(drop=True)
    df[f"roll{w}_meantemp"] = df["meantemp"].rolling(window=w, min_periods=1).mean()

    for lag in lags:
        df[f"lag{lag}_meantemp"] = df["meantemp"].shift(lag)
        df[f"lag{lag}_humidity"] = df["humidity"].shift(lag)

    # After feature creation, drop rows where lags are undefined (optional but typical)
    # Keep it simple: drop rows with any NA produced by shifting
    lag_cols = [f"lag{lag}_meantemp" for lag in lags] + [f"lag{lag}_humidity" for lag in lags]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    df.to_parquet(silver_path, index=False)
    print(f"Wrote silver: {silver_path} rows={len(df)}")


if __name__ == "__main__":
    main()
