from __future__ import annotations

import pandas as pd

from utils import ROOT, ensure_dirs, load_params


def main() -> None:
    ensure_dirs()
    params = load_params()
    date_col = params["data"]["date_col"]
    horizon = int(params["gold"]["target_horizon_days"])

    silver_path = ROOT / "data/silver/silver.parquet"
    gold_path = ROOT / "data/gold/gold.parquet"

    df = pd.read_parquet(silver_path).copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.sort_values(date_col).reset_index(drop=True)

    # Define supervised target: next-day meantemp
    df[f"target_meantemp_t_plus_{horizon}"] = df["meantemp"].shift(-horizon)

    # Select features for forecasting (keep engineered lags + rolling + calendar)
    feature_cols = [
        date_col,
        "meantemp",
        "humidity",
        "wind_speed",
        "meanpressure",
        "month",
        "dayofyear",
    ] + [c for c in df.columns if c.startswith("lag") or c.startswith("roll")]

    target_col = f"target_meantemp_t_plus_{horizon}"
    cols = feature_cols + [target_col]

    gold = df[cols].dropna(subset=[target_col]).reset_index(drop=True)
    gold.to_parquet(gold_path, index=False)

    print(f"Wrote gold: {gold_path} rows={len(gold)} cols={len(gold.columns)}")


if __name__ == "__main__":
    main()
