from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils import ROOT, load_params


def main() -> None:
    params = load_params()
    date_col = params["data"]["date_col"]

    raw_path = ROOT / "data/raw/DailyDelhiClimateTrain.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    df = pd.read_csv(raw_path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.sort_values(date_col).reset_index(drop=True)

    # Split into 5 contiguous batches (roughly equal size)
    n = len(df)
    idx = [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n]

    out_dir = ROOT / "data/batches"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(5):
        batch = df.iloc[idx[i] : idx[i + 1]].copy()
        out_path = out_dir / f"batch_{i+1:02d}.csv"
        batch.to_csv(out_path, index=False)
        print(f"Wrote {out_path} rows={len(batch)} dates=[{batch[date_col].min().date()}..{batch[date_col].max().date()}]")


if __name__ == "__main__":
    main()
