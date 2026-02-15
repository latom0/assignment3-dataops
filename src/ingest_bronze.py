from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils import (
    ROOT,
    coerce_date,
    ensure_dirs,
    load_params,
    read_json_or_default,
    read_parquet_or_empty,
    sha256_file,
    time_span,
    write_json,
)

# Toggle to simulate issues (missing values, duplicates, dropped rows)
SIMULATE_ISSUES = True
ISSUE_SEED = 42


def simulate_quality_issues(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    rng = np.random.default_rng(ISSUE_SEED)
    out = df.copy()

    # Randomly drop ~0.5% rows
    if len(out) > 200:
        drop_n = max(1, int(0.005 * len(out)))
        drop_idx = rng.choice(out.index.to_numpy(), size=drop_n, replace=False)
        out = out.drop(index=drop_idx)

    # Introduce missing values in numeric columns (~0.3% of cells per column)
    num_cols = [c for c in out.columns if c != date_col]
    for c in num_cols:
        if len(out) > 0:
            miss_n = max(1, int(0.003 * len(out)))
            miss_idx = rng.choice(out.index.to_numpy(), size=miss_n, replace=False)
            out.loc[miss_idx, c] = np.nan

    # Duplicate a few timestamps (copy rows)
    if len(out) > 100:
        dup_n = 2
        dup_idx = rng.choice(out.index.to_numpy(), size=dup_n, replace=False)
        out = pd.concat([out, out.loc[dup_idx]], ignore_index=True)

    return out


def main() -> None:
    ensure_dirs()
    params = load_params()
    date_col = params["data"]["date_col"]

    incoming_dir = ROOT / "data/incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)

    bronze_path = ROOT / "data/bronze/bronze.parquet"
    lineage_path = ROOT / "data/bronze/lineage.json"

    lineage: Dict[str, Any] = read_json_or_default(lineage_path, default={"ingested_batches": []})
    ingested = {b["batch_name"] for b in lineage.get("ingested_batches", [])}

    batch_files = sorted(incoming_dir.glob("batch_*.csv"))
    new_files = [p for p in batch_files if p.name not in ingested]

    if not new_files:
        print("No new incoming batches to ingest.")
        # still ensure files exist for DVC stage stability
        if not bronze_path.exists():
            pd.DataFrame().to_parquet(bronze_path, index=False)
        if not lineage_path.exists():
            write_json(lineage_path, lineage)
        return

    bronze_df = read_parquet_or_empty(bronze_path)

    for p in new_files:
        raw = pd.read_csv(p)
        raw = coerce_date(raw, date_col=date_col)
        raw["__batch_name"] = p.name

        if SIMULATE_ISSUES:
            raw = simulate_quality_issues(raw, date_col=date_col)

        batch_hash = sha256_file(p)
        mn, mx = time_span(raw, date_col)

        record = {
            "batch_name": p.name,
            "file_sha256": batch_hash,
            "rows_read": int(len(raw)),
            "date_min": mn,
            "date_max": mx,
        }

        lineage["ingested_batches"].append(record)

        bronze_df = pd.concat([bronze_df, raw], ignore_index=True)
        print(f"Ingested {p.name}: rows={len(raw)} sha256={batch_hash[:10]}... span={mn}..{mx}")

    # Preserve raw schema/values as much as possible; just store what arrived
    bronze_df.to_parquet(bronze_path, index=False)
    write_json(lineage_path, lineage)
    print(f"Bronze updated: {bronze_path} rows_total={len(bronze_df)}")


if __name__ == "__main__":
    main()
