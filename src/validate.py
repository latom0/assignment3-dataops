from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

from utils import ROOT, ensure_dirs, load_params, read_json_or_default, write_json


def build_schema(date_col: str, ranges: Dict[str, List[float]], allow_na: bool) -> DataFrameSchema:
    # Numeric columns may have NA in bronze; silver ideally fewer NA after imputation
    return DataFrameSchema(
        {
            date_col: Column(pa.DateTime, nullable=False),
            "meantemp": Column(float, nullable=allow_na, checks=Check.in_range(*ranges["meantemp"])),
            "humidity": Column(float, nullable=allow_na, checks=Check.in_range(*ranges["humidity"])),
            "wind_speed": Column(float, nullable=allow_na, checks=Check.in_range(*ranges["wind_speed"])),
            "meanpressure": Column(float, nullable=allow_na, checks=Check.in_range(*ranges["meanpressure"])),
        },
        strict=False,  # allow extra columns like __batch_name or engineered features (silver/gold)
        coerce=True,
    )


def continuity_check(df: pd.DataFrame, date_col: str) -> Tuple[bool, int]:
    if df.empty:
        return True, 0
    days = df[date_col].dropna().sort_values().unique()
    if len(days) == 0:
        return False, 0
    full = pd.date_range(start=days[0], end=days[-1], freq="D")
    missing = len(set(full) - set(days))
    return missing == 0, int(missing)


def duplicate_days_check(df: pd.DataFrame, date_col: str) -> Tuple[bool, int]:
    if df.empty:
        return True, 0
    dup_count = int(df[date_col].duplicated().sum())
    return dup_count == 0, dup_count


def expected_no_silent_loss(layer: str, df: pd.DataFrame, lineage: Dict[str, Any]) -> Dict[str, Any]:
    # Bronze: compare rows in parquet to sum(rows_read) in lineage
    if layer == "bronze":
        expected = int(sum(b.get("rows_read", 0) for b in lineage.get("ingested_batches", [])))
        actual = int(len(df))
        return {"expected_rows_from_lineage": expected, "actual_rows_in_layer": actual, "match": expected == actual}

    # Silver: report drops vs bronze (not "must match", but must be explicit)
    if layer == "silver":
        bronze_rows = lineage.get("bronze_rows_total")
        if bronze_rows is None:
            return {"note": "bronze_rows_total missing; run bronze validation first"}
        actual = int(len(df))
        return {"bronze_rows_total": int(bronze_rows), "silver_rows_total": actual, "dropped_rows": int(bronze_rows) - actual}

    return {"note": "no silent loss check not implemented for this layer"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=["bronze", "silver"], required=True)
    args = parser.parse_args()

    ensure_dirs()
    params = load_params()
    date_col = params["data"]["date_col"]
    ranges = params["ranges"]

    if args.layer == "bronze":
        data_path = ROOT / "data/bronze/bronze.parquet"
        lineage_path = ROOT / "data/bronze/lineage.json"
        allow_na = True
        out_path = ROOT / "evidence/logs/validate_bronze.json"
    else:
        data_path = ROOT / "data/silver/silver.parquet"
        lineage_path = ROOT / "data/bronze/lineage.json"
        allow_na = False
        out_path = ROOT / "evidence/logs/validate_silver.json"

    lineage = read_json_or_default(lineage_path, default={"ingested_batches": []})
    df = pd.read_parquet(data_path)

    # Force datetime normalization
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    results: Dict[str, Any] = {"layer": args.layer, "rows": int(len(df))}

    # Pandera schema / range checks
    schema = build_schema(date_col, ranges, allow_na=allow_na)
    try:
        schema.validate(df, lazy=True)
        results["schema_validation"] = {"pass": True}
    except pa.errors.SchemaErrors as e:
        results["schema_validation"] = {
            "pass": False,
            "failure_cases_sample": e.failure_cases.head(25).to_dict(orient="records"),
        }

    # Temporal checks
    cont_ok, missing_days = continuity_check(df, date_col)
    dup_ok, dup_days = duplicate_days_check(df, date_col)
    results["time_continuity"] = {"pass": bool(cont_ok), "missing_days": int(missing_days)}
    results["duplicate_days"] = {"pass": bool(dup_ok), "duplicate_count": int(dup_days)}

    # “No silent loss” style accounting
    if args.layer == "bronze":
        # store bronze row total for later silver check
        lineage["bronze_rows_total"] = int(len(df))
        write_json(lineage_path, lineage)

    results["row_accounting"] = expected_no_silent_loss(args.layer, df, lineage)

    # Overall pass/fail (bronze can have NA but should still pass checks ideally)
    results["overall_pass"] = bool(
        results["schema_validation"]["pass"] and results["time_continuity"]["pass"] and results["duplicate_days"]["pass"]
    )

    write_json(out_path, results)
    print(f"Wrote validation log: {out_path}")
    print(f"Overall pass: {results['overall_pass']}")


if __name__ == "__main__":
    main()
