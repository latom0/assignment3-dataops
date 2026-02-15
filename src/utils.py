from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_params() -> Dict[str, Any]:
    with open(ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    (ROOT / "data/bronze").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/silver").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/gold").mkdir(parents=True, exist_ok=True)
    (ROOT / "evidence/logs").mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_parquet_or_empty(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=columns) if columns else pd.DataFrame()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_json_or_default(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def coerce_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    return out


def time_span(df: pd.DataFrame, date_col: str) -> Tuple[str | None, str | None]:
    if df.empty:
        return None, None
    mn = df[date_col].min()
    mx = df[date_col].max()
    return (mn.date().isoformat() if pd.notna(mn) else None), (mx.date().isoformat() if pd.notna(mx) else None)
