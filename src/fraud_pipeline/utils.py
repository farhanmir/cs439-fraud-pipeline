from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def coerce_binary_target(series: pd.Series) -> pd.Series:
    if set(series.dropna().unique()).issubset({0, 1}):
        return series.astype(int)

    if series.dtype == object:
        normalized = series.astype(str).str.strip().str.lower()
        mapping = {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1,
            "no": 0,
            "yes": 1,
            "non-fraud": 0,
            "fraud": 1,
            "legit": 0,
            "fraudulent": 1,
        }
        if set(normalized.unique()).issubset(mapping):
            return normalized.map(mapping).astype(int)

    raise ValueError(
        "Target column must be binary or mappable to binary labels. "
        "Supported string labels include yes/no, true/false, fraud/non-fraud."
    )

