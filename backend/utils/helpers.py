"""
helpers.py
----------
Generic utility functions shared across routes.
"""
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def df_to_records(df: pd.DataFrame, max_rows: int = 500) -> List[Dict[str, Any]]:
    """Convert a DataFrame to a JSON-serialisable list of dicts (handles pd.NA, np.nan)."""
    subset = df.head(max_rows)
    records = []
    for row in subset.itertuples(index=False):
        records.append(
            {col: safe_json(val) for col, val in zip(subset.columns, row)}
        )
    return records


def infer_task_type(series: pd.Series) -> str:
    """
    Heuristic: if the target has ≤20 unique values and is object/int
    with few unique entries → classification, else regression.
    """
    if series.dtype == object or series.nunique() <= 20:
        return "classification"
    return "regression"


def safe_json(obj: Any) -> Any:
    """
    Recursively convert any non-JSON-serialisable value to a native
    Python type.  Handles:
      - numpy integers / floats / booleans / ndarrays
      - pandas NA  (pd.NA, pd.NaT, np.nan)
      - pandas Timestamp / Timedelta
      - Python sets
      - Any remaining unknown type → None (safe fallback)
    """
    # --- dict / list containers ---
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, set):
        return [safe_json(v) for v in sorted(obj, key=str)]

    # --- numpy scalars ---
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [safe_json(v) for v in obj.tolist()]

    # --- pandas NA / NaT / NaN ---
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, float) and np.isnan(obj):
        return None

    # --- pandas Timestamp / Timedelta ---
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)

    # --- native Python types (pass through) ---
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj

    # --- unknown / exotic type: log a warning and return None ---
    logger.warning("safe_json: unhandled type %s for value %r — coercing to None", type(obj).__name__, obj)
    return None
