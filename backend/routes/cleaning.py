"""
cleaning.py  —  /api/clean

Basic Data Cleaning endpoints (all changes are user-triggered, never automatic):
  GET  /api/clean/analyze        → scan dataset: duplicates, dtype issues, irrelevant columns
  POST /api/clean/duplicates     → remove duplicate rows
  POST /api/clean/fix-dtypes     → apply suggested / user-confirmed dtype conversions
  POST /api/clean/drop-columns   → drop user-selected columns
  GET  /api/clean/preview        → return current processed_df preview + stats
"""
import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.services.session_store import store
from backend.utils.helpers import df_to_records, safe_json

router = APIRouter(prefix="/api/clean", tags=["Cleaning"])
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Request schemas (local — keep cleaning self-contained)
# ─────────────────────────────────────────────────────────────────────────────
class FixDtypesRequest(BaseModel):
    conversions: List[Dict[str, str]]   # [{ "column": "Age", "to": "numeric" }, ...]


class DropColumnsRequest(BaseModel):
    columns: List[str]                  # column names to drop


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
_ID_PATTERNS = re.compile(
    r"^(id|index|uuid|key|rownum|row_num|row_id|serial|record_?id|_id)$",
    re.IGNORECASE,
)


def _get_df() -> pd.DataFrame:
    """
    Return the working DataFrame — same fallback logic as the EDA route:
    prefer processed_df (may have had cleaning applied), fall back to raw_df.
    Raises 404 if neither is available.
    """
    df = store.get("processed_df")
    if df is None:
        df = store.get("raw_df")
    if df is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset loaded. Please upload a CSV file first.",
        )
    # If processed_df was None but raw_df is available, sync them so future
    # cleaning operations always have a processed_df to mutate.
    if store.get("processed_df") is None:
        store.set("processed_df", df)
    return df


def _suggest_irrelevant(df: pd.DataFrame, target: Optional[str]) -> List[str]:
    """Heuristically flag likely-irrelevant columns (ID cols, near-unique object cols)."""
    suggestions = []
    for col in df.columns:
        if col == target:
            continue
        # Name matches known ID patterns
        if _ID_PATTERNS.match(col.strip()):
            suggestions.append(col)
            continue
        # Object column where almost every value is unique (like an ID or name field)
        if df[col].dtype == object:
            n_unique = df[col].nunique(dropna=True)
            if n_unique > 0.9 * len(df) and len(df) > 20:
                suggestions.append(col)
    return suggestions


def _detect_dtype_issues(df: pd.DataFrame, target: Optional[str]) -> List[Dict[str, str]]:
    """
    Identify columns stored as object that look numeric or date-like.
    Returns a list of { column, current_dtype, suggested_dtype, sample_values }.
    """
    issues = []
    for col in df.columns:
        if col == target:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        current_dtype = str(df[col].dtype)

        # Only flag object columns
        if df[col].dtype != object:
            continue

        # Try numeric conversion
        try:
            converted = pd.to_numeric(series, errors="raise")
            suggested = "float64" if converted.dtype in [float] else "int64"
            try:
                suggested = str(pd.to_numeric(series, errors="coerce").dropna().astype(int).dtype)
                if (pd.to_numeric(series, errors="coerce") % 1).sum() != 0:
                    suggested = "float64"
            except Exception:
                suggested = "numeric"
            issues.append({
                "column": col,
                "current_dtype": current_dtype,
                "suggested_dtype": "numeric",
                "suggested_label": f"{col}: object → numeric",
                "sample_values": [str(v) for v in series.head(3).tolist()],
            })
            continue
        except (ValueError, TypeError):
            pass

        # Try datetime conversion — do NOT use infer_datetime_format
        # (deprecated in pandas 2.0, removed in pandas 2.2).
        sample = series.head(50)
        try:
            pd.to_datetime(sample, errors="raise")
            issues.append({
                "column": col,
                "current_dtype": current_dtype,
                "suggested_dtype": "datetime",
                "suggested_label": f"{col}: object → datetime",
                "sample_values": [str(v) for v in series.head(3).tolist()],
            })
        except Exception:
            pass

    return issues


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/analyze", summary="Scan dataset for cleaning opportunities")
async def analyze():
    """
    Non-destructive scan of the current dataset.
    Falls back to raw_df when processed_df is unavailable (e.g. after server
    restart) — same resilience pattern used by the EDA route.
    """
    try:
        df = _get_df()
        target = store.get("target_column")

        # Duplicate analysis
        # keep='first'  → marks only the rows that will be REMOVED by drop_duplicates(keep='first')
        #                  i.e. 2nd, 3rd, … copies — NOT the original row kept.
        # keep=False     → marks ALL copies including the first → inflated count equal to total rows
        #                  when every row has at least one duplicate.
        try:
            dup_mask  = df.duplicated(keep="first")   # rows that will actually be removed
            dup_count = int(dup_mask.sum())
            dup_preview = df_to_records(df[dup_mask].head(10)) if dup_count > 0 else []
        except Exception:
            dup_count, dup_preview = 0, []

        # Dtype issues (defensive — some exotic dtypes may trip the heuristic)
        try:
            dtype_issues = _detect_dtype_issues(df, target)
        except Exception as exc:
            logger.warning("[Clean] dtype detection failed: %s", exc)
            dtype_issues = []

        # Irrelevant column suggestions
        try:
            suggested_irrelevant = _suggest_irrelevant(df, target)
        except Exception:
            suggested_irrelevant = []

        # Per-column info
        columns_info = [
            {
                "name"         : col,
                "dtype"        : str(df[col].dtype),
                "missing"      : int(df[col].isnull().sum()),
                "n_unique"     : int(df[col].nunique(dropna=True)),
                "is_target"    : col == target,
                "suggested_drop": col in suggested_irrelevant,
            }
            for col in df.columns
        ]

        return JSONResponse(content=safe_json({
            "shape"              : {"rows": len(df), "columns": len(df.columns)},
            "duplicates"         : {"count": dup_count, "preview": dup_preview},
            "dtype_issues"       : dtype_issues,
            "columns_info"       : columns_info,
            "suggested_irrelevant": suggested_irrelevant,
        }))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Clean /analyze] unexpected error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@router.post("/duplicates", summary="Remove duplicate rows")
async def remove_duplicates():
    """Remove duplicate rows from processed_df (keeps first occurrence)."""
    df = _get_df()

    before = len(df)
    df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = before - len(df_clean)

    store.set("processed_df", df_clean)

    # Append to cleaning log
    log = store.get("cleaning_log") or []
    log.append(f"Removed {removed} duplicate row(s). Dataset: {len(df_clean)} rows remaining.")
    store.set("cleaning_log", log)

    logger.info("[Clean] Removed %d duplicates — %d rows remain", removed, len(df_clean))
    return JSONResponse(content=safe_json({
        "removed": removed,
        "rows_after": len(df_clean),
        "columns_after": len(df_clean.columns),
        "message": f"Removed {removed} duplicate row(s). Dataset now has {len(df_clean)} rows.",
    }))


@router.post("/fix-dtypes", summary="Apply dtype conversions to selected columns")
async def fix_dtypes(body: FixDtypesRequest):
    """
    Convert columns to their suggested type (numeric or datetime).
    Only columns explicitly listed in `body.conversions` are changed.
    """
    df = _get_df().copy()
    applied = []
    errors = []

    for item in body.conversions:
        col = item.get("column")
        to  = item.get("to", "").lower()

        if col not in df.columns:
            errors.append(f"Column '{col}' not found.")
            continue

        try:
            if to == "numeric":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                applied.append(f"{col}: object → numeric")
            elif to == "datetime":
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                applied.append(f"{col}: object → datetime")
            else:
                errors.append(f"Unknown target type '{to}' for column '{col}'.")
        except Exception as exc:
            errors.append(f"Failed to convert '{col}': {exc}")

    store.set("processed_df", df)

    log = store.get("cleaning_log") or []
    for entry in applied:
        log.append(f"Fixed dtype: {entry}")
    store.set("cleaning_log", log)

    logger.info("[Clean] Dtype fixes applied: %s", applied)
    return JSONResponse(content=safe_json({
        "applied": applied,
        "errors": errors,
        "rows_after": len(df),
        "columns_after": len(df.columns),
        "message": f"Applied {len(applied)} conversion(s)." + (f" {len(errors)} error(s)." if errors else ""),
    }))


@router.post("/drop-columns", summary="Drop user-selected columns")
async def drop_columns(body: DropColumnsRequest):
    """Drop the columns listed in `body.columns` from processed_df."""
    df = _get_df()

    target = store.get("target_column")
    to_drop = [c for c in body.columns if c in df.columns]
    protected = [c for c in body.columns if c == target]

    if protected:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot drop the target column: {protected}.",
        )
    if not to_drop:
        raise HTTPException(status_code=400, detail="None of the specified columns were found.")

    df_clean = df.drop(columns=to_drop).reset_index(drop=True)
    store.set("processed_df", df_clean)

    # Update feature_columns list if set
    feature_cols = store.get("feature_columns")
    if feature_cols:
        store.set("feature_columns", [c for c in feature_cols if c not in to_drop])

    log = store.get("cleaning_log") or []
    log.append(f"Dropped {len(to_drop)} column(s): {', '.join(to_drop)}.")
    store.set("cleaning_log", log)

    logger.info("[Clean] Dropped columns: %s", to_drop)
    return JSONResponse(content=safe_json({
        "dropped": to_drop,
        "rows_after": len(df_clean),
        "columns_after": len(df_clean.columns),
        "remaining_columns": list(df_clean.columns),
        "message": f"Dropped {len(to_drop)} column(s): {', '.join(to_drop)}.",
    }))


@router.get("/preview", summary="Get current cleaned dataset preview")
async def get_preview():
    """Return a preview of the current processed_df and the action log."""
    df = _get_df()

    log = store.get("cleaning_log") or []
    return JSONResponse(content=safe_json({
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "missing_counts": {col: int(cnt) for col, cnt in df.isnull().sum().items()},
        "preview": df_to_records(df, max_rows=50),
        "cleaning_log": log,
    }))
