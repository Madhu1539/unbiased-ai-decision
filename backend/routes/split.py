"""
split.py  —  /api/split

Production-grade train/test split with:
  GET  /api/split/analyze   → pre-split dataset analysis (types, classes, datetime, groups)
  POST /api/split           → execute split (random / stratified / chronological / group-based)
  GET  /api/split/status    → current split info + leakage audit results
"""
import logging
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from backend.services.session_store import store
from backend.utils.helpers import infer_task_type, safe_json

router = APIRouter(prefix="/api/split", tags=["Split"])
logger = logging.getLogger(__name__)

# ─── Patterns ─────────────────────────────────────────────────────────────────
_GROUP_ID_PATTERN = re.compile(
    r"(^|[_\-\s])(id|uuid|key|hash|code|token|reference|ref|tx|transaction|patient|user|customer|order|account|serial|record|row)([_\-\s]|$)",
    re.IGNORECASE,
)


# ─── Request schema ───────────────────────────────────────────────────────────
class SplitRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True
    stratify: bool = True
    split_method: str = "stratified"      # random | stratified | chronological | group
    datetime_column: Optional[str] = None  # required for chronological
    group_column: Optional[str] = None    # required for group-based


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _load_df() -> pd.DataFrame:
    df = store.get("processed_df")
    if df is None:
        df = store.get("raw_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded. Please upload a CSV first.")
    if store.get("processed_df") is None:
        store.set("processed_df", df)
    return df


def _detect_datetime_cols(df: pd.DataFrame) -> List[str]:
    """Find object columns whose values look like dates."""
    candidates = []
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(30)
            if sample.empty:
                continue
            try:
                pd.to_datetime(sample, errors="raise")
                candidates.append(col)
            except Exception:
                pass
    return candidates


def _detect_group_cols(df: pd.DataFrame, target: Optional[str]) -> List[str]:
    """Detect likely ID / group columns by name pattern AND high cardinality."""
    result = []
    n = len(df)
    for col in df.columns:
        if col == target:
            continue
        # Name matches known group/ID patterns
        if _GROUP_ID_PATTERN.search(col):
            result.append(col)
            continue
        # High-cardinality object column (>80% unique) is likely an ID
        if df[col].dtype == object:
            n_unique = df[col].nunique(dropna=True)
            if n > 10 and n_unique / n > 0.80:
                result.append(col)
    return result


def _class_distribution(series: pd.Series) -> Dict[str, Any]:
    counts = series.value_counts()
    total  = len(series)
    return {
        str(cls): {
            "count": int(cnt),
            "pct":   round(100 * cnt / total, 2),
        }
        for cls, cnt in counts.items()
    }


def _leakage_audit(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    target:  Optional[str],
) -> List[Dict[str, Any]]:
    """
    Check ONLY high-cardinality columns for value overlap between train and test.
    High-cardinality = unique values > max(50, 50% of total rows).
    This avoids flagging common categorical values (Male/Female, city names).
    """
    total    = len(X_train) + len(X_test)
    min_unique = max(50, int(0.50 * total))
    findings = []

    all_cols = [c for c in X_train.columns if c != target]
    for col in all_cols:
        if col not in X_test.columns:
            continue
        # Only audit columns that have many unique values overall
        combined_unique = pd.concat([X_train[col], X_test[col]]).nunique(dropna=True)
        if combined_unique < min_unique:
            continue        # Skip low-cardinality → not an ID / leakage risk

        train_vals = set(X_train[col].dropna().unique())
        test_vals  = set(X_test[col].dropna().unique())
        overlap    = train_vals & test_vals

        if overlap:
            overlap_pct = round(100 * len(overlap) / len(test_vals), 1) if test_vals else 0
            findings.append({
                "column":      col,
                "unique_total": combined_unique,
                "overlap_count": len(overlap),
                "test_unique":  len(test_vals),
                "overlap_pct":  overlap_pct,
            })

    return findings


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/analyze", summary="Pre-split dataset analysis")
async def analyze():
    """
    Analyses the dataset before splitting:
    - Detects problem type (classification / regression)
    - Detects datetime columns (chronological split candidate)
    - Detects potential group/ID columns
    - Returns class distribution (classification only)
    - Returns dataset-size guidance
    - Validates minority class counts
    """
    try:
        df     = _load_df()
        target = store.get("target_column")

        if not target or target not in df.columns:
            raise HTTPException(status_code=400, detail="Target column not set. Complete Data Upload first.")

        _task = store.get("task_type")
        task  = _task if _task else infer_task_type(df[target])
        n    = len(df)

        # Feature columns — always re-derive from current df to avoid stale lists
        _stored_feats = store.get("feature_columns")
        if _stored_feats and isinstance(_stored_feats, list) and len(_stored_feats) > 0:
            feature_cols = [c for c in _stored_feats if c in df.columns and c != target]
        else:
            feature_cols = [c for c in df.columns if c != target]

        # Class distribution (classification only)
        class_dist:       Dict[str, Any] = {}
        minority_count:   int  = 0
        minority_class:   str  = ""
        stratify_ok:      bool = True
        class_warnings:   List[str] = []
        class_errors:     List[str] = []

        if task == "classification":
            class_dist   = _class_distribution(df[target])
            counts       = df[target].value_counts()
            minority_count = int(counts.min())
            minority_class = str(counts.idxmin())

            if minority_count < 2:
                stratify_ok = False
                class_errors.append(
                    f"Class '{minority_class}' has only {minority_count} sample(s). "
                    "Splitting is not possible — add more data for this class."
                )
            elif minority_count < 10:
                class_warnings.append(
                    f"Minority class '{minority_class}' has only {minority_count} sample(s). "
                    "Results may be unstable. Consider data balancing after splitting."
                )

        # Datetime detection
        datetime_cols = _detect_datetime_cols(df)

        # Group/ID column detection
        group_cols = _detect_group_cols(df, target)

        # Dataset-size guidance
        size_guidance: Optional[str] = None
        if n < 500:
            size_guidance = (
                f"Your dataset has only {n} rows. "
                "Consider using cross-validation instead of a fixed test split for more reliable estimates."
            )
        elif n > 1_000_000:
            size_guidance = (
                f"Your dataset has {n:,} rows. "
                "A smaller test size (e.g. 5% or 1%) may be sufficient."
            )

        return JSONResponse(content=safe_json({
            "n_rows":          n,
            "n_cols":          len(df.columns),
            "n_features":      len(feature_cols),
            "task_type":       task,
            "target":          target,
            "class_dist":      class_dist,
            "minority_count":  minority_count,
            "minority_class":  minority_class,
            "stratify_ok":     stratify_ok,
            "class_warnings":  class_warnings,
            "class_errors":    class_errors,
            "datetime_cols":   datetime_cols,
            "group_cols":      group_cols,
            "size_guidance":   size_guidance,
            "feature_cols":    feature_cols,
        }))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Split /analyze] error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@router.post("", summary="Execute train/test split")
async def perform_split(body: SplitRequest):
    """
    Execute split using the chosen method:
    - random       : standard random split
    - stratified   : stratified by target class (classification only)
    - chronological: sort by datetime column → split sequentially
    - group        : GroupShuffleSplit so groups don't span train and test
    """
    # ── Validation ────────────────────────────────────────────────────
    if not (0.01 <= body.test_size <= 0.6):
        raise HTTPException(status_code=400, detail="test_size must be between 0.01 and 0.60.")

    df     = _load_df()
    target = store.get("target_column")
    if not target:
        raise HTTPException(status_code=400, detail="Target column not set. Complete Data Upload first.")
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in dataset.")

    _task = store.get("task_type")
    task  = _task if _task else infer_task_type(df[target])

    _stored_feats = store.get("feature_columns")
    if _stored_feats and isinstance(_stored_feats, list) and len(_stored_feats) > 0:
        feature_cols = [c for c in _stored_feats if c in df.columns and c != target]
    else:
        feature_cols = [c for c in df.columns if c != target]
    if not feature_cols:
        raise HTTPException(status_code=400, detail="No feature columns found.")

    n = len(df)
    if n < 10:
        raise HTTPException(status_code=400, detail="Dataset too small to split (minimum 10 rows).")

    # ── Minority class safety check ───────────────────────────────────
    if task == "classification":
        counts = df[target].value_counts()
        min_count = int(counts.min())
        min_class = str(counts.idxmin())
        if min_count < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Class '{min_class}' has fewer than 2 samples. Splitting is not possible.",
            )

    # ── Method-specific validation ────────────────────────────────────
    if body.split_method == "chronological":
        if not body.datetime_column:
            raise HTTPException(status_code=400, detail="A datetime column must be selected for chronological split.")
        if body.datetime_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Datetime column '{body.datetime_column}' not found.")

    if body.split_method == "group":
        if not body.group_column:
            raise HTTPException(status_code=400, detail="A group column must be selected for group-based split.")
        if body.group_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Group column '{body.group_column}' not found.")

    # ── Execute split ─────────────────────────────────────────────────
    X = df[feature_cols]
    y = df[target]
    warning_msg: Optional[str] = None
    method_label: str = ""

    try:
        if body.split_method == "chronological":
            # Sort by datetime → sequential split
            df_sorted  = df.copy()
            df_sorted[body.datetime_column] = pd.to_datetime(df_sorted[body.datetime_column], errors="coerce")
            df_sorted  = df_sorted.sort_values(body.datetime_column).reset_index(drop=True)
            split_idx  = int(len(df_sorted) * (1 - body.test_size))
            train_df   = df_sorted.iloc[:split_idx]
            test_df    = df_sorted.iloc[split_idx:]
            X_train    = train_df[feature_cols]
            X_test     = test_df[feature_cols]
            y_train    = train_df[target]
            y_test     = test_df[target]
            method_label = "Chronological Split"

        elif body.split_method == "group":
            groups = df[body.group_column]
            gss    = GroupShuffleSplit(n_splits=1, test_size=body.test_size, random_state=body.random_state)
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test  = y.iloc[test_idx]
            method_label = f"Group-Based Split (column: {body.group_column})"

        elif body.split_method == "stratified" and task == "classification":
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=body.test_size,
                    random_state=body.random_state,
                    shuffle=body.shuffle,
                    stratify=y,
                )
                method_label = "Stratified Random Split"
            except ValueError as exc:
                warning_msg = (
                    "Stratified split failed (a class is too rare). "
                    "Falling back to standard random split."
                )
                logger.warning("[Split] stratify fallback: %s", exc)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=body.test_size,
                    random_state=body.random_state,
                    shuffle=body.shuffle,
                )
                method_label = "Standard Random Split (stratify fallback)"

        else:  # random
            if not body.shuffle:
                warning_msg = "Shuffle is OFF. Rows will be split in their current order — ensure data is not sorted."
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=body.test_size,
                random_state=body.random_state,
                shuffle=body.shuffle,
            )
            method_label = "Standard Random Split"

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("[Split] error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Split failed: {exc}")

    # ── Leakage audit ─────────────────────────────────────────────────
    leakage_findings = _leakage_audit(X_train, X_test, target)
    leakage_warning  = None
    if leakage_findings:
        cols = ", ".join(f["column"] for f in leakage_findings[:3])
        leakage_warning = (
            f"Potential data leakage: same high-cardinality entities detected "
            f"in both Train and Test for column(s): {cols}. "
            "Consider using Group-Based Splitting."
        )

    # ── Class distributions ────────────────────────────────────────────
    class_dist_train: dict = {}
    class_dist_test:  dict = {}
    if task == "classification":
        class_dist_train = _class_distribution(y_train)
        class_dist_test  = _class_distribution(y_test)

    # ── Store in session ───────────────────────────────────────────────
    store.update({
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "feature_columns": feature_cols,
        "split_config": {
            "test_size":      body.test_size,
            "random_state":   body.random_state,
            "shuffle":        body.shuffle,
            "stratify":       body.split_method == "stratified",
            "split_method":   body.split_method,
            "datetime_column":body.datetime_column,
            "group_column":   body.group_column,
            "task_type":      task,
            "method_label":   method_label,
        },
    })

    logger.info(
        "[Split] method=%s  total=%d  train=%d  test=%d  features=%d  task=%s",
        body.split_method, n, len(X_train), len(X_test), len(feature_cols), task,
    )

    return JSONResponse(content=safe_json({
        "message":          warning_msg or f"Split applied: {method_label}",
        "method_label":     method_label,
        "total_rows":       n,
        "train_rows":       int(X_train.shape[0]),
        "test_rows":        int(X_test.shape[0]),
        "train_pct":        round(100 * (1 - body.test_size), 1),
        "test_pct":         round(100 * body.test_size, 1),
        "features":         len(feature_cols),
        "stratified":       body.split_method == "stratified",
        "task_type":        task,
        "random_state":     body.random_state,
        "class_dist_train": class_dist_train,
        "class_dist_test":  class_dist_test,
        "leakage_findings": leakage_findings,
        "leakage_warning":  leakage_warning,
        "warning":          warning_msg,
    }))


@router.get("/status", summary="Current split info")
async def split_status():
    """Return current split stored in session, or split_done=False."""
    X_train = store.get("X_train")
    X_test  = store.get("X_test")
    config  = store.get("split_config")

    if X_train is None or X_test is None:
        return JSONResponse(content={"split_done": False})

    return JSONResponse(content=safe_json({
        "split_done": True,
        "train_rows": int(X_train.shape[0]),
        "test_rows":  int(X_test.shape[0]),
        "features":   int(X_train.shape[1]),
        "config":     config if config is not None else {},
    }))
