"""
features.py  —  /api/features

Leakage-free Feature Engineering with strict train-only learning.

Endpoints:
  GET  /api/features/status        → check if split has been done, return X_train shape
  GET  /api/features/analyze       → scan X_train for opportunities + correlation warnings
  POST /api/features/formula       → create formula-based feature (safe for both sets)
  POST /api/features/select        → feature selection fitted on X_train only
  GET  /api/features/correlation   → correlation matrix on X_train, flag pairs > 0.95
  POST /api/features/apply         → build & store sklearn Pipeline, transform both sets
  POST /api/features/undo          → undo last feature action (restore checkpoint)
  POST /api/features/reset         → reset to original split state
  GET  /api/features/preview       → current X_train head + feature list
"""
import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from starlette.concurrency import run_in_threadpool

from backend.services.session_store import store
from backend.utils.helpers import infer_task_type, safe_json

router = APIRouter(prefix="/api/features", tags=["Feature Engineering"])
logger = logging.getLogger(__name__)


# ─── Schemas ──────────────────────────────────────────────────────────────────
class FormulaRequest(BaseModel):
    name: str
    col_a: str
    col_b: Optional[str] = None
    operation: str = "ratio"    # ratio | multiply | add | subtract | log | square


class SelectRequest(BaseModel):
    method: str = "kbest"       # kbest | importance | rfe
    k: int = 10


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_split():
    """Return (X_train, X_test, y_train, y_test) or raise 400."""
    X_train = store.get("X_train")
    X_test  = store.get("X_test")
    y_train = store.get("y_train")
    y_test  = store.get("y_test")
    if X_train is None or X_test is None:
        raise HTTPException(
            status_code=400,
            detail="No train/test split found. Please complete the 'Split Data' step first.",
        )
    return X_train.copy(), X_test.copy(), y_train, y_test


def _save_checkpoint():
    """Save current X_train/X_test snapshot for undo."""
    X_train = store.get("X_train")
    X_test  = store.get("X_test")
    if X_train is not None:
        store.set("fe_checkpoint_train", X_train.copy())
    if X_test is not None:
        store.set("fe_checkpoint_test",  X_test.copy())


def _leakage_check(name: str, target: Optional[str]) -> Optional[str]:
    """Warn if feature name matches the target."""
    if target and name.lower() == target.lower():
        return f"Feature name '{name}' matches the target column — this would cause target leakage."
    return None


def _quality_warnings(series: pd.Series, name: str) -> List[str]:
    ws = []
    if series.nunique(dropna=True) <= 1:
        ws.append(f"'{name}' is a constant feature (all values identical) — it adds no information.")
    return ws


def _apply_formula(X: pd.DataFrame, name: str, col_a: str,
                   col_b: Optional[str], op: str) -> pd.DataFrame:
    """Apply a formula-based feature transformation. Safe to apply identically on train and test."""
    X = X.copy()
    a = pd.to_numeric(X[col_a], errors="coerce")

    if op == "log":
        X[name] = np.log1p(a.clip(lower=0))
    elif op == "square":
        X[name] = a ** 2
    elif op in ("ratio", "multiply", "add", "subtract"):
        if col_b is None or col_b not in X.columns:
            raise ValueError(f"Operation '{op}' requires a second column (col_b).")
        b = pd.to_numeric(X[col_b], errors="coerce")
        if op == "ratio":
            X[name] = a / b.replace(0, np.nan)
        elif op == "multiply":
            X[name] = a * b
        elif op == "add":
            X[name] = a + b
        elif op == "subtract":
            X[name] = a - b
    else:
        raise ValueError(f"Unknown operation: '{op}'")

    return X


def _numeric_cols(X: pd.DataFrame) -> List[str]:
    return X.select_dtypes(include=[np.number]).columns.tolist()


def _encode_y(y) -> np.ndarray:
    if y is None:
        return np.zeros(0)
    if not pd.api.types.is_numeric_dtype(y):
        return LabelEncoder().fit_transform(y.astype(str))
    return y.fillna(0).values.astype(np.float64)


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/status", summary="Check if split is done and return shapes")
async def fe_status():
    X_train = store.get("X_train")
    X_test  = store.get("X_test")
    fe_log  = store.get("fe_action_log")
    if X_train is None:
        return JSONResponse(content={"split_done": False})
    return JSONResponse(content=safe_json({
        "split_done":  True,
        "train_rows":  int(X_train.shape[0]),
        "test_rows":   int(X_test.shape[0]) if X_test is not None else 0,
        "n_features":  int(X_train.shape[1]),
        "features":    list(X_train.columns),
        "action_count": len(fe_log) if fe_log else 0,
    }))


@router.get("/analyze", summary="Scan X_train for FE opportunities")
async def analyze():
    """
    Analyses X_train (train data ONLY) for:
    - Numeric column stats
    - Highly correlated pairs (> 0.90) → redundancy warnings
    - Constant / near-constant columns
    - Target leakage candidates (columns highly correlated with target)
    """
    try:
        X_train, _, y_train, _ = _get_split()
        target  = store.get("target_column")
        task    = store.get("task_type") or "classification"
        num_cols = _numeric_cols(X_train)

        # Column summary
        col_summary = []
        for col in X_train.columns:
            s = X_train[col]
            is_num = col in num_cols
            col_summary.append({
                "name":     col,
                "dtype":    str(s.dtype),
                "numeric":  is_num,
                "n_unique": int(s.nunique(dropna=True)),
                "missing":  int(s.isnull().sum()),
                "constant": bool(s.nunique(dropna=True) <= 1),
                "mean":     round(float(s.mean()), 4) if is_num else None,
                "std":      round(float(s.std()),  4) if is_num else None,
            })

        # Correlation check (numeric cols only, computed on X_train)
        corr_warnings: List[Dict] = []
        if len(num_cols) >= 2:
            corr_mat = X_train[num_cols].corr().abs()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    val = float(corr_mat.iloc[i, j])
                    if val > 0.90:
                        corr_warnings.append({
                            "col_a": num_cols[i],
                            "col_b": num_cols[j],
                            "correlation": round(val, 4),
                            "severe": val > 0.95,
                        })

        # Target correlation (leakage signal)
        target_corr: List[Dict] = []
        if task == "classification" and y_train is not None and len(num_cols) > 0:
            y_enc = _encode_y(y_train)
            for col in num_cols:
                try:
                    x_col = pd.to_numeric(X_train[col], errors="coerce").fillna(0).values
                    if len(x_col) == len(y_enc):
                        corr_val = float(abs(np.corrcoef(x_col, y_enc)[0, 1]))
                        if corr_val > 0.98:
                            target_corr.append({"col": col, "correlation": round(corr_val, 4)})
                except Exception:
                    pass

        # Formula suggestions (based on column types)
        suggestions: List[Dict] = []
        for i, ca in enumerate(num_cols):
            for cb in num_cols[i+1:i+4]:   # only suggest a few pairs
                suggestions.append({
                    "col_a": ca, "col_b": cb,
                    "operation": "ratio",
                    "suggested_name": f"{ca}_div_{cb}",
                    "reason": "Ratio between numeric columns often captures relative magnitude.",
                })
            if i < 3:   # first few columns get log suggestion
                suggestions.append({
                    "col_a": ca, "col_b": None,
                    "operation": "log",
                    "suggested_name": f"log_{ca}",
                    "reason": "Log transform reduces skewness in right-tailed distributions.",
                })

        return JSONResponse(content=safe_json({
            "train_rows":    int(X_train.shape[0]),
            "n_features":    int(X_train.shape[1]),
            "numeric_cols":  num_cols,
            "col_summary":   col_summary,
            "corr_warnings": corr_warnings,
            "target_corr":   target_corr,
            "suggestions":   suggestions[:12],   # cap at 12
        }))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[FE /analyze] %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@router.post("/formula", summary="Create a formula-based feature on train + test")
async def create_formula(body: FormulaRequest):
    """
    Formula features (ratio, multiply, log, etc.) are deterministic —
    they do NOT learn from data, so they can safely be applied identically
    to both X_train and X_test without any fitting step.
    """
    try:
        X_train, X_test, y_train, y_test = _get_split()
        target = store.get("target_column")

        # Leakage guard
        lk = _leakage_check(body.name, target)
        if lk:
            raise HTTPException(status_code=400, detail=lk)

        # Column existence
        for col in [body.col_a] + ([body.col_b] if body.col_b else []):
            if col not in X_train.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in training set.")

        # Duplicate name
        if body.name in X_train.columns:
            raise HTTPException(status_code=400, detail=f"Feature '{body.name}' already exists.")

        _save_checkpoint()

        # Apply identically to both sets (no fitting needed)
        X_train_new = await run_in_threadpool(_apply_formula, X_train, body.name, body.col_a, body.col_b, body.operation)
        X_test_new  = await run_in_threadpool(_apply_formula, X_test,  body.name, body.col_a, body.col_b, body.operation)

        # Quality warnings
        warnings = _quality_warnings(X_train_new[body.name], body.name)

        # Persist
        store.update({"X_train": X_train_new, "X_test": X_test_new})

        # Action log
        log = store.get("fe_action_log") if store.get("fe_action_log") is not None else []
        log.append({
            "type":    "formula",
            "name":    body.name,
            "op":      body.operation,
            "col_a":   body.col_a,
            "col_b":   body.col_b,
        })
        store.set("fe_action_log", log)

        # Sample preview
        sample = X_train_new[body.name].head(5).tolist()

        return JSONResponse(content=safe_json({
            "message":   f"Feature '{body.name}' created ({body.operation}).",
            "name":      body.name,
            "operation": body.operation,
            "sample":    sample,
            "warnings":  warnings,
            "n_features": int(X_train_new.shape[1]),
            "features":   list(X_train_new.columns),
        }))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[FE /formula] %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Formula feature failed: {exc}")


@router.post("/select", summary="Feature selection fitted on X_train only")
async def select_features(body: SelectRequest):
    """
    Feature selection using SelectKBest.
    CRITICAL: fitted exclusively on (X_train, y_train) — X_test is never seen.
    The selected feature mask is applied to both sets identically.
    """
    try:
        X_train, X_test, y_train, y_test = _get_split()
        task = store.get("task_type") or "classification"
        num_cols = _numeric_cols(X_train)

        if not num_cols:
            raise HTTPException(status_code=400, detail="No numeric features available for selection.")

        k = min(body.k, len(num_cols))
        y_enc = _encode_y(y_train)

        if len(y_enc) == 0:
            raise HTTPException(status_code=400, detail="No target labels available.")

        def _do_select():
            score_fn = f_classif if task == "classification" else f_regression
            selector = SelectKBest(score_fn, k=k)
            # FIT ONLY ON TRAIN
            X_num_train = X_train[num_cols].fillna(0).values
            selector.fit(X_num_train, y_enc)
            mask = selector.get_support()
            selected_num = [num_cols[i] for i, m in enumerate(mask) if m]
            scores       = {num_cols[i]: round(float(selector.scores_[i]), 4) for i in range(len(num_cols))}
            # Non-numeric cols are always kept
            non_num = [c for c in X_train.columns if c not in num_cols]
            selected_all = non_num + selected_num
            return selected_all, selected_num, scores

        _save_checkpoint()
        selected_all, selected_num, scores = await run_in_threadpool(_do_select)

        # Apply selection to both sets
        X_train_sel = X_train[selected_all]
        X_test_sel  = X_test[[c for c in selected_all if c in X_test.columns]]

        store.update({
            "X_train":         X_train_sel,
            "X_test":          X_test_sel,
            "feature_columns": selected_all,
        })

        log = store.get("fe_action_log") if store.get("fe_action_log") is not None else []
        log.append({"type": "selection", "method": body.method, "k": k, "selected": selected_all})
        store.set("fe_action_log", log)

        dropped = [c for c in num_cols if c not in selected_num]

        return JSONResponse(content=safe_json({
            "message":       f"Selected {len(selected_all)} features (SelectKBest, k={k}, fitted on train only).",
            "selected":      selected_all,
            "dropped":       dropped,
            "scores":        scores,
            "n_features":    int(X_train_sel.shape[1]),
        }))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[FE /select] %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feature selection failed: {exc}")


@router.get("/correlation", summary="Correlation matrix on X_train only")
async def correlation():
    """Compute correlation matrix on X_train, return high-correlation pairs."""
    try:
        X_train, _, _, _ = _get_split()
        num_cols = _numeric_cols(X_train)

        if len(num_cols) < 2:
            return JSONResponse(content={"pairs": [], "matrix": {}, "columns": []})

        corr_mat = X_train[num_cols].corr().round(4)

        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                val = float(corr_mat.iloc[i, j])
                if abs(val) > 0.90:
                    pairs.append({
                        "col_a": num_cols[i],
                        "col_b": num_cols[j],
                        "corr":  round(val, 4),
                        "severe": abs(val) > 0.95,
                    })

        return JSONResponse(content=safe_json({
            "columns": num_cols,
            "matrix":  corr_mat.to_dict(),
            "pairs":   sorted(pairs, key=lambda x: -abs(x["corr"])),
        }))

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Correlation failed: {exc}")


@router.get("/preview", summary="Preview current X_train state")
async def preview():
    """Return head of X_train + feature list + action log."""
    X_train = store.get("X_train")
    X_test  = store.get("X_test")
    if X_train is None:
        raise HTTPException(status_code=400, detail="No split found. Complete Split Data first.")

    fe_log = store.get("fe_action_log") or []
    head = X_train.head(5).fillna("").astype(str).to_dict(orient="records")

    return JSONResponse(content=safe_json({
        "train_rows":  int(X_train.shape[0]),
        "test_rows":   int(X_test.shape[0]) if X_test is not None else 0,
        "n_features":  int(X_train.shape[1]),
        "features":    list(X_train.columns),
        "head":        head,
        "action_log":  fe_log,
    }))


@router.post("/undo", summary="Undo last feature engineering action")
async def undo():
    """Restore X_train/X_test from the last saved checkpoint."""
    cp_train = store.get("fe_checkpoint_train")
    cp_test  = store.get("fe_checkpoint_test")

    if cp_train is None:
        raise HTTPException(status_code=400, detail="No checkpoint available to undo.")

    store.update({"X_train": cp_train, "X_test": cp_test})
    # Pop last action from log
    log = store.get("fe_action_log") or []
    if log:
        log.pop()
    store.set("fe_action_log", log)

    return JSONResponse(content=safe_json({
        "message":    "Last action undone.",
        "n_features": int(cp_train.shape[1]),
        "features":   list(cp_train.columns),
        "actions_remaining": len(log),
    }))


@router.post("/reset", summary="Reset to original post-split state")
async def reset():
    """
    Clear all feature engineering actions and restore original split columns.
    Does NOT re-split — just removes engineered features.
    """
    # The original split stores ALL columns from processed_df.
    # We restore from the raw split stored by the split step.
    # Since we don't store the "original" X_train separately, we rebuild
    # from processed_df using the original feature_columns saved at split time.
    df = store.get("processed_df")
    target = store.get("target_column")
    split_cfg = store.get("split_config")

    if df is None or split_cfg is None:
        raise HTTPException(status_code=400, detail="Cannot reset — no split configuration found. Re-run Split Data.")

    store.set("fe_action_log",       [])
    store.set("fe_checkpoint_train", None)
    store.set("fe_checkpoint_test",  None)
    store.set("engineered_specs",    [])

    # Re-derive feature columns from current processed_df
    feature_cols = [c for c in df.columns if c != target]
    store.set("feature_columns", feature_cols)

    # Guidance — user should re-run split after reset for clean state
    return JSONResponse(content={"message": "Feature engineering reset. Re-run the Split Data step to get a fresh split."})
