"""
training.py route  —  /api/train

Performance fix: all blocking ML calls are wrapped in run_in_threadpool
so they execute in a thread pool and never freeze the FastAPI event loop.
"""
import hashlib
import logging
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.models.schemas import ImbalanceAnalysisRequest, MultiTrainRequest, TrainRequest
from backend.services.session_store import store
from backend.utils.helpers import safe_json
from ml_pipeline.training import analyze_training_imbalance, get_available_models, train_model
from ml_pipeline.model_recommender import recommend_models

router = APIRouter(prefix="/api/train", tags=["Training"])
logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────

def _get_train_context():
    """
    Validate session state needed before training.
    Returns (X_train, X_test, y_train, y_test, task, numeric_feats).
    Pre-split data is read from session (set by /api/split).
    """
    from backend.utils.helpers import infer_task_type

    X_train = store.get("X_train")
    X_test  = store.get("X_test")
    y_train = store.get("y_train")
    y_test  = store.get("y_test")

    # Fall back to full df path for backward compat if split not done yet
    if X_train is None or X_test is None or y_train is None or y_test is None:
        raise HTTPException(
            status_code=400,
            detail="No train/test split found. Complete the 'Split Data' step before training.",
        )

    target = store.get("target_column")
    if not target:
        raise HTTPException(status_code=400, detail="Target column not configured. Go to Data Upload.")

    df = store.get("processed_df")
    if df is None:
        df = store.get("raw_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    task = store.get("task_type")
    if not task:
        task = infer_task_type(df[target])

    # Feature columns — use stored list or derive from X_train columns
    feats = store.get("feature_columns")
    if feats and isinstance(feats, list):
        numeric_feats = [c for c in feats if c in X_train.columns] if hasattr(X_train, 'columns') else feats
    else:
        if hasattr(X_train, 'columns'):
            numeric_feats = list(X_train.columns)
        else:
            numeric_feats = [f"feature_{i}" for i in range(X_train.shape[1])]

    if not numeric_feats:
        raise HTTPException(status_code=400, detail="No feature columns found. Run preprocessing first.")

    return X_train, X_test, y_train, y_test, task, numeric_feats


# ── GET /api/train/models ─────────────────────────────────────────────

@router.get("/models", summary="Get available models for current task type")
async def available_models():
    task = store.get("task_type")
    if not task:
        raise HTTPException(status_code=400, detail="Task type not set. Set target column first.")
    return JSONResponse(content=safe_json({"task_type": task, "models": get_available_models(task)}))


# ── GET /api/train/recommendations ────────────────────────────────────────

@router.get("/recommendations", summary="Get intelligent model recommendations based on dataset context")
async def model_recommendations():
    """
    Reads dataset metadata, EDA v2 quality stats, and imbalance info from
    session to produce a prioritised list of model recommendations.
    Never triggers training.
    """
    task = store.get("task_type")
    if not task:
        raise HTTPException(status_code=400, detail="Task type not set. Set target column first.")

    df = store.get("processed_df")
    if df is None:
        df = store.get("raw_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")

    target = store.get("target_column") or ""
    feat_cols = [c for c in df.columns if c != target]

    n_rows      = int(len(df))
    n_features  = int(len(feat_cols))
    n_numeric   = int(df[feat_cols].select_dtypes(include=["number"]).shape[1])
    n_categorical = int(n_features - n_numeric)

    # ── Imbalance ratio from session or fallback computation ───────
    imbalance_ratio: float = 1.0
    try:
        if task == "classification" and target and target in df.columns:
            vc = df[target].value_counts()
            if len(vc) >= 2:
                imbalance_ratio = round(float(vc.iloc[-1]) / float(vc.iloc[0]), 4)
    except Exception:
        pass

    # ── EDA v2 quality signals ───────────────────────────────
    has_high_outliers:    bool = False
    has_skewed_target:    bool = False
    has_high_cardinality: bool = False
    try:
        from ml_pipeline.eda import compute_data_quality, compute_feature_diagnostics
        quality     = compute_data_quality(df, target)
        diagnostics = compute_feature_diagnostics(df, target)
        has_high_outliers    = any(c["outlier_pct"] > 5 for c in quality["outliers"]["columns"])
        has_high_cardinality = any("High Cardinality" in d["flags"] for d in diagnostics)
        if task == "regression" and target in df.columns:
            tgt_ser = df[target].dropna()
            has_skewed_target = bool(abs(float(tgt_ser.skew())) > 2) if len(tgt_ser) >= 3 else False
    except Exception:
        pass  # non-fatal — fall back to defaults

    all_model_info = get_available_models(task)
    result = recommend_models(
        task_type            = task,
        n_rows               = n_rows,
        n_features           = n_features,
        n_numeric            = n_numeric,
        n_categorical        = n_categorical,
        imbalance_ratio      = imbalance_ratio,
        has_high_outliers    = has_high_outliers,
        has_skewed_target    = has_skewed_target,
        has_high_cardinality = has_high_cardinality,
        model_info           = all_model_info,
        all_model_names      = list(all_model_info.keys()),
    )

    # Inject task_type and dataset stats for the frontend context panel
    result["task_type"]    = task
    result["dataset_size"] = n_rows
    result["imbalance_ratio"] = imbalance_ratio

    return JSONResponse(content=safe_json(result))


# ── POST /api/train/multi ────────────────────────────────────────────

@router.post("/multi", summary="Train multiple models and return a comparison")
async def train_multi(body: MultiTrainRequest):
    """
    Trains each model in body.model_names using an IDENTICAL train-test split
    (same random_state + test_size).  Balancing is applied on X_train ONLY.
    Returns per-model metrics plus the best model selection.

    Best model:
      - classification → highest weighted F1-score
      - regression     → lowest RMSE

    The best model (fitted pipeline) and its test arrays are stored in the
    session so downstream Evaluation / Bias / Prediction routes work unchanged.
    """
    import traceback as _tb
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    )
    import numpy as _np

    if not body.model_names:
        raise HTTPException(status_code=400, detail="model_names must not be empty.")

    try:
        X_train, X_test, y_train, y_test, task, numeric_feats = _get_train_context()
    except HTTPException:
        raise

    session_config = store.get("balancing_config")
    cat_indices: list = []
    if session_config and session_config.get("enabled"):
        effective_technique = session_config.get("strategy") or "none"
        cat_indices = session_config.get("cat_indices") or []
    else:
        effective_technique = body.balancing_technique or "none"

    n_rows = (len(X_train) + len(X_test)) if hasattr(X_train, '__len__') else 0
    ts_now = datetime.now(timezone.utc)
    dataset_hash = "unknown"
    try:
        df = store.get("processed_df")
        if df is None:
            df = store.get("raw_df")
        if df is not None:
            csv_bytes    = df.to_csv(index=False).encode("utf-8", errors="replace")
            dataset_hash = hashlib.md5(csv_bytes).hexdigest()[:16]
    except Exception:
        pass

    # ── Before-balancing class distribution (from session y_train) ────
    before_class_dist: dict = {}
    if task == "classification":
        try:
            import numpy as _np_pre
            _vals, _cnts = _np_pre.unique(
                np.array(y_train).astype(str), return_counts=True
            )
            before_class_dist = {str(v): int(c) for v, c in zip(_vals, _cnts)}
        except Exception as _be:
            logger.warning("[MultiTrain] before_class_dist failed: %s", _be)

    results      = []
    best_model   = None
    best_name    = ""
    best_score   = None   # higher-is-better for F1; lower-is-better negated for RMSE
    best_artefacts = {}   # stores X_train, X_test, y_train, y_test, y_pred, y_prob

    for model_name in body.model_names:
        logger.info("[MultiTrain] Training '%s' — task=%s  balancing=%s",
                    model_name, task, effective_technique)
        try:
            mdl, X_tr, X_te, y_tr, y_te, y_pd, y_pb, target_le = await run_in_threadpool(
                train_model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name=model_name,
                task_type=task,
                hyperparams={},
                balancing_technique=effective_technique,
                cat_indices=cat_indices,
                use_calibration=body.use_calibration,
                scaler_type=body.scaler,
                apply_skewness=body.apply_skewness,
            )
        except Exception as exc:
            logger.warning("[MultiTrain] '%s' failed: %s", model_name, exc)
            results.append({"name": model_name, "status": "failed", "error": str(exc), "metrics": {}})
            continue

        # ── Compute metrics ─────────────────────────────────────────────
        metrics: dict = {}
        score_for_best: float

        if task == "classification":
            try:
                avg = "binary" if len(_np.unique(y_te)) == 2 else "weighted"
                acc = float(accuracy_score(y_te, y_pd))
                prec = float(precision_score(y_te, y_pd, average=avg, zero_division=0))
                rec  = float(recall_score(y_te, y_pd, average=avg, zero_division=0))
                f1   = float(f1_score(y_te, y_pd, average=avg, zero_division=0))
                auc  = None
                if y_pb is not None:
                    try:
                        auc = float(roc_auc_score(
                            y_te, y_pb,
                            multi_class="ovr" if len(_np.unique(y_te)) > 2 else "raise",
                        ))
                    except Exception:
                        auc = None
                pred_vals, pred_cnts = _np.unique(y_pd, return_counts=True)
                pred_dist = {str(v): int(c) for v, c in zip(pred_vals, pred_cnts)}

                # Detect degenerate model (all one class)
                degenerate = len(pred_vals) <= 1

                metrics = {
                    "accuracy":  round(acc, 4),
                    "precision": round(prec, 4),
                    "recall":    round(rec, 4),
                    "f1":        round(f1, 4),
                    "roc_auc":   round(auc, 4) if auc is not None else None,
                }
                score_for_best = f1

                # After-balancing distribution (from the actual y_train used to fit)
                after_class_dist: dict = {}
                try:
                    _av, _ac = _np.unique(y_tr.astype(str), return_counts=True)
                    after_class_dist = {str(v): int(c) for v, c in zip(_av, _ac)}
                except Exception:
                    pass

                result_entry = {
                    "name":              model_name,
                    "status":            "success",
                    "metrics":           metrics,
                    "pred_distribution": pred_dist,
                    "train_samples":     int(X_tr.shape[0]),
                    "test_samples":      int(X_te.shape[0]),
                    "degenerate":        degenerate,
                    "before_class_dist": before_class_dist,
                    "after_class_dist":  after_class_dist,
                    "warning":           (
                        "Model is predicting only one class. Accuracy may be misleading."
                        if degenerate else None
                    ),
                }
            except Exception as me:
                logger.warning("[MultiTrain] Metrics failed for '%s': %s", model_name, me)
                result_entry = {"name": model_name, "status": "metrics_error", "metrics": {}, "error": str(me)}
                score_for_best = -1.0
        else:
            try:
                rmse = float(_np.sqrt(mean_squared_error(y_te, y_pd)))
                mae  = float(mean_absolute_error(y_te, y_pd))
                r2   = float(r2_score(y_te, y_pd))
                metrics = {
                    "rmse": round(rmse, 4),
                    "mae":  round(mae, 4),
                    "r2":   round(r2, 4),
                }
                score_for_best = -rmse   # negate: lower RMSE = higher score
                result_entry = {
                    "name":          model_name,
                    "status":        "success",
                    "metrics":       metrics,
                    "train_samples": int(X_tr.shape[0]),
                    "test_samples":  int(X_te.shape[0]),
                    "degenerate":    False,
                    "warning":       None,
                }
            except Exception as me:
                logger.warning("[MultiTrain] Metrics failed for '%s': %s", model_name, me)
                result_entry = {"name": model_name, "status": "metrics_error", "metrics": {}, "error": str(me)}
                score_for_best = float("-inf")

        results.append(result_entry)

        # Track best model
        if best_score is None or score_for_best > best_score:
            best_score      = score_for_best
            best_name       = model_name
            best_model      = mdl
            best_target_le  = target_le
            best_artefacts  = {
                "X_train": X_tr, "X_test": X_te,
                "y_train": y_tr, "y_test": y_te,
                "y_pred":  y_pd, "y_prob": y_pb,
            }

    # ── Persist best model to session ────────────────────────────────────
    best_target_le = locals().get('best_target_le', None)  # safe fallback
    if best_model is not None:
        calibrated = type(best_model).__name__ == "CalibratedClassifierCV"
        ts_str     = ts_now.strftime("%Y%m%d_%H%M%S")
        ts_iso     = ts_now.isoformat()
        model_slug = best_name.lower().replace(" ", "_")
        model_id   = f"{model_slug}_{ts_str}"

        store.update({
            "model"                : best_model,
            "post_split_pipeline"  : best_model,      # full fitted pipeline
            "model_name"           : best_name,
            "X_train"              : best_artefacts["X_train"],
            "X_test"               : best_artefacts["X_test"],
            "y_train"              : best_artefacts["y_train"],
            "y_test"               : best_artefacts["y_test"],
            "y_pred"               : best_artefacts["y_pred"],
            "y_prob"               : best_artefacts["y_prob"],
            "applied_threshold"    : 0.5,
            "feature_columns"      : numeric_feats,
            "balancing_used"       : effective_technique,
            "model_id"             : model_id,
            "dataset_hash"         : dataset_hash,
            "training_timestamp"   : ts_iso,
            "calibrated"           : False,
            "target_label_encoder" : best_target_le,  # None if target was already numeric
        })
        await run_in_threadpool(store.save_model)
        await run_in_threadpool(store.save_test_data)

    # Best model metric for response
    best_result = next((r for r in results if r["name"] == best_name and r.get("status") == "success"), {})
    best_metric_label = "f1" if task == "classification" else "rmse"
    best_metric_value = best_result.get("metrics", {}).get(best_metric_label)

    return JSONResponse(content=safe_json({
        "models":      results,
        "best_model":  best_name,
        "best_metric": {"name": best_metric_label, "value": best_metric_value},
        "selection_criterion": "Highest F1-score" if task == "classification" else "Lowest RMSE",
        "dataset_info": {
            "train_size":          int(best_artefacts["X_train"].shape[0]) if best_artefacts else 0,
            "test_size":           int(best_artefacts["X_test"].shape[0])  if best_artefacts else 0,
            "total_rows":          n_rows,
            "dataset_hash":        dataset_hash,
            "balancing_used":      effective_technique,
            "balancing_applied":   effective_technique not in ("none", None, ""),
            "before_class_dist":   before_class_dist,
            "after_class_dist":    (
                # Use the first successful model's after_class_dist
                next((r.get("after_class_dist", {}) for r in results if r.get("status") == "success"), {})
            ),
        },
        "task_type":   task,
        "trained_at":  ts_now.isoformat(),
    }))


# ── POST /api/train/imbalance-analysis ───────────────────────────────

@router.post("/imbalance-analysis", summary="Analyse class imbalance on training split")
async def imbalance_analysis(body: ImbalanceAnalysisRequest):
    """
    Performs a DRY-RUN train-test split and analyses the CLASS DISTRIBUTION
    of y_train only. Uses processed_df so engineered features are included
    and feature_cols always aligns with the DataFrame columns.
    """
    try:
        df, target, task, numeric_feats = _get_train_context()
        # ── Use processed_df for analysis — raw_df may lack engineered cols ──
        # numeric_feats is already filtered to cols present in processed_df.

        if task != "classification":
            return JSONResponse(content=safe_json({
                "status"     : "N/A",
                "message"    : "Imbalance analysis only applies to classification tasks.",
                "is_balanced": True,
            }))

        result = await run_in_threadpool(
            analyze_training_imbalance,
            df=df,                       # processed_df — has all engineered cols
            target_col=target,
            feature_cols=numeric_feats,  # already filtered to cols in df
            test_size=body.test_size,
            random_state=body.random_state,
            eda_minority_pct=body.eda_minority_pct,
        )
        return JSONResponse(content=safe_json(result))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Training /imbalance-analysis] error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Imbalance analysis failed: {str(exc)}")


# ── POST /api/train ───────────────────────────────────────────────────

@router.post("", summary="Train a model")
async def train(body: TrainRequest):
    """
    Trains the selected model in a thread pool.

    Balancing strategy priority:
      1. Session balancing_config (confirmed via /api/imbalance/confirm) — uses
         the full config including cat_indices for SMOTENC.
      2. body.balancing_technique (HTTP request body override).
      3. None (no balancing).

    Resampling is applied to the TRAINING split ONLY inside train_model().
    The test split is NEVER modified.
    """
    try:
        df, target, task, numeric_feats = _get_train_context()

        # ── Resolve balancing strategy ─────────────────────────────────
        # Prefer the confirmed session config (set by ClassImbalance step).
        # Falls back to the request body if no session config exists.
        session_config = store.get("balancing_config")
        cat_indices: list = []

        if session_config and session_config.get("enabled"):
            effective_technique = session_config.get("strategy") or "none"
            cat_indices = session_config.get("cat_indices") or []
            logger.info(
                "[Training] Using confirmed session balancing: strategy=%s  "
                "use_smotenc=%s  cat_indices=%s",
                effective_technique,
                session_config.get("use_smotenc"),
                cat_indices[:5] if cat_indices else [],
            )
        else:
            effective_technique = body.balancing_technique or "none"
            logger.info(
                "[Training] No confirmed session config — using request body: balancing=%s",
                effective_technique,
            )

        n_rows = len(df)
        logger.info(
            "[Training] Starting '%s' on %d rows, task=%s, effective_balancing=%s",
            body.model_name, n_rows, task, effective_technique,
        )

        # ── Run blocking model training in a thread pool ───────────────
        model, X_train, X_test, y_train, y_test, y_pred, y_prob = await run_in_threadpool(
            train_model,
            df=df,
            target_col=target,
            feature_cols=numeric_feats,
            model_name=body.model_name,
            task_type=task,
            test_size=body.test_size,
            random_state=body.random_state,
            hyperparams=body.hyperparams,
            balancing_technique=effective_technique,
            cat_indices=cat_indices,
            use_calibration=body.use_calibration,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Training /train] error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Training failed: {str(exc)}")

    # ── Versioning & Reproducibility metadata ───────────────────────────────
    ts_now     = datetime.now(timezone.utc)
    ts_str     = ts_now.strftime("%Y%m%d_%H%M%S")
    ts_iso     = ts_now.isoformat()
    model_slug = body.model_name.lower().replace(" ", "_")
    model_id   = f"{model_slug}_{ts_str}"

    # Dataset hash: MD5 of the CSV representation of the raw/processed df
    try:
        csv_bytes    = df.to_csv(index=False).encode("utf-8", errors="replace")
        dataset_hash = hashlib.md5(csv_bytes).hexdigest()[:16]
    except Exception:
        dataset_hash = "unknown"

    # Check whether calibration was applied (detect CalibratedClassifierCV)
    calibrated = type(model).__name__ == "CalibratedClassifierCV"

    store.update({
        "model"             : model,
        "model_name"        : body.model_name,
        "X_train"           : X_train,
        "X_test"            : X_test,
        "y_train"           : y_train,
        "y_test"            : y_test,
        "y_pred"            : y_pred,
        "y_prob"            : y_prob,
        "applied_threshold" : 0.5,
        "feature_columns"   : numeric_feats,
        "balancing_used"    : effective_technique,
        # Versioning
        "model_id"          : model_id,
        "dataset_hash"      : dataset_hash,
        "training_timestamp": ts_iso,
        "calibrated"        : calibrated,
    })

    # Persist to disk in thread pool (I/O bound)
    await run_in_threadpool(store.save_model)
    await run_in_threadpool(store.save_test_data)

    logger.info(
        "[Training] '%s' trained. model_id=%s  dataset_hash=%s  "
        "calibrated=%s  balancing=%s  train=%d  test=%d",
        body.model_name, model_id, dataset_hash,
        calibrated, body.balancing_technique, X_train.shape[0], X_test.shape[0],
    )

    return JSONResponse(content=safe_json({
        "message"           : f"Model '{body.model_name}' trained successfully.",
        "model"             : body.model_name,
        "task_type"         : task,
        "balancing_used"    : effective_technique,
        "balancing_from_session": bool(session_config and session_config.get("enabled")),
        "calibrated"        : calibrated,
        "train_samples"     : int(X_train.shape[0]),
        "test_samples"      : int(X_test.shape[0]),
        "features_used"     : numeric_feats,
        "dataset_size"      : n_rows,
        # Versioning in response
        "model_id"          : model_id,
        "dataset_hash"      : dataset_hash,
        "training_timestamp": ts_iso,
    }))
