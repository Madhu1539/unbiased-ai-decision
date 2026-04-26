"""
evaluation.py route  —  /api/evaluate

Correctness fixes:
  - GET /api/evaluate         now rebuilds metrics from y_prob + default threshold (0.5)
                              instead of using stale y_pred from training
  - GET /api/evaluate/metrics ?threshold=X&debug=Y  — live recomputation at any threshold
  - GET /api/evaluate/threshold   — optimal threshold sweep (unchanged)
  - POST /api/evaluate/threshold/apply — apply & persist a chosen threshold (unchanged)
"""
import logging
import traceback

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.services.session_store import store
from backend.utils.helpers import safe_json
from ml_pipeline.evaluation import (
    EvaluationPipeline,
    actual_vs_predicted,
    classification_metrics,
    compute_threshold_metrics,
    get_y_prob_from_model,
    regression_metrics,
    validate_data_integrity,
)
from ml_pipeline.threshold_optimizer import apply_threshold, find_optimal_threshold

router = APIRouter(prefix="/api/evaluate", tags=["Evaluation"])
logger = logging.getLogger(__name__)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ApplyThresholdRequest(BaseModel):
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    strategy: str = "auto"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe_cast_X(X_test) -> np.ndarray | None:
    """Cast X_test to float64 safely. Returns None on failure."""
    if X_test is None:
        return None
    try:
        if hasattr(X_test, 'dtype') and X_test.dtype == object:
            return (
                pd.DataFrame(X_test)
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
                .to_numpy(dtype=np.float64)
            )
        return np.asarray(X_test, dtype=np.float64)
    except Exception:
        return None


def _get_y_prob(model, X_test) -> np.ndarray | None:
    """
    Extract P(positive class) with full model-compatibility fallback:
      predict_proba() → decision_function() normalized to [0,1] → None
    """
    X_safe = _safe_cast_X(X_test)
    if X_safe is None:
        return None
    return get_y_prob_from_model(model, X_safe)


def _load_from_disk_if_needed():
    """Load model + test data from disk into the session store when not in memory."""
    model = store.get("model")
    if model is None:
        loaded = store.load_model()
        if loaded:
            store.load_test_data()
    return store.get("model")


def _ensure_y_prob(model, X_test, task: str) -> np.ndarray | None:
    """
    Return y_prob from the session store if available; otherwise recompute it.
    Raises ValueError if the model doesn't support predict_proba.
    """
    if task != "classification":
        return None

    # Prefer the stored y_prob (computed at training time)
    y_prob = store.get("y_prob")
    if y_prob is not None and len(y_prob) > 0:
        return np.asarray(y_prob, dtype=np.float64)

    # Fallback: compute live from model
    logger.info("[Evaluation] y_prob not in session — computing live from model")
    y_prob = _get_y_prob(model, X_test)
    if y_prob is not None:
        store.set("y_prob", y_prob)
    return y_prob


def _get_session_meta(require: bool = True) -> dict:
    """
    Collect versioning + reproducibility metadata from the session.

    Parameters
    ----------
    require : bool
        If True, raise HTTPException(422) when model_id or dataset_hash is
        missing.  Set False when you want a best-effort dict (e.g. fallback).
    """
    model_id   = store.get("model_id")
    dset_hash  = store.get("dataset_hash")
    ts         = store.get("training_timestamp")
    calibrated = store.get("calibrated") or False

    if require:
        missing = []
        if not model_id:
            missing.append("model_id")
        if not dset_hash:
            missing.append("dataset_hash")
        if missing:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Reproducibility fields missing from session: {missing}. "
                    "Please re-train the model to generate versioning metadata."
                ),
            )

    return {
        "model_id"           : model_id or "unknown",
        "dataset_hash"       : dset_hash or "unknown",
        "training_timestamp" : ts,
        "calibrated"         : calibrated,
    }


# ── GET /api/evaluate ─────────────────────────────────────────────────────────

@router.get("", summary="Get evaluation metrics (default threshold=0.5)")
async def evaluate():
    """
    Returns evaluation metrics computed from y_prob at threshold=0.5.
    Metrics are rebuilt live — never stale from training.
    """
    def _run():
        model  = _load_from_disk_if_needed()
        if model is None:
            raise HTTPException(
                status_code=404,
                detail="No trained model found. Train a model first.",
            )

        y_test = store.get("y_test")
        X_test = store.get("X_test")
        task   = store.get("task_type")

        if y_test is None:
            raise HTTPException(
                status_code=404,
                detail="Test data not found. Please train the model again.",
            )

        if task is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Task type (classification/regression) could not be determined. "
                    "This usually means the backend restarted and the old session files "
                    "are missing 'task_type'. Please retrain the model to fix this."
                ),
            )

        thresh = float(store.get("applied_threshold") or 0.5)

        if task == "classification":
            y_prob = _ensure_y_prob(model, X_test, task)

            if y_prob is not None:
                # ── Centralized pipeline: compute_threshold_metrics ───────────
                # Uses predict_proba output + single threshold consistently.
                metrics = compute_threshold_metrics(
                    y_test, y_prob, threshold=thresh
                )
            else:
                # Fallback: use stored y_pred (model.predict() only, no proba)
                y_pred = store.get("y_pred")
                if y_pred is None:
                    raise HTTPException(
                        status_code=404, detail="Predictions not available.",
                    )
                metrics = classification_metrics(
                    y_test, y_pred,
                    model=model,
                    X_test=_safe_cast_X(X_test),
                )
            avp = {}   # not used for classification plots
        else:
            y_pred = store.get("y_pred")
            if y_pred is None:
                raise HTTPException(status_code=404, detail="Predictions not available.")
            metrics = regression_metrics(y_test, y_pred)
            avp = actual_vs_predicted(y_test, y_pred)

        logger.info(
            "[Evaluation] threshold=%.4f  accuracy=%.4f  f1=%.4f  n=%d",
            thresh,
            metrics.get("accuracy", 0),
            metrics.get("f1_score", 0),
            len(y_test),
        )

        # Versioning metadata (best-effort — does not fail old sessions)
        meta = _get_session_meta(require=False)

        return {
            "task_type"           : task,
            "model"               : store.get("model_name"),
            "metrics"             : metrics,
            "actual_vs_predicted" : avp,          # ← regression scatter data
            "test_samples"        : int(len(y_test)),
            "applied_threshold"   : thresh,
            # Reproducibility
            "model_id"            : meta["model_id"],
            "dataset_hash"        : meta["dataset_hash"],
            "training_timestamp"  : meta["training_timestamp"],
            "calibrated"          : meta["calibrated"],
            # Warnings passthrough
            "imbalance_warnings"  : metrics.get("imbalance_warnings", []),
            "structural_warnings" : metrics.get("warnings", []),
        }

    try:
        result = await run_in_threadpool(_run)
    except HTTPException:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        # Always print to server stdout so it appears in uvicorn console
        print(f"[Evaluation] CRASH:\n{tb}", flush=True)
        logger.error("[Evaluation] Error:\n%s", tb)
        raise HTTPException(
            status_code=500,
            detail=f"Metric computation failed: {str(exc)}\n\nTraceback:\n{tb}",
        )

    return JSONResponse(content=safe_json(result))


# ── GET /api/evaluate/metrics ─────────────────────────────────────────────────

@router.get("/metrics", summary="Live metric recomputation at given threshold")
async def live_metrics(threshold: float = 0.5, debug: bool = False):
    """
    Recompute ALL classification metrics at the given threshold in real time.
    Does NOT persist the threshold — purely stateless.

    Query params:
      threshold  float   (0.0 → 1.0, default 0.5)
      debug      bool    (default False) — logs TP/FP/TN/FN + sample preds
    """
    threshold = max(0.0, min(1.0, float(threshold)))

    def _run():
        model  = _load_from_disk_if_needed()
        if model is None:
            raise HTTPException(status_code=404, detail="No trained model found.")

        y_test = store.get("y_test")
        X_test = store.get("X_test")
        task   = store.get("task_type")

        if task != "classification":
            raise HTTPException(
                status_code=400,
                detail="Live threshold metrics only available for classification.",
            )
        if y_test is None:
            raise HTTPException(status_code=404, detail="Test data not found.")

        y_prob = _ensure_y_prob(model, X_test, task)
        if y_prob is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Model does not support probability outputs (predict_proba). "
                    "Live threshold control requires a probabilistic model."
                ),
            )

        metrics = compute_threshold_metrics(
            y_test, y_prob, threshold=threshold, debug=debug
        )

        logger.info(
            "[LiveMetrics] threshold=%.4f  accuracy=%.4f  precision=%.4f  "
            "recall=%.4f  f1=%.4f  TP=%d  FP=%d  TN=%d  FN=%d",
            threshold,
            metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1_score"],
            metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"],
        )

        meta = _get_session_meta(require=False)

        return {
            "model"    : store.get("model_name"),
            "threshold": threshold,
            "metrics"  : metrics,
            # Reproducibility
            "model_id"           : meta["model_id"],
            "dataset_hash"       : meta["dataset_hash"],
            "training_timestamp" : meta["training_timestamp"],
            "calibrated"         : meta["calibrated"],
        }

    try:
        result = await run_in_threadpool(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[LiveMetrics] Error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Live metrics failed: {str(exc)}")

    return JSONResponse(content=safe_json(result))


# ── GET /api/evaluate/threshold ───────────────────────────────────────────────

def _run_threshold_opt(model, X_test, y_test, strategy: str) -> dict:
    """Synchronous; runs in threadpool."""
    y_prob = _get_y_prob(model, X_test)
    if y_prob is None:
        raise ValueError(
            "This model does not support probability outputs (predict_proba). "
            "Threshold optimisation requires probability estimates — "
            "use Logistic Regression, Random Forest, Gradient Boosting, or XGBoost."
        )
    return find_optimal_threshold(y_test, y_prob, strategy=strategy)


@router.get("/threshold", summary="Find optimal probability threshold")
async def get_optimal_threshold(strategy: str = "auto"):
    """Sweeps thresholds 0.01 → 0.99 and returns the optimal one."""
    def _run():
        model  = _load_from_disk_if_needed()
        if model is None:
            raise HTTPException(status_code=404, detail="No trained model found.")
        y_test = store.get("y_test")
        X_test = store.get("X_test")
        task   = store.get("task_type")
        if task != "classification":
            raise HTTPException(
                status_code=400,
                detail="Threshold optimisation only applies to binary classification tasks.",
            )
        if y_test is None:
            raise HTTPException(status_code=404, detail="Test data not found.")

        logger.info(
            "[Threshold] Optimising — strategy=%s  model=%s  n_test=%d",
            strategy, store.get("model_name"), len(y_test),
        )
        return _run_threshold_opt(model, X_test, y_test, strategy)

    try:
        result = await run_in_threadpool(_run)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("[Threshold] Failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Threshold optimisation failed: {str(exc)}")

    logger.info(
        "[Threshold] best=%.4f  strategy=%s  F1=%.4f  Recall=%.4f",
        result["best_threshold"], result["strategy_used"],
        result["f1_score"], result["recall"],
    )
    return JSONResponse(content=safe_json({"model": store.get("model_name"), **result}))


# ── POST /api/evaluate/threshold/apply ───────────────────────────────────────

@router.post("/threshold/apply", summary="Apply threshold and persist predictions")
async def apply_threshold_route(body: ApplyThresholdRequest):
    """
    Applies body.threshold, recomputes all metrics via the centralized pipeline,
    and persists the threshold in the session so that /evaluate reflects it.
    """
    thresh = float(body.threshold)

    def _run():
        model  = store.get("model")
        y_test = store.get("y_test")
        X_test = store.get("X_test")
        task   = store.get("task_type")

        if model is None:
            raise HTTPException(status_code=404, detail="No trained model found.")
        if task != "classification":
            raise HTTPException(
                status_code=400,
                detail="Threshold application only applies to classification tasks.",
            )
        if y_test is None:
            raise HTTPException(status_code=404, detail="Test data not available.")

        y_prob = _ensure_y_prob(model, X_test, task)
        if y_prob is None:
            raise ValueError("Model does not support predict_proba.")

        # Centralized pipeline at the requested threshold
        metrics = compute_threshold_metrics(y_test, y_prob, threshold=thresh)

        # Build decoded y_pred for session storage (bias detection still needs it)
        y_pred_bin = apply_threshold(y_prob, thresh)
        unique_labels = np.unique(y_test)
        if not set(unique_labels.tolist()).issubset({0, 1, 0.0, 1.0}):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(np.asarray(y_test).astype(str))
            y_pred_out = le.inverse_transform(y_pred_bin)
        else:
            y_pred_out = y_pred_bin.astype(y_test.dtype if hasattr(y_test, 'dtype') else int)

        avp = actual_vs_predicted(y_test, y_pred_out)
        return y_pred_out, metrics, avp

    try:
        y_pred_new, metrics, avp = await run_in_threadpool(_run)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("[Threshold/Apply] Failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Apply-threshold failed: {str(exc)}")

    # Persist
    store.set("y_pred", y_pred_new)
    store.set("applied_threshold", thresh)
    await run_in_threadpool(store.save_test_data)

    logger.info(
        "[Threshold/Apply] threshold=%.4f  accuracy=%.4f  f1=%.4f  TP=%d  FP=%d  TN=%d  FN=%d",
        thresh, metrics["accuracy"], metrics["f1_score"],
        metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"],
    )

    return JSONResponse(content=safe_json({
        "message"            : f"Threshold {thresh:.4f} applied successfully.",
        "applied_threshold"  : thresh,
        "model"              : store.get("model_name"),
        "metrics"            : metrics,
        "actual_vs_predicted": avp,
        "test_samples"       : int(len(store.get("y_test"))),
    }))


# ── GET /api/evaluate/probabilities ──────────────────────────────────────────

_FRONTEND_SAMPLE_LIMIT = 10_000   # max points sent to the frontend metric engine
_LARGE_DATASET_WARN    = 100_000  # warn user above this size


@router.get("/probabilities", summary="Get y_prob + y_test for frontend metric caching")
async def get_probabilities():
    """
    Returns predicted probabilities (y_prob) and encoded labels (y_test_enc)
    so the frontend can compute classification metrics locally for any threshold
    without making additional API calls.

    For datasets > 10,000 samples a stratified random sample is returned;
    the pre-computed ROC curve always uses the full dataset.
    """
    def _run():
        model  = _load_from_disk_if_needed()
        if model is None:
            raise HTTPException(status_code=404, detail="No trained model found. Train a model first.")

        y_test = store.get("y_test")
        X_test = store.get("X_test")
        task   = store.get("task_type")

        if task != "classification":
            raise HTTPException(
                status_code=400,
                detail="Probability caching is only available for classification tasks.",
            )
        if y_test is None:
            raise HTTPException(status_code=404, detail="Test data not found.")

        y_prob = _ensure_y_prob(model, X_test, task)
        if y_prob is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This model does not support probability outputs (no predict_proba or "
                    "decision_function). Frontend metric engine requires probability scores."
                ),
            )

        # ── Data integrity check ──────────────────────────────────────────────
        integrity = validate_data_integrity(y_test, y_prob)
        if not integrity["valid"]:
            raise HTTPException(
                status_code=422,
                detail=f"Data integrity errors: {'; '.join(integrity['errors'])}",
            )

        # ── Encode y_test to {0, 1} ───────────────────────────────────────────
        from ml_pipeline.evaluation import _encode_binary
        y_test_arr  = np.asarray(y_test)
        y_prob_arr  = np.asarray(y_prob, dtype=np.float64)
        y_pred_half = (y_prob_arr >= 0.5).astype(int)
        y_test_enc, _, le = _encode_binary(y_test_arr, y_pred_half)
        class_labels = le.classes_.tolist() if le is not None else ["0", "1"]

        n = len(y_prob_arr)
        is_large   = n > _LARGE_DATASET_WARN
        is_sampled = n > _FRONTEND_SAMPLE_LIMIT

        # ── Stratified sample for frontend (preserves class ratio) ────────────
        if is_sampled:
            rng    = np.random.RandomState(42)
            idx_pos = np.where(y_test_enc == 1)[0]
            idx_neg = np.where(y_test_enc == 0)[0]
            n_pos  = min(len(idx_pos), _FRONTEND_SAMPLE_LIMIT // 2)
            n_neg  = min(len(idx_neg), _FRONTEND_SAMPLE_LIMIT - n_pos)
            idx    = np.concatenate([
                rng.choice(idx_pos, n_pos, replace=False) if len(idx_pos) >= n_pos else idx_pos,
                rng.choice(idx_neg, n_neg, replace=False) if len(idx_neg) >= n_neg else idx_neg,
            ])
            y_prob_out     = y_prob_arr[idx]
            y_test_enc_out = y_test_enc[idx]
        else:
            y_prob_out     = y_prob_arr
            y_test_enc_out = y_test_enc

        # ── Pre-compute ROC on FULL data (not the sample) ─────────────────────
        roc_data: dict | None = None
        try:
            from sklearn.metrics import roc_auc_score, roc_curve as sk_roc
            auc      = float(roc_auc_score(y_test_enc, y_prob_arr))
            fpr, tpr, _ = sk_roc(y_test_enc, y_prob_arr)
            step     = max(1, len(fpr) // 150)
            roc_data = {
                "fpr"   : fpr[::step].tolist(),
                "tpr"   : tpr[::step].tolist(),
                "auc"   : round(auc, 6),
                "labels": class_labels,
            }
        except Exception as roc_err:
            logger.debug("[Probabilities] ROC skipped: %s", roc_err)

        logger.info(
            "[Probabilities] model=%s  n_total=%d  n_frontend=%d  sampled=%s  large=%s",
            store.get("model_name"), n, len(y_prob_out), is_sampled, is_large,
        )

        return {
            "y_prob"           : np.round(y_prob_out, 6).tolist(),
            "y_test_enc"       : y_test_enc_out.tolist(),
            "class_labels"     : class_labels,
            "n_samples"        : n,
            "n_frontend"       : int(len(y_prob_out)),
            "is_sampled"       : is_sampled,
            "is_large_dataset" : is_large,
            "roc_curve"        : roc_data,
            "integrity"        : integrity,
        }

    try:
        result = await run_in_threadpool(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Probabilities] Error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to fetch probabilities: {str(exc)}")

    return JSONResponse(content=safe_json(result))


# ── GET /api/evaluate/error-analysis ─────────────────────────────────────────

@router.get("/error-analysis", summary="Return misclassified samples with confidence + top features")
async def error_analysis(limit: int = 100):
    """
    Identify FP and FN samples from predicted probabilities at the session threshold.

    For each misclassified sample returns:
      - sample_id        : original row index
      - actual           : true label (decoded)
      - predicted        : predicted label (decoded)
      - type             : 'FP' or 'FN'
      - confidence       : predicted probability for the positive class
      - threshold        : threshold used
      - distance         : distance of confidence from threshold
      - top_features     : top-3 features by absolute importance (if available)

    Results sorted by descending confidence (highest-confidence wrong predictions first).
    """
    def _run():
        model  = _load_from_disk_if_needed()
        if model is None:
            raise HTTPException(status_code=404, detail="No trained model found. Train a model first.")

        y_test  = store.get("y_test")
        X_test  = store.get("X_test")
        task    = store.get("task_type")
        feat_cols = store.get("feature_columns")
        feat_cols = list(feat_cols) if feat_cols is not None else []

        if task != "classification":
            raise HTTPException(
                status_code=400,
                detail="Error analysis only applies to classification tasks.",
            )
        if y_test is None:
            raise HTTPException(status_code=404, detail="Test data not found.")

        thresh   = float(store.get("applied_threshold") or 0.5)
        y_prob   = _ensure_y_prob(model, X_test, task)
        if y_prob is None:
            raise HTTPException(
                status_code=400,
                detail="Model does not support predict_proba — error analysis requires probability outputs.",
            )

        # ── Encode labels to {0, 1} ───────────────────────────────────────
        from ml_pipeline.evaluation import _encode_binary
        y_test_arr = np.asarray(y_test)
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        y_pred_bin = (y_prob_arr >= thresh).astype(int)
        y_enc, y_pred_enc, le = _encode_binary(y_test_arr, y_pred_bin)
        class_labels = le.classes_.tolist() if le is not None else ["0", "1"]

        # ── Feature importance lookup (model-agnostic) ────────────────────
        global_importances: np.ndarray | None = None
        try:
            if hasattr(model, "feature_importances_"):
                global_importances = np.asarray(model.feature_importances_, dtype=np.float64)
            elif hasattr(model, "coef_"):
                coef = np.asarray(model.coef_)
                global_importances = np.abs(coef[0] if coef.ndim > 1 else coef)
            elif hasattr(model, "estimator") and hasattr(model.estimator, "coef_"):
                # CalibratedClassifierCV wraps the base estimator
                coef = np.asarray(model.estimator.coef_)
                global_importances = np.abs(coef[0] if coef.ndim > 1 else coef)
            elif hasattr(model, "base_estimator") and hasattr(model.base_estimator, "feature_importances_"):
                global_importances = np.asarray(model.base_estimator.feature_importances_)
        except Exception as fi_err:
            logger.debug("[ErrorAnalysis] Feature importance unavailable: %s", fi_err)

        # Normalise importances to 0-1 range for consistent display
        if global_importances is not None and len(global_importances) > 0:
            imp_max = global_importances.max()
            if imp_max > 0:
                global_importances = global_importances / imp_max

        def _top_features(sample_idx: int) -> list:
            """Return top-3 feature names + normalised importance scores."""
            if global_importances is None or not feat_cols:
                return []
            n = min(len(global_importances), len(feat_cols))
            scores = global_importances[:n]
            top_idx = np.argsort(scores)[::-1][:3]
            return [
                {"feature": feat_cols[i], "impact": round(float(scores[i]), 4)}
                for i in top_idx if i < len(feat_cols)
            ]

        # ── Decode helper ─────────────────────────────────────────────────
        def _decode(enc_val: int) -> str:
            if le is not None:
                try:
                    return str(le.inverse_transform([enc_val])[0])
                except Exception:
                    pass
            return str(enc_val)

        # ── Build error records ───────────────────────────────────────────
        errors = []
        for i in range(len(y_enc)):
            actual_enc   = int(y_enc[i])
            pred_enc     = int(y_pred_enc[i])
            if actual_enc == pred_enc:
                continue   # correctly classified — skip

            prob         = float(y_prob_arr[i])
            is_fp        = (pred_enc == 1 and actual_enc == 0)
            err_type     = "FP" if is_fp else "FN"

            errors.append({
                "sample_id"   : int(i),
                "actual"      : _decode(actual_enc),
                "predicted"   : _decode(pred_enc),
                "type"        : err_type,
                "confidence"  : round(prob, 4),
                "threshold"   : thresh,
                "distance"    : round(abs(prob - thresh), 4),
                "top_features": _top_features(i),
            })

        # Sort: highest-confidence wrong predictions first
        errors.sort(key=lambda r: r["confidence"], reverse=True)
        errors = errors[:limit]

        # ── Summary stats ─────────────────────────────────────────────────
        fp_count = sum(1 for e in errors if e["type"] == "FP")
        fn_count = sum(1 for e in errors if e["type"] == "FN")
        avg_conf = round(float(np.mean([e["confidence"] for e in errors])), 4) if errors else 0.0

        logger.info(
            "[ErrorAnalysis] threshold=%.3f  total_errors=%d  FP=%d  FN=%d  returning=%d",
            thresh, fp_count + fn_count, fp_count, fn_count, len(errors),
        )

        return {
            "error_analysis"  : errors,
            "summary": {
                "total_errors"     : fp_count + fn_count,
                "fp_count"         : fp_count,
                "fn_count"         : fn_count,
                "avg_confidence"   : avg_conf,
                "threshold_used"   : thresh,
                "class_labels"     : class_labels,
                "has_feature_info" : global_importances is not None,
                "returned"         : len(errors),
            },
        }

    try:
        result = await run_in_threadpool(_run)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[ErrorAnalysis] Failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analysis failed: {str(exc)}")

    return JSONResponse(content=safe_json(result))
