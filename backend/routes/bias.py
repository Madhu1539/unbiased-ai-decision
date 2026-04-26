"""
bias.py route  —  /api/bias

Key correctness fixes:
  - y_pred is NOW recomputed from y_prob at the requested threshold
    before bias analysis: y_pred = (y_prob >= threshold).astype(int)
    This ensures threshold consistency (req #7).
  - If y_prob is unavailable, stored y_pred is used with a warning.
  - debug mode passes through to bias_detection.py (req #8).
  - GET /api/bias/sanity-tests exposes the automated test suite.
  - DataFrame 'or' ambiguity fixed: uses 'is not None'.
"""
import logging
import traceback

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split
from starlette.concurrency import run_in_threadpool

from backend.models.schemas import BiasRequest
from backend.services.session_store import store
from backend.utils.helpers import safe_json
from ml_pipeline.bias_detection import (
    compute_per_group_metrics,
    check_attribute_leakage,
    full_bias_report,
    run_sanity_tests,
    validate_inputs,
    _multi_group_di_dp_scalars,
)
from ml_pipeline.bias_explanation import explain_bias
from ml_pipeline.sensitive_feature_detection import detect_sensitive_features
from backend.services.gemini_service import reset_gemini_session

router = APIRouter(prefix="/api/bias", tags=["Bias Detection"])
logger = logging.getLogger(__name__)


def _get_df():
    """
    Return the active DataFrame.
    MUST use 'is not None' — never 'or' — on a DataFrame.
    """
    df = store.processed_df if store.processed_df is not None else store.raw_df
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset loaded.")
    return df


# ── GET /api/bias/attributes ──────────────────────────────────────────

@router.get("/attributes", summary="Detect sensitive features for fairness analysis")
async def protected_attributes():
    """
    Run the Intelligent Sensitive Feature Detector on every non-target column.
    """
    try:
        df = _get_df()
    except HTTPException:
        raise

    target = store.target_column

    try:
        analysis = await run_in_threadpool(detect_sensitive_features, df, target)
    except Exception as exc:
        logger.error("[Bias /attributes] detection error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Feature detection failed: {str(exc)}")

    eligible = analysis["eligible_columns"]

    logger.info(
        "[Bias] /attributes — fairness_applicable=%s "
        "sensitive=%d potentially=%d non_sensitive=%d total=%d target=%s",
        analysis["fairness_applicable"],
        len(analysis["sensitive_columns"]),
        len(analysis["potentially_sensitive_columns"]),
        len(analysis["non_sensitive_columns"]),
        analysis["total_columns_analysed"],
        target,
    )

    return JSONResponse(content=safe_json({
        **analysis,
        "columns"             : eligible,
        "protected_attributes": eligible,
    }))


# ── POST /api/bias/analyze ────────────────────────────────────────────

def _build_y_pred_from_threshold(
    y_prob: np.ndarray | None,
    y_pred_stored: np.ndarray | None,
    threshold: float,
    task_type: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns (y_pred_binary, warnings).

    Priority:
    1. If task=classification and y_prob available → recompute y_pred at threshold
    2. Else use stored y_pred and warn
    """
    warnings: list[str] = []

    if task_type == "classification" and y_prob is not None and len(y_prob) > 0:
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        y_pred_bin = (y_prob_arr >= threshold).astype(int)
        warnings.append(
            f"y_pred recomputed from y_prob at threshold={threshold:.4f} "
            "to ensure threshold consistency."
        )
        logger.info(
            "[Bias] y_pred recomputed at threshold=%.4f  pos_rate=%.4f",
            threshold, float(y_pred_bin.mean()),
        )
        return y_pred_bin, warnings

    # Fallback: stored y_pred
    if y_pred_stored is None:
        raise HTTPException(
            status_code=400,
            detail="No predictions available. Train a model first.",
        )
    warnings.append(
        "y_prob not available — using stored y_pred (model.predict()). "
        "Bias metrics may not reflect the current threshold setting."
    )
    return np.asarray(y_pred_stored), warnings


def _check_threshold_mismatch(
    y_pred_stored: np.ndarray | None,
    y_pred_recomputed: np.ndarray,
    threshold: float,
) -> dict:
    """
    Req #1: Compare stored y_pred vs threshold-recomputed y_pred.
    Returns a dict with mismatch_rate, severity, warnings list.

    Severity:
      - mismatch_rate == 0        → 'none'
      - 0 < rate <= 0.05          → 'low'
      - rate > 0.05               → 'significant'
    """
    result = {
        "mismatch_rate"   : 0.0,
        "severity"        : "none",
        "warnings"        : [],
        "threshold_checked": threshold,
    }
    if y_pred_stored is None:
        return result

    stored_arr = np.asarray(y_pred_stored).ravel()
    new_arr    = np.asarray(y_pred_recomputed).ravel()
    n          = min(len(stored_arr), len(new_arr))
    if n == 0:
        return result

    mismatch_rate = float(np.mean(stored_arr[:n] != new_arr[:n]))
    result["mismatch_rate"] = round(mismatch_rate, 6)

    if mismatch_rate < 1e-6:
        result["severity"] = "none"
        return result

    if mismatch_rate > 0.05:
        severity = "significant"
        msg = (
            f"Significant threshold mismatch detected ({mismatch_rate:.2%}): "
            f"stored y_pred was computed at a different threshold than the requested {threshold:.4f}. "
            "Fairness metrics now use the recomputed y_pred — results may differ from the "
            "Evaluation page which displays stored metrics."
        )
    else:
        severity = "low"
        msg = (
            f"Minor threshold mismatch ({mismatch_rate:.2%}): "
            f"{mismatch_rate * 100:.1f}% of predictions differ between stored and recomputed at threshold={threshold:.4f}."
        )

    result["severity"] = severity
    result["warnings"].append(msg)
    logger.warning("[Bias] Threshold mismatch [%s]: %s", severity, msg)
    return result


def _run_bias_analysis(df, body, y_test, y_pred, feature_cols, target, task_type):
    """Synchronous bias computation — runs in a thread pool."""

    # ── Reproduce the SAME train/test split used during training ──────
    y_all    = df[target].to_numpy()
    indices  = np.arange(len(df))
    stratify = y_all if task_type == "classification" else None
    try:
        _, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=stratify,
        )
    except Exception:
        _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Align protected attribute to test indices
    protected_series = (
        df[body.protected_attribute]
        .reset_index(drop=True)
        .iloc[test_idx]
        .reset_index(drop=True)
    )

    # Align lengths
    n = min(len(y_test), len(protected_series))
    protected_series = protected_series.iloc[:n]
    y_test_aligned   = np.asarray(y_test)[:n]
    y_pred_aligned   = np.asarray(y_pred)[:n]

    report = full_bias_report(
        y_test=y_test_aligned,
        y_pred=y_pred_aligned,
        protected=protected_series,
        privileged_value=body.privileged_value,
        debug=body.debug,
    )
    return report, protected_series


@router.post("/analyze", summary="Run full bias analysis")
async def analyze_bias(body: BiasRequest):
    # FIX #5: Reset Gemini session guard — new analysis = allow fresh audit
    reset_gemini_session()

    try:
        df = _get_df()
    except HTTPException:
        raise

    model   = store.get("model")
    y_test  = store.get("y_test")
    y_pred  = store.get("y_pred")
    y_prob  = store.get("y_prob")

    if model is None or y_test is None:
        raise HTTPException(
            status_code=400,
            detail="Train a model before running bias analysis.",
        )

    if body.protected_attribute not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{body.protected_attribute}' not found in the dataset.",
        )

    task_type    = store.task_type
    feature_cols = store.get("feature_columns") or []
    target       = store.target_column
    threshold    = max(0.0, min(1.0, float(body.threshold)))

    # ── Recompute y_pred at requested threshold (threshold consistency) ──
    try:
        y_pred_for_bias, threshold_warns = _build_y_pred_from_threshold(
            y_prob, y_pred, threshold, task_type,
        )
    except HTTPException:
        raise

    try:
        result = await run_in_threadpool(
            _run_bias_analysis,
            df, body, y_test, y_pred_for_bias, feature_cols, target, task_type,
        )
        report, protected_series = result
    except Exception as exc:
        logger.error("[Bias /analyze] error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Bias analysis failed: {str(exc)}")

    # ── Inject threshold warnings into report meta ────────────────────
    if "meta" in report:
        report["meta"]["threshold_applied"] = threshold
        report["meta"]["threshold_warnings"] = threshold_warns

    # ── Threshold mismatch check (Req #1 enhanced) ────────────────────
    mismatch_result = _check_threshold_mismatch(y_pred, y_pred_for_bias, threshold)
    mismatch_warns  = mismatch_result["warnings"]

    # Req #8: debug threshold echo
    if body.debug:
        logger.info("[BiasDebug] Threshold used: %.4f", threshold)
    if "meta" in report:
        report["meta"]["threshold_used_debug"] = threshold   # Req #8

    # ── Per-group metrics breakdown (Req #4) ──────────────────────────
    try:
        per_group = await run_in_threadpool(
            compute_per_group_metrics,
            report.get("equalized_odds", {}).get("_y_test_aligned") or y_test,
            y_pred_for_bias,
            protected_series,
        )
    except Exception as exc:
        logger.warning("[Bias] compute_per_group_metrics failed: %s", exc)
        per_group = []

    # ── Protected attribute leakage check (Req #5) ────────────────────
    leakage = check_attribute_leakage(
        protected_attr=body.protected_attribute,
        feature_cols=feature_cols,
    )

    # ── Bias Explanation Engine ──────────────────────────────────
    try:
        bias_explanation = await run_in_threadpool(
            explain_bias,
            df,
            model,
            feature_cols,
            body.protected_attribute,
            y_pred_for_bias,
            per_group,
            5,                # top_n drivers
        )
    except Exception as exc:
        logger.warning("[Bias] explain_bias failed: %s", exc)
        bias_explanation = {
            "explanation_text": [
                "Bias explanation could not be generated — "
                "ensure the model and dataset are loaded."
            ],
            "bias_drivers"    : [],
            "group_disparity" : {},
        }

    # ── Multi-group scalar summary (Req #3, more than 2 groups) ──────
    multi_group_summary = None
    prot_series_temp    = protected_series
    try:
        from ml_pipeline.bias_detection import _bin_protected as _bp
        prot_binned, _ = _bp(prot_series_temp)
        if prot_binned.nunique() > 2:
            multi_group_summary = await run_in_threadpool(
                _multi_group_di_dp_scalars, y_pred_for_bias, protected_series
            )
    except Exception as exc:
        logger.debug("[Bias] multi-group scalar summary failed: %s", exc)

    # ── Promote global small-group warning (Req #5) ────────────────────
    global_small_group_warns: list[str] = []
    if any(g.get("small_group") for g in per_group):
        global_small_group_warns.append(
            "One or more groups have very small sample size — "
            "fairness metrics may be unreliable. "
            "Collect more data for representative fairness evaluation."
        )

    # ── Dynamic Fairness vs Accuracy Note (Req #6) ──────────────────
    # Compare current threshold accuracy vs baseline (threshold=0.5) accuracy
    # Use overall group accuracy from per_group for the comparison
    try:
        # baseline: what accuracy would y_prob >= 0.5 give?
        y_prob_arr = store.get("y_prob")
        _y_test_np = np.asarray(y_test).ravel()
        if y_prob_arr is not None and task_type == "classification":
            y_pred_baseline  = (np.asarray(y_prob_arr).ravel() >= 0.5).astype(int)
            n_match          = min(len(_y_test_np), len(y_pred_baseline))
            baseline_acc     = float(np.mean(
                # binarize y_test: already binary for classification
                (np.asarray(_y_test_np[:n_match]) > 0.5).astype(int) == y_pred_baseline[:n_match]
            ))
            current_acc_vals = [g["accuracy"] for g in per_group if g["accuracy"] is not None]
            current_acc      = float(np.mean(current_acc_vals)) if current_acc_vals else None

            if current_acc is not None and current_acc < baseline_acc - 0.02:
                fairness_accuracy_note = (
                    f"Improving fairness is reducing accuracy — evaluate trade-off. "
                    f"Current threshold ({threshold:.2f}) accuracy ≈ {current_acc:.3f} vs "
                    f"baseline (0.50) accuracy ≈ {baseline_acc:.3f}. "
                    "Consider whether the fairness gain justifies this accuracy cost."
                )
            else:
                _cur_acc_str = f"{current_acc:.3f}" if current_acc is not None else "N/A"
                fairness_accuracy_note = (
                    "Fairness adjustments are not significantly impacting accuracy. "
                    f"Current accuracy ≈ {_cur_acc_str}, "
                    f"baseline ≈ {baseline_acc:.3f}."
                )
        else:
            fairness_accuracy_note = (
                "Improving fairness may impact model performance. "
                "Fairness constraints often trade a small reduction in overall accuracy "
                "for more equitable outcomes across groups."
            )
    except Exception:
        fairness_accuracy_note = (
            "Improving fairness may impact model performance. "
            "Fairness constraints often trade a small reduction in overall accuracy "
            "for more equitable outcomes across groups."
        )

    # ── Aggregate all warnings (priority order) ───────────────────────
    all_warnings: list[str] = []
    # 1. Leakage first (highest severity)
    if leakage["leakage_detected"] and leakage["warning"]:
        all_warnings.append(leakage["warning"])
    # 2. Zero-rate warnings from DI (extreme bias per group) — Req #7
    di_report = report.get("disparate_impact", {})
    for w in di_report.get("warnings", []):
        if "zero" in w.lower() or "extreme bias" in w.lower():
            all_warnings.append(w)
    # 3. Threshold mismatch
    all_warnings += list(threshold_warns)
    all_warnings += list(mismatch_warns)
    # 4. Global small-group
    all_warnings += global_small_group_warns
    # 5. Per-group individual warnings (smaller detail)
    for g in per_group:
        if g.get("warning"):
            all_warnings.append(g["warning"])
    # De-duplicate while preserving priority order
    seen: set = set()
    deduped_warns: list[str] = []
    for w in all_warnings:
        if w not in seen:
            seen.add(w)
            deduped_warns.append(w)

    insights = _generate_bias_insights(report)
    verdict  = _compute_verdict(report)

    return JSONResponse(content=safe_json({
        # ── Structured keys per Req #10 ───────────────────────────────
        "threshold_used"          : threshold,       # Req #8 echo
        "fairness_metrics"        : {
            "disparate_impact"    : report.get("disparate_impact", {}),
            "demographic_parity"  : report.get("demographic_parity", {}),
            "equalized_odds"      : report.get("equalized_odds", {}),
            "group_accuracy"      : report.get("group_accuracy", {}),
        },
        "per_group_metrics"       : per_group,
        "warnings"                : deduped_warns,
        "leakage_check"           : leakage,
        "multi_group_summary"     : multi_group_summary,
        "threshold_mismatch"      : {               # Req #1 structured + Req #6 rounded
            "mismatch_rate" : round(mismatch_result["mismatch_rate"], 4),
            "severity"      : mismatch_result["severity"],
        },
        "fairness_accuracy_note"  : fairness_accuracy_note,
        "bias_explanation"        : bias_explanation,
        "insights"                : insights,
        "verdict"                 : verdict,
        # ── Legacy keys (backward compatibility) ─────────────────────
        "protected_attribute"     : body.protected_attribute,
        "threshold_applied"       : threshold,
        "threshold_warnings"      : threshold_warns,
        "debug_mode"              : body.debug,
        "report"                  : report,
    }))


# ── GET /api/bias/sanity-tests ────────────────────────────────────────

@router.get("/sanity-tests", summary="Run automated fairness metric sanity tests")
async def sanity_tests():
    """
    Run the automated sanity test suite to verify metric correctness.
    Tests:
      1. All-positive predictions → DI≈1, DP≈0, EO≈0
      2. Fully biased            → DI≈0, DP high, EO high
      3. Shuffled protected      → metrics improve
      4. Formula cross-check     → manual 6-sample verification
    """
    try:
        result = await run_in_threadpool(run_sanity_tests)
    except Exception as exc:
        logger.error("[Bias /sanity-tests] error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Sanity tests failed: {str(exc)}")

    return JSONResponse(content=safe_json(result))


# ── Insight generator ─────────────────────────────────────────────────

def _generate_bias_insights(report: dict) -> list:
    insights = []

    di = report.get("disparate_impact", {})

    # New schema: pairs-based DI
    if "pairs" in di:
        for pair_key, data in di.get("pairs", {}).items():
            di_val = data.get("disparate_impact")
            if di_val is None:
                continue
            if data.get("biased"):
                insights.append(
                    f"⚠️ Disparate Impact between '{data.get('group_0')}' and "
                    f"'{data.get('group_1')}' = {di_val:.3f} (< 0.80). "
                    "This violates the 80% rule. Consider resampling or re-weighting."
                )
            else:
                insights.append(
                    f"✅ Disparate Impact between '{data.get('group_0')}' and "
                    f"'{data.get('group_1')}' = {di_val:.3f} (acceptable)."
                )
    # Legacy schema fallback
    elif "groups" in di:
        for group, data in di["groups"].items():
            if data.get("biased"):
                insights.append(
                    f"⚠️ Group '{group}' DI = {data['disparate_impact']:.2f} (< 0.80). "
                    "Consider resampling or re-weighting training data."
                )

    dp  = report.get("demographic_parity", {})
    dpd = dp.get("demographic_parity_difference")
    if dpd is not None:
        insights.append(
            f"{'⚠️' if dpd > 0.1 else '✅'} Demographic Parity Difference = {dpd:.4f} "
            f"({'> 0.10 — biased' if dpd > 0.1 else 'acceptable'})."
        )

    eo     = report.get("equalized_odds", {})
    eo_val = eo.get("equalized_odds_difference")
    if eo_val is not None:
        insights.append(
            f"{'⚠️' if eo_val > 0.1 else '✅'} Equalized Odds (TPR diff) = {eo_val:.4f} "
            f"({'> 0.10 — biased' if eo_val > 0.1 else 'acceptable'})."
        )

    # Surface small-group warnings
    all_warns = (
        di.get("warnings", []) +
        dp.get("warnings", []) +
        eo.get("warnings", [])
    )
    for w in set(all_warns):
        if "small group" in w.lower() or "unreliable" in w.lower():
            insights.append(f"⚠️ {w}")

    return insights


# ── Verdict engine ────────────────────────────────────────────────────

def _compute_verdict(report: dict) -> dict:
    """
    Evaluate all three fairness metrics and return a structured verdict.

    Rules
    -----
    - Disparate Impact overall_di < 0.80       → FAIL
    - Demographic Parity Difference > 0.10     → FAIL
    - Equalized Odds difference > 0.10         → FAIL
    """
    failed: list = []
    passed: list = []

    # ── 1. Disparate Impact ───────────────────────────────────────────
    di     = report.get("disparate_impact", {})
    di_val = di.get("overall_di")
    if di_val is not None:
        if di_val < 0.80:
            failed.append({
                "metric"     : "Disparate Impact",
                "value"      : round(di_val, 3),
                "threshold"  : "≥ 0.80",
                "status"     : "fail",
                "description": (
                    f"Minimum DI ratio = {di_val:.3f} (< 0.80). "
                    "This violates the 80% rule (symmetric min/max formula)."
                ),
                "formula"    : "DI = min(rate_0, rate_1) / max(rate_0, rate_1)",
            })
        else:
            passed.append({
                "metric"     : "Disparate Impact",
                "value"      : round(di_val, 3),
                "threshold"  : "≥ 0.80",
                "status"     : "pass",
                "description": "All group pairs meet the 80% rule.",
            })

    # ── 2. Demographic Parity ─────────────────────────────────────────
    dp  = report.get("demographic_parity", {})
    dpd = dp.get("demographic_parity_difference")
    if dpd is not None:
        dpd_r = round(dpd, 3)
        entry = {
            "metric"    : "Demographic Parity",
            "value"     : dpd_r,
            "threshold" : "< 0.10",
            "status"    : "fail" if dpd > 0.10 else "pass",
            "formula"   : "DP = |rate_0 - rate_1|",
        }
        if dpd > 0.10:
            entry["description"] = (
                f"Positive prediction rates differ by {dpd_r} — model favours one group."
            )
            failed.append(entry)
        else:
            entry["description"] = "Positive prediction rates are similar across groups."
            passed.append(entry)

    # ── 3. Equalized Odds ─────────────────────────────────────────────
    eo      = report.get("equalized_odds", {})
    eo_diff = eo.get("equalized_odds_difference")
    if eo_diff is not None:
        eo_r  = round(eo_diff, 3)
        entry = {
            "metric"    : "Equalized Odds",
            "value"     : eo_r,
            "threshold" : "< 0.10 TPR variation",
            "status"    : "fail" if eo_diff > 0.10 else "pass",
            "formula"   : "EO = |TPR_0 - TPR_1|  where TPR = TP / (TP + FN)",
        }
        if eo_diff > 0.10:
            entry["description"] = (
                f"TPR varies by {eo_r} across groups — recall differs by demographic."
            )
            failed.append(entry)
        else:
            entry["description"] = "True positive rates are consistent across groups."
            passed.append(entry)

    # ── Verdict summary ───────────────────────────────────────────────
    is_biased = len(failed) > 0
    n_failed  = len(failed)
    n_total   = n_failed + len(passed)
    score     = round(len(passed) / n_total * 100) if n_total > 0 else 100

    if not is_biased:
        verdict     = "Model is Fair"
        severity    = "fair"
        explanation = "All fairness metrics are within acceptable thresholds across all groups."
    elif n_failed == 1:
        t           = failed[0]
        verdict     = "Model is Biased"
        severity    = "mild"
        explanation = f"{t['metric']} failed — {t['description']}"
    elif n_failed == 2:
        names       = " and ".join(t["metric"] for t in failed)
        verdict     = "Model is Biased"
        severity    = "moderate"
        explanation = f"{names} both exceed fairness thresholds."
    else:
        verdict     = "Model is Biased"
        severity    = "severe"
        explanation = "All three fairness metrics failed — systemic bias detected."

    return {
        "verdict"       : verdict,
        "is_biased"     : is_biased,
        "severity"      : severity,
        "explanation"   : explanation,
        "fairness_score": score,
        "failed_metrics": failed,
        "passed_metrics": passed,
        "total_checks"  : n_total,
    }
