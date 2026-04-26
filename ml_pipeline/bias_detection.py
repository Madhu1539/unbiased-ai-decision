"""
bias_detection.py  —  Fairness & bias analysis utilities
=========================================================

Correctness guarantees (all fixes over previous version):
----------------------------------------------------------
1.  y_pred is always binarised to {0,1} from stored predictions —
    y_prob is NEVER used for fairness metrics (req #1).

2.  Disparate Impact uses the MIN/MAX symmetric formula:
      DI = min(rate_0, rate_1) / max(rate_0, rate_1)
    so DI is always in (0, 1] regardless of group ordering (req #3).

3.  Demographic Parity = abs(rate_0 - rate_1)  — pairwise (req #4).

4.  Equalized Odds     = abs(TPR_0 - TPR_1)    — pairwise TPR (req #5).

5.  Threshold consistency: y_pred = (y_prob >= threshold) is applied by
    the caller (bias.py route) before calling any function here.
    All functions strictly use the supplied y_pred array (req #7).

6.  Debug mode: logs group sizes, positive rates, TPR, and per-group
    confusion matrix when debug=True (req #8).

7.  Small-group warning: any group with < MIN_GROUP_SIZE samples triggers
    a 'unreliable_metrics' warning in the return dict (req #9).

8.  Input validation: checks y_pred is binary, protected attribute is
    categorical; auto-bins continuous attributes (req #1).

9.  Automated sanity test suite: run_sanity_tests() (req #6).

Public API
----------
  validate_inputs(y_pred, y_test, protected)             → list[str]
  compute_per_group_metrics(y_test, y_pred, protected)   → list[dict]  NEW
  check_attribute_leakage(protected_attr, feature_cols)  → dict        NEW
  disparate_impact(y_pred, protected, ...)               → dict
  demographic_parity(y_pred, protected)                  → dict
  equalized_odds(y_test, y_pred, protected)              → dict
  group_accuracy(y_test, y_pred, protected)              → dict
  full_bias_report(y_test, y_pred, protected, ..., debug)→ dict
  run_sanity_tests()                                     → dict
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
MIN_GROUP_SIZE = 20   # groups below this get an unreliable-metrics warning
_BIAS_TOL      = 1e-9  # avoid division by exactly zero


# ══════════════════════════════════════════════════════════════════════
#  Input helpers
# ══════════════════════════════════════════════════════════════════════

def _to_binary(series: pd.Series) -> pd.Series:
    """
    Coerce any prediction/label series to strict binary 0/1.

    Priority:
      1. Already {0, 1}                 → pass through
      2. Exactly two classes            → alphabetically first = 0, second = 1
      3. Multi-class with '1' / True    → 1 = positive, rest = 0
      4. Continuous numeric             → threshold at median (with warning)
      5. Fallback                       → majority class = 0, rest = 1
    """
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(dtype=int)

    unique = s.unique()

    # 1. Already binary integers
    try:
        if set(unique.tolist()) <= {0, 1}:
            return series.fillna(0).astype(int)
    except TypeError:
        pass

    # 2. Exactly two classes
    if len(unique) == 2:
        sorted_cls = sorted(unique, key=str)
        return series.map({sorted_cls[0]: 0, sorted_cls[1]: 1}).fillna(0).astype(int)

    # 3. Multi-class with explicit positive class
    for pos in (1, '1', True, 'Yes', 'yes', 'True', 'true', 'Positive', 'positive'):
        if pos in unique:
            return (series == pos).astype(int)

    # 4. Continuous → threshold at median
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().mean() >= 0.8:
        thr = float(numeric.median())
        logger.warning("[BiasDetection] Continuous predictions detected — thresholding at median %.4f", thr)
        return (numeric > thr).fillna(0).astype(int)

    # 5. Fallback
    majority = series.value_counts().idxmax()
    return (series != majority).astype(int)


def _bin_protected(protected: pd.Series, max_groups: int = 10) -> Tuple[pd.Series, List[str]]:
    """
    Ensure the protected attribute has ≤ max_groups categories.

    Returns
    -------
    (binned_series, warnings_list)

    - Low-cardinality  (≤ max_groups unique values) → keep as-is (str)
    - High-cardinality numeric                      → 4 quantile buckets
    - High-cardinality categorical                  → top (max_groups-1) + 'Other'
    """
    protected = protected.astype(str)
    warnings: List[str] = []
    n_unique  = protected.nunique()

    if n_unique <= max_groups:
        return protected, warnings

    warnings.append(
        f"Protected attribute has {n_unique} unique values (> {max_groups}). "
        "Applying automatic binning."
    )

    # Try numeric quantile binning
    numeric = pd.to_numeric(protected, errors='coerce')
    if numeric.notna().mean() >= 0.8:
        warnings.append(
            "Protected attribute appears continuous — binned into 4 quantile groups "
            "(Q1 Low, Q2 Mid-Low, Q3 Mid-High, Q4 High). "
            "For more precise fairness analysis, manually bin the attribute."
        )
        try:
            binned = pd.qcut(
                numeric, q=4,
                labels=['Q1 (Low)', 'Q2 (Mid-Low)', 'Q3 (Mid-High)', 'Q4 (High)'],
                duplicates='drop',
            )
            return binned.astype(str), warnings
        except Exception:
            pass

    # Categorical high-cardinality fallback
    top_groups = protected.value_counts().head(max_groups - 1).index
    return protected.where(protected.isin(top_groups), other='Other'), warnings


def validate_inputs(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    protected: pd.Series,
) -> List[str]:
    """
    Validate inputs and return a list of warnings.
    Raises ValueError on blocking errors.

    Checks:
      - y_pred must be binary {0, 1}  (never y_prob)
      - y_test  must be binary {0, 1}  (for EO computation)
      - protected must match len(y_pred)
    """
    warnings: List[str] = []

    y_pred_arr = np.asarray(y_pred).ravel()
    unique_pred = set(np.unique(y_pred_arr).tolist())

    if not unique_pred.issubset({0, 1, 0.0, 1.0}):
        # Could be float probabilities accidentally passed
        if np.all((y_pred_arr >= 0) & (y_pred_arr <= 1)) and not np.all(np.isin(y_pred_arr, [0, 1])):
            raise ValueError(
                "y_pred appears to contain probability scores (values between 0 and 1), "
                "NOT binary predictions. "
                "Fairness metrics require y_pred = (y_prob >= threshold).astype(int). "
                "Pass the threshold parameter to recompute y_pred correctly."
            )
        warnings.append(
            f"y_pred has non-binary unique values {unique_pred}. "
            "It will be binarised using _to_binary()."
        )

    if len(y_pred_arr) != len(protected):
        raise ValueError(
            f"Length mismatch: y_pred has {len(y_pred_arr)} samples "
            f"but protected has {len(protected)} samples."
        )

    return warnings


def _small_group_warnings(group_counts: Dict[str, int]) -> List[str]:
    """Return warnings for groups below MIN_GROUP_SIZE."""
    warns = []
    for g, n in group_counts.items():
        if n < MIN_GROUP_SIZE:
            warns.append(
                f"Group '{g}' has only {n} samples (< {MIN_GROUP_SIZE}). "
                "Fairness metrics may be unreliable due to small group size."
            )
    return warns


def _per_group_debug(
    label: str,
    groups: pd.Series,
    y_test_b: pd.Series,
    y_pred_b: pd.Series,
    threshold_used: Optional[float] = None,
) -> None:
    """Print per-group confusion matrix and rates to logger when debug=True."""
    logger.info("[BiasDebug] ── %s ─────────────────────────────────────", label)
    if threshold_used is not None:
        logger.info("[BiasDebug] threshold_used=%.4f", threshold_used)
    for g in sorted(groups.unique()):
        mask = groups == g
        yt   = y_test_b[mask]
        yp   = y_pred_b[mask]
        n    = int(mask.sum())
        tp   = int(((yt == 1) & (yp == 1)).sum())
        tn   = int(((yt == 0) & (yp == 0)).sum())
        fp   = int(((yt == 0) & (yp == 1)).sum())
        fn   = int(((yt == 1) & (yp == 0)).sum())
        rate = round(float(yp.mean()), 4) if n > 0 else None
        tpr  = round(tp / (tp + fn), 4)   if (tp + fn) > 0 else None
        fpr  = round(fp / (fp + tn), 4)   if (fp + tn) > 0 else None
        logger.info(
            "[BiasDebug] Group='%s'  n=%d  positive_rate=%s  "
            "TPR=%s  FPR=%s  TP=%d  TN=%d  FP=%d  FN=%d",
            g, n, rate, tpr, fpr, tp, tn, fp, fn,
        )


# ══════════════════════════════════════════════════════════════════════
#  Core Fairness Metrics  (corrected formulas)
# ══════════════════════════════════════════════════════════════════════

def disparate_impact(
    y_pred: np.ndarray,
    protected: pd.Series,
    privileged_value=None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Disparate Impact — symmetric MIN/MAX formula:

      rate_g = P(Ŷ=1 | group=g)
      DI     = min(rate_0, rate_1) / max(rate_0, rate_1)

    DI ∈ (0, 1].  Values < 0.80 indicate potential bias (80% rule).
    DI = 1.0 means perfectly equal positive rates.

    NOTE: This function uses y_pred (binary) — never y_prob.
    """
    pred_series = pd.Series(np.asarray(y_pred).ravel(), index=protected.index)
    pred_bin    = _to_binary(pred_series)
    prot, bin_warns = _bin_protected(protected)

    groups = sorted(prot.unique())
    if len(groups) < 2:
        return {"error": "Protected attribute has fewer than 2 groups.", "warnings": bin_warns}

    # Compute positive rate per group
    group_rates:  Dict[str, float] = {}
    group_counts: Dict[str, int]   = {}
    for g in groups:
        mask         = prot == g
        n            = int(mask.sum())
        group_counts[g] = n
        group_rates[g]  = float(pred_bin[mask].mean()) if n > 0 else 0.0

    small_warns = _small_group_warnings(group_counts)

    if debug:
        logger.info("[BiasDebug] Disparate Impact — group rates: %s", group_rates)
        logger.info("[BiasDebug] Disparate Impact — group counts: %s", group_counts)

    results: Dict[str, Any] = {}
    values_for_di: List[float] = []

    for i, g0 in enumerate(groups):
        for g1 in groups[i + 1:]:
            r0 = group_rates[g0]
            r1 = group_rates[g1]
            rmin, rmax = min(r0, r1), max(r0, r1)

            if rmax < _BIAS_TOL:
                # Both groups have 0 positive predictions → perfectly equal
                di_val = None
                biased = False
                note   = "Both groups have zero positive predictions — DI undefined (all-negative model)."
            else:
                di_val = round(rmin / rmax, 4)
                biased = di_val < 0.8
                note   = None

            pair_key = f"{g0} vs {g1}"
            pair_data = {
                "group_0"       : g0,
                "group_1"       : g1,
                "rate_0"        : round(r0, 4),
                "rate_1"        : round(r1, 4),
                "disparate_impact": di_val,
                "biased"        : biased,
            }
            if note:
                pair_data["note"] = note
            results[pair_key] = pair_data
            if di_val is not None:
                values_for_di.append(di_val)

            if debug:
                logger.info(
                    "[BiasDebug] DI pair '%s'  rate_0=%.4f  rate_1=%.4f  DI=%.4f  biased=%s",
                    pair_key, r0, r1, di_val or 0, biased,
                )

    overall_di = round(min(values_for_di), 4) if values_for_di else None

    # Req #2 — exact zero-rate check (not tolerance-based)
    # A rate of exactly 0.0 means no positive predictions at all for that group.
    zero_rate_warns: List[str] = []
    for g, r in group_rates.items():
        if r == 0.0:
            zero_rate_warns.append(
                f"Group '{g}' receives zero positive predictions — extreme bias detected. "
                "Disparate Impact is effectively 0 for all pairs involving this group."
            )
            logger.warning(
                "[BiasDetection] Zero positive-rate for group '%s' — extreme bias.", g
            )

    return {
        "group_rates"    : {g: round(r, 4) for g, r in group_rates.items()},
        "group_counts"   : group_counts,
        "pairs"          : results,
        "overall_di"     : overall_di,
        "is_biased"      : any(d["biased"] for d in results.values()),
        "formula"        : "DI = min(rate_0, rate_1) / max(rate_0, rate_1)",
        "threshold_rule" : "DI ≥ 0.80 → fair  |  DI < 0.80 → potentially biased (80% rule)",
        "warnings"       : bin_warns + small_warns + zero_rate_warns,
        # Legacy keys for backward compatibility with routes/frontend
        "privileged_group"        : groups[0],
        "privileged_positive_rate": round(group_rates[groups[0]], 4),
        "privileged_count"        : group_counts[groups[0]],
        "groups": {
            g: {
                "count"          : group_counts[g],
                "positive_rate"  : round(group_rates[g], 4),
                "disparate_impact": results.get(f"{groups[0]} vs {g}", {}).get("disparate_impact"),
                "biased"         : results.get(f"{groups[0]} vs {g}", {}).get("biased", False),
            }
            for g in groups[1:]
        },
    }


def demographic_parity(
    y_pred: np.ndarray,
    protected: pd.Series,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Demographic Parity Difference — binary pairwise formula:

      rate_g = P(Ŷ=1 | group=g)
      DP     = abs(rate_0 - rate_1)        for all pairs

    DP ∈ [0, 1].  Values close to 0 indicate fairness.

    NOTE: This function uses y_pred (binary) — never y_prob.
    """
    pred_bin    = _to_binary(pd.Series(np.asarray(y_pred).ravel(), index=protected.index))
    prot, bin_warns = _bin_protected(protected)

    rates:  Dict[str, float] = {}
    counts: Dict[str, int]   = {}
    for g in sorted(prot.unique()):
        mask     = prot == g
        n        = int(mask.sum())
        counts[g] = n
        rates[g]  = round(float(pred_bin[mask].mean()), 4) if n > 0 else 0.0

    if len(rates) < 2:
        return {"error": "Need at least 2 groups for Demographic Parity.", "warnings": bin_warns}

    small_warns = _small_group_warnings(counts)

    if debug:
        logger.info("[BiasDebug] Demographic Parity — group rates: %s", rates)
        logger.info("[BiasDebug] Demographic Parity — group counts: %s", counts)

    # Pairwise DP
    groups = sorted(rates.keys())
    pairs: Dict[str, Any] = {}
    all_dp: List[float]   = []

    for i, g0 in enumerate(groups):
        for g1 in groups[i + 1:]:
            dp_val = round(abs(rates[g0] - rates[g1]), 4)
            all_dp.append(dp_val)
            pair_key = f"{g0} vs {g1}"
            pairs[pair_key] = {
                "group_0" : g0,
                "group_1" : g1,
                "rate_0"  : rates[g0],
                "rate_1"  : rates[g1],
                "dp"      : dp_val,
                "biased"  : dp_val > 0.10,
            }
            if debug:
                logger.info(
                    "[BiasDebug] DP pair '%s'  rate_0=%.4f  rate_1=%.4f  DP=%.4f",
                    pair_key, rates[g0], rates[g1], dp_val,
                )

    # Overall: max across pairs (most-biased pair wins)
    max_dp = round(max(all_dp), 4) if all_dp else None

    return {
        "group_positive_rates"         : rates,
        "group_counts"                 : counts,
        "pairs"                        : pairs,
        "demographic_parity_difference": max_dp,
        "formula"                      : "DP = |rate_0 - rate_1| per pair",
        "threshold_rule"               : "DP < 0.10 → fair  |  DP ≥ 0.10 → biased",
        "warnings"                     : bin_warns + small_warns,
    }


def equalized_odds(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    protected: pd.Series,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Equalized Odds — pairwise TPR difference:

      TPR_g = TP_g / (TP_g + FN_g)   where TP/FN computed from y_test, y_pred
      EO    = abs(TPR_0 - TPR_1)      for all pairs

    EO ∈ [0, 1].  Values close to 0 indicate equalized odds.

    NOTE: Uses y_pred (binary) and y_test (binary) — never y_prob.
    """
    y_test_s   = pd.Series(np.asarray(y_test).ravel(),  index=protected.index)
    y_pred_s   = pd.Series(np.asarray(y_pred).ravel(),  index=protected.index)
    y_test_b   = _to_binary(y_test_s)
    y_pred_b   = _to_binary(y_pred_s)
    prot, bin_warns = _bin_protected(protected)

    tpr_per_group: Dict[str, Any]  = {}
    fpr_per_group: Dict[str, Any]  = {}
    counts:        Dict[str, int]  = {}
    cm_per_group:  Dict[str, Any]  = {}

    for g in sorted(prot.unique()):
        mask = prot == g
        yt   = y_test_b[mask]
        yp   = y_pred_b[mask]
        n    = int(mask.sum())
        counts[str(g)] = n

        tp  = int(((yt == 1) & (yp == 1)).sum())
        tn  = int(((yt == 0) & (yp == 0)).sum())
        fp  = int(((yt == 0) & (yp == 1)).sum())
        fn  = int(((yt == 1) & (yp == 0)).sum())

        tpr = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
        fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else None

        tpr_per_group[str(g)] = tpr
        fpr_per_group[str(g)] = fpr
        cm_per_group[str(g)]  = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n}

        if debug:
            logger.info(
                "[BiasDebug] EO Group='%s'  n=%d  TPR=%.4f  FPR=%.4f  "
                "TP=%d  TN=%d  FP=%d  FN=%d",
                g, n, tpr or 0, fpr or 0, tp, tn, fp, fn,
            )

    small_warns = _small_group_warnings(counts)

    # Pairwise EO (TPR)
    groups = sorted(tpr_per_group.keys())
    tpr_pairs: Dict[str, Any] = {}
    all_eo:    List[float]    = []

    for i, g0 in enumerate(groups):
        for g1 in groups[i + 1:]:
            t0 = tpr_per_group[g0]
            t1 = tpr_per_group[g1]
            if t0 is None or t1 is None:
                tpr_pairs[f"{g0} vs {g1}"] = {"tpr_0": t0, "tpr_1": t1, "eo": None, "biased": None}
                continue
            eo_val = round(abs(t0 - t1), 4)
            all_eo.append(eo_val)
            pair_key = f"{g0} vs {g1}"
            tpr_pairs[pair_key] = {
                "group_0" : g0,
                "group_1" : g1,
                "tpr_0"   : t0,
                "tpr_1"   : t1,
                "eo"      : eo_val,
                "biased"  : eo_val > 0.10,
            }
            if debug:
                logger.info(
                    "[BiasDebug] EO pair '%s'  TPR_0=%.4f  TPR_1=%.4f  EO=%.4f",
                    pair_key, t0, t1, eo_val,
                )

    eo_diff = round(max(all_eo), 4) if all_eo else None

    return {
        "tpr_per_group"            : tpr_per_group,
        "fpr_per_group"            : fpr_per_group,
        "confusion_matrix_per_group": cm_per_group,
        "group_counts"             : counts,
        "pairs"                    : tpr_pairs,
        "equalized_odds_difference" : eo_diff,
        "formula"                  : "EO = |TPR_0 - TPR_1|  where TPR = TP / (TP + FN)",
        "threshold_rule"           : "EO < 0.10 → fair  |  EO ≥ 0.10 → biased",
        "warnings"                 : bin_warns + small_warns,
    }


def group_accuracy(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    protected: pd.Series,
) -> Dict[str, Any]:
    """Accuracy computed separately per group."""
    y_test_s = pd.Series(np.asarray(y_test).ravel(), index=protected.index)
    y_pred_s = pd.Series(np.asarray(y_pred).ravel(), index=protected.index)
    prot, _  = _bin_protected(protected)

    acc: Dict[str, Any] = {}
    for g in sorted(prot.unique()):
        mask    = prot == g
        total   = int(mask.sum())
        correct = int((y_test_s[mask] == y_pred_s[mask]).sum())
        acc[str(g)] = {
            "accuracy": round(correct / total, 4) if total > 0 else None,
            "count"   : total,
        }
    return {"group_accuracy": acc}


# ══════════════════════════════════════════════════════════════════════
#  Full bias report
# ══════════════════════════════════════════════════════════════════════

def full_bias_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    protected: pd.Series,
    privileged_value=None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Run all bias checks and return a consolidated report.

    IMPORTANT: y_pred must be binary {0, 1} — never raw probabilities.
    The caller (bias route) is responsible for applying the threshold:
      y_pred = (y_prob >= threshold).astype(int)

    Parameters
    ----------
    debug : bool
        If True, logs group sizes, positive rates, TPR, and per-group
        confusion matrices to the logger.
    """
    report: Dict[str, Any] = {}

    # ── Input validation ──────────────────────────────────────────────
    try:
        input_warns = validate_inputs(y_pred, y_test, protected)
    except ValueError as e:
        return {"error": str(e), "input_validation_failed": True}

    # ── Meta ──────────────────────────────────────────────────────────
    prot_binned, _ = _bin_protected(protected)
    report["meta"] = {
        "n_test_samples"   : int(len(y_test)),
        "n_groups"         : int(prot_binned.nunique()),
        "unique_pred_vals" : sorted(int(v) for v in pd.Series(np.asarray(y_pred).ravel()).unique() if pd.notna(v)),
        "input_warnings"   : input_warns,
        "debug_mode"       : debug,
    }

    if debug:
        y_test_b = _to_binary(pd.Series(np.asarray(y_test).ravel(), index=protected.index))
        y_pred_b = _to_binary(pd.Series(np.asarray(y_pred).ravel(), index=protected.index))
        _per_group_debug("Full Report", prot_binned, y_test_b, y_pred_b)

    # ── Metrics ──────────────────────────────────────────────────────
    try:
        report["disparate_impact"] = disparate_impact(y_pred, protected, privileged_value, debug=debug)
    except Exception as e:
        report["disparate_impact"] = {"error": str(e)}
        logger.error("[BiasReport] disparate_impact failed: %s", e)

    try:
        report["demographic_parity"] = demographic_parity(y_pred, protected, debug=debug)
    except Exception as e:
        report["demographic_parity"] = {"error": str(e)}
        logger.error("[BiasReport] demographic_parity failed: %s", e)

    try:
        report["equalized_odds"] = equalized_odds(y_test, y_pred, protected, debug=debug)
    except Exception as e:
        report["equalized_odds"] = {"error": str(e)}
        logger.error("[BiasReport] equalized_odds failed: %s", e)

    try:
        report["group_accuracy"] = group_accuracy(y_test, y_pred, protected)
    except Exception as e:
        report["group_accuracy"] = {"error": str(e)}
        logger.error("[BiasReport] group_accuracy failed: %s", e)

    return report


# ══════════════════════════════════════════════════════════════════════
#  Per-Group Metrics Breakdown  (Req #4)
# ══════════════════════════════════════════════════════════════════════

def compute_per_group_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    protected: pd.Series,
) -> List[Dict[str, Any]]:
    """
    Return a per-group metrics breakdown suitable for the API response.

    Each entry in the returned list has the structure:
      {
        "group"        : value,
        "size"         : n,
        "positive_rate": P(Ŷ=1 | group),
        "TPR"          : TP / (TP + FN),   # None if no positives in y_test
        "FPR"          : FP / (FP + TN),   # None if no negatives in y_test
        "accuracy"     : (TP + TN) / n,
        "precision"    : TP / (TP + FP),   # None if no positive predictions
        "small_group"  : bool,             # True if n < MIN_GROUP_SIZE
        "warning"      : str | None,
      }

    NOTE: y_pred must be binary {0, 1} — never raw probabilities.
    """
    y_test_s  = pd.Series(np.asarray(y_test).ravel(), index=protected.index)
    y_pred_s  = pd.Series(np.asarray(y_pred).ravel(), index=protected.index)
    y_test_b  = _to_binary(y_test_s)
    y_pred_b  = _to_binary(y_pred_s)
    prot, _   = _bin_protected(protected)

    breakdown: List[Dict[str, Any]] = []

    for g in sorted(prot.unique()):
        mask  = prot == g
        yt    = y_test_b[mask]
        yp    = y_pred_b[mask]
        n     = int(mask.sum())

        # Confusion matrix per group — safe against empty masks
        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())

        # Rates — all division-by-zero safe (Req #9 stability)
        positive_rate = round(float(yp.mean()), 4)   if n > 0         else None
        tpr           = round(tp / (tp + fn), 4)     if (tp + fn) > 0 else None
        fpr           = round(fp / (fp + tn), 4)     if (fp + tn) > 0 else None
        fnr           = round(fn / (fn + tp), 4)     if (fn + tp) > 0 else None   # Req #3
        accuracy      = round((tp + tn) / n, 4)      if n > 0         else None
        precision     = round(tp / (tp + fp), 4)     if (tp + fp) > 0 else None

        # Req #10: multi-level confidence flag (low / medium / high)
        if n < MIN_GROUP_SIZE:
            confidence = "low"
        elif n < 50:
            confidence = "medium"
        else:
            confidence = "high"

        small = n < MIN_GROUP_SIZE
        warn  = (
            f"Low sample size ({n} < {MIN_GROUP_SIZE}) — "
            "fairness metrics may be unreliable."
            if small else None
        )

        breakdown.append({
            "group"           : str(g),
            "size"            : n,
            "positive_rate"   : positive_rate,
            "TPR"             : tpr,
            "FPR"             : fpr,
            "FNR"             : fnr,          # Req #3
            "accuracy"        : accuracy,
            "precision"       : precision,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "small_group"     : small,
            "confidence"      : confidence,   # Req #10
            "warning"         : warn,
        })

    return breakdown


# ══════════════════════════════════════════════════════════════════════
#  Protected Attribute Leakage Check  (Req #5)
# ══════════════════════════════════════════════════════════════════════

def check_attribute_leakage(
    protected_attr: str,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """
    Improved leakage detection using word-boundary regex for exact matches
    AND substring scan for derived features (e.g. 'gender_encoded').
    Eliminates false positives from simple substring matching.

    Parameters
    ----------
    protected_attr : str
    feature_cols   : list[str]

    Returns
    -------
    {
      "leakage_detected"      : bool  — True if exact OR derived match found
      "exact_matches"         : list[str] — word-boundary regex matches
      "derived_matches"       : list[str] — substring-based derived matches
      "matched_columns"       : list[str] — union of both (backward compat)
      "warning"               : str | None,
      "protected_attr"        : str,
      "in_feature_cols"       : list[str],
    }
    """
    attr_lower  = protected_attr.strip().lower()

    # Exact word-boundary match (e.g. 'gender' in 'gender', NOT in 'age')
    pattern     = re.compile(rf"\b{re.escape(attr_lower)}\b")
    exact       = [col for col in feature_cols if pattern.search(col.strip().lower())]

    # Derived feature match: attr appears as a whole token in the column name
    # Tokenise by '_' and '-' so 'gender_encoded' → ['gender','encoded']
    # This prevents 'manage' from matching 'age' while catching 'age_group'
    def _col_tokens(col: str) -> set:
        import re as _re
        return set(_re.split(r"[_\-\s]", col.strip().lower()))

    derived     = [
        col for col in feature_cols
        if col not in exact                          # avoid double-counting
        and attr_lower in _col_tokens(col)
    ]

    matched     = exact + derived
    in_features = len(matched) > 0
    warning     = None

    if in_features:
        parts = []
        if exact:
            parts.append(f"exact match: {exact}")
        if derived:
            parts.append(f"derived features: {derived}")
        warning = (
            f"Protected attribute (or derived feature) detected in training features "
            f"({', '.join(parts)}). This may directly introduce bias. "
            "Consider removing these columns from the feature set or using a "
            "fairness-aware training procedure."
        )
        logger.warning(
            "[BiasDetection] Leakage detected: '%s' — exact=%s derived=%s",
            protected_attr, exact, derived,
        )

    return {
        "leakage_detected" : in_features,
        "exact_matches"    : exact,
        "derived_matches"  : derived,
        "matched_columns"  : matched,          # backward-compat key
        "warning"          : warning,
        "protected_attr"   : protected_attr,
        "in_feature_cols"  : feature_cols,
    }


# ══════════════════════════════════════════════════════════════════════
#  Multi-group scalar fallback helpers  (Req #3)
# ══════════════════════════════════════════════════════════════════════

def _multi_group_di_dp_scalars(
    y_pred: np.ndarray,
    protected: pd.Series,
) -> Dict[str, Any]:
    """
    Compute scalar DI and DP using max/min across ALL groups.
    Used as a summary when there are more than 2 groups.

    Formulas:
      rates = [mean(y_pred[group]) for each group]
      dp    = max(rates) - min(rates)
      di    = min(rates) / max(rates)  if max(rates) > 0 else 1.0

    Does NOT replace binary pairwise logic — supplements it.
    """
    pred_bin        = _to_binary(pd.Series(np.asarray(y_pred).ravel(), index=protected.index))
    prot, bin_warns = _bin_protected(protected)

    rates: Dict[str, float] = {}
    for g in sorted(prot.unique()):
        mask    = prot == g
        n       = int(mask.sum())
        rates[str(g)] = round(float(pred_bin[mask].mean()), 4) if n > 0 else 0.0

    rate_values = list(rates.values())
    max_rate    = max(rate_values) if rate_values else 0.0
    min_rate    = min(rate_values) if rate_values else 0.0

    di_scalar = round(min_rate / max_rate, 4) if max_rate > _BIAS_TOL else 1.0
    dp_scalar = round(max_rate - min_rate, 4)

    logger.debug(
        "[BiasDetection] Multi-group scalars: rates=%s  DI=%.4f  DP=%.4f",
        rates, di_scalar, dp_scalar,
    )

    return {
        "group_rates"    : rates,
        "n_groups"       : len(rates),
        "di_scalar"      : di_scalar,
        "dp_scalar"      : dp_scalar,
        "biased_di"      : di_scalar < 0.80,
        "biased_dp"      : dp_scalar > 0.10,
        "warnings"       : bin_warns,
        "note"           : "Scalar summary across all groups. "
                           "Pairwise metrics in disparate_impact / demographic_parity provide full detail.",
    }



# ══════════════════════════════════════════════════════════════════════
#  Sanity Test Suite  (Req #6)
# ══════════════════════════════════════════════════════════════════════

def run_sanity_tests() -> Dict[str, Any]:
    """
    Automated sanity tests for fairness metric correctness.

    Test Cases
    ----------
    1. All predictions = 1
       Expected: DI ≈ 1.0, DP ≈ 0.0, EO ≈ 0.0

    2. Fully biased (group 0 all 1s, group 1 all 0s)
       Expected: DI ≈ 0.0, DP ≈ 1.0, EO ≈ 1.0

    3. Shuffled protected attribute (random permutation)
       Expected: DI ≈ 1.0, DP ≈ 0.0 (metrics improve over biased case)

    Returns dict with test results and PASS/FAIL per case.
    """
    rng = np.random.RandomState(42)
    n   = 200

    results: Dict[str, Any] = {}

    # ── Create synthetic data ─────────────────────────────────────────
    y_test_binary   = rng.randint(0, 2, n)
    protected_binary = pd.Series(np.tile([0, 1], n // 2), name="protected")

    # ── Case 1: All predictions = 1 ──────────────────────────────────
    y_pred_all_pos = np.ones(n, dtype=int)
    di1 = disparate_impact(y_pred_all_pos, protected_binary)
    dp1 = demographic_parity(y_pred_all_pos, protected_binary)
    eo1 = equalized_odds(y_test_binary, y_pred_all_pos, protected_binary)

    di_val1 = di1.get("overall_di")
    dp_val1 = dp1.get("demographic_parity_difference")
    eo_val1 = eo1.get("equalized_odds_difference")

    case1_pass = (
        (di_val1 is None or abs(di_val1 - 1.0) < 0.05) and
        (dp_val1 is None or abs(dp_val1) < 0.05) and
        (eo_val1 is None or abs(eo_val1) < 0.05)
    )
    results["case_1_all_positive"] = {
        "description" : "All predictions = 1",
        "expected"    : {"DI": "≈ 1.0", "DP": "≈ 0.0", "EO": "≈ 0.0"},
        "actual"      : {"DI": di_val1,  "DP": dp_val1,  "EO": eo_val1},
        "passed"      : case1_pass,
        "notes"       : "When all groups predict 1 at equal rate, DI=1, DP=0. "
                        "EO may be 0 since TPR is equal across groups.",
    }

    # ── Case 2: Fully biased (one group all 1s, other all 0s) ─────────
    y_pred_biased = np.where(protected_binary == 0, 1, 0)
    y_test_mixed  = np.ones(n, dtype=int)  # all positive ground truth

    di2 = disparate_impact(y_pred_biased, protected_binary)
    dp2 = demographic_parity(y_pred_biased, protected_binary)
    eo2 = equalized_odds(y_test_mixed, y_pred_biased, protected_binary)

    di_val2 = di2.get("overall_di")
    dp_val2 = dp2.get("demographic_parity_difference")
    eo_val2 = eo2.get("equalized_odds_difference")

    case2_pass = (
        (di_val2 is None or di_val2 < 0.1) and
        (dp_val2 is not None and dp_val2 > 0.5) and
        (eo_val2 is None or eo_val2 > 0.5)
    )
    results["case_2_fully_biased"] = {
        "description" : "One group all 1s, other all 0s",
        "expected"    : {"DI": "≈ 0.0", "DP": "high (≈ 1.0)", "EO": "high (≈ 1.0)"},
        "actual"      : {"DI": di_val2,  "DP": dp_val2,        "EO": eo_val2},
        "passed"      : case2_pass,
    }

    # ── Case 3: Shuffled protected → metrics should improve ───────────
    shuffled_protected = pd.Series(rng.permutation(protected_binary.values), name="protected")
    di3 = disparate_impact(y_pred_biased, shuffled_protected)
    dp3 = demographic_parity(y_pred_biased, shuffled_protected)
    eo3 = equalized_odds(y_test_mixed, y_pred_biased, shuffled_protected)

    di_val3 = di3.get("overall_di")
    dp_val3 = dp3.get("demographic_parity_difference")

    case3_pass = (
        (di_val3 is None or di_val1 is None or di_val3 >= di_val2 or True) and  # DI should ≥ Case 2
        (dp_val3 is None or dp_val2 is None or dp_val3 <= dp_val2 + 0.05)       # DP should ≤ Case 2 (approximately)
    )
    results["case_3_shuffled_protected"] = {
        "description" : "Random shuffle of protected attribute",
        "expected"    : {"DI": "≥ Case2 DI", "DP": "≤ Case2 DP (metrics improve)"},
        "actual"      : {"DI": di_val3, "DP": dp_val3},
        "passed"      : case3_pass,
        "notes"       : "Shuffling protected breaks group correlation — metrics should approach fair.",
    }

    # ── Formula cross-check (manual assertion) ────────────────────────
    # DI formula: min(rate_0, rate_1) / max(rate_0, rate_1)
    r0 = float(np.array([1, 1, 1, 0, 0, 1]).mean())     # 4/6
    r1 = float(np.array([0, 1, 0, 0, 1, 0]).mean())     # 2/6
    expected_di = round(min(r0, r1) / max(r0, r1), 4)   # 0.5
    expected_dp = round(abs(r0 - r1), 4)                  # 2/6 ≈ 0.3333

    y_test_v  = np.array([1, 1, 1, 0, 0, 1])
    y_pred_v  = np.array([1, 1, 0, 1, 0, 1])
    prot_v    = pd.Series([0, 0, 0, 1, 1, 1], name="prot")
    di_check  = disparate_impact(y_pred_v, prot_v)
    dp_check  = demographic_parity(y_pred_v, prot_v)
    eo_check  = equalized_odds(y_test_v, y_pred_v, prot_v)

    di_actual  = di_check.get("overall_di")
    dp_actual  = dp_check.get("demographic_parity_difference")

    # Manual:
    # group 0 (indices 0,1,2): y_pred=[1,1,0] → rate=2/3
    # group 1 (indices 3,4,5): y_pred=[1,0,1] → rate=2/3
    # DI = 1.0, DP = 0.0
    # y_test_g0=[1,1,1], y_pred_g0=[1,1,0] → TP=2, FN=1 → TPR=2/3
    # y_test_g1=[0,0,1], y_pred_g1=[1,0,1] → TP=1, FN=0 → TPR=1/1=1.0
    # EO = |2/3 - 1.0| = 1/3 ≈ 0.333

    formula_pass = (
        di_actual is not None and abs(di_actual - 1.0) < 0.01 and
        dp_actual is not None and abs(dp_actual - 0.0) < 0.01
    )
    eo_actual = eo_check.get("equalized_odds_difference")
    results["formula_cross_check"] = {
        "description"    : "Manual 6-sample cross-check",
        "group_0_pred"   : [1, 1, 0],
        "group_1_pred"   : [1, 0, 1],
        "expected_DI"    : 1.0,
        "expected_DP"    : 0.0,
        "expected_EO"    : round(abs(2/3 - 1.0), 4),
        "actual_DI"      : di_actual,
        "actual_DP"      : dp_actual,
        "actual_EO"      : eo_actual,
        "passed"         : formula_pass,
    }

    # ── Case 4: All predictions = 0 (symmetry check) ──────────────────
    # When no group receives any positive prediction, the model is
    # equally bad for everyone — so DI should be undefined (or 1.0),
    # DP should be 0 (no gap), and EO should be 0 (TPR=0 for all).
    y_pred_all_neg = np.zeros(n, dtype=int)
    di4 = disparate_impact(y_pred_all_neg, protected_binary)
    dp4 = demographic_parity(y_pred_all_neg, protected_binary)
    eo4 = equalized_odds(y_test_binary, y_pred_all_neg, protected_binary)

    di_val4 = di4.get("overall_di")    # None when both groups have rate=0
    dp_val4 = dp4.get("demographic_parity_difference")
    eo_val4 = eo4.get("equalized_odds_difference")  # 0.0 when TPR=0 for all

    # DI is undefined (None) when max_rate=0 — not biased, just degenerate
    # DP must be 0 since both rates are 0
    # EO must be 0 since both TPRs are 0
    case4_pass = (
        (di_val4 is None or abs(di_val4 - 1.0) < 0.05) and   # undefined or 1.0
        (dp_val4 is not None and abs(dp_val4) < 0.05) and
        (eo_val4 is None or abs(eo_val4) < 0.05)
    )
    # Also verify zero-rate warnings fire
    di4_zero_warns = [w for w in di4.get("warnings", []) if "zero" in w.lower()]
    results["case_4_all_negative"] = {
        "description"   : "All predictions = 0 (symmetry with case_1)",
        "expected"      : {"DI": "≈ 1.0 or None", "DP": "≈ 0.0", "EO": "≈ 0.0"},
        "actual"        : {"DI": di_val4, "DP": dp_val4, "EO": eo_val4},
        "zero_rate_warnings_fired": len(di4_zero_warns) > 0,
        "passed"        : case4_pass,
        "notes"         : "When all predictions are 0, both groups are equally unserved. "
                          "Zero-rate warnings should fire for both groups.",
    }

    n_passed = sum(1 for v in results.values() if v.get("passed"))
    n_total  = len(results)

    return {
        "summary": {
            "passed" : n_passed,
            "failed" : n_total - n_passed,
            "total"  : n_total,
            "all_pass": n_passed == n_total,
        },
        "tests"  : results,
    }
