"""
bias_explanation.py  —  "Why Bias is Happening" Explanation Engine
===================================================================

Provides human-readable, non-technical explanations for observed
fairness disparities by combining three independent signals:

  1. Group-level prediction disparity   — who is getting what
  2. Feature distribution across groups — what data differs
  3. Feature importance from the model  — what the model relies on

The final "bias driver score" for each feature is:

    driver_score = feature_importance * group_variation

Features with high driver scores are the most likely structural
causes of the observed bias.

Public API
----------
  explain_bias(
      df            : pd.DataFrame,
      model         : sklearn estimator,
      feature_cols  : list[str],
      protected_col : str,
      y_pred        : np.ndarray,
      per_group     : list[dict],     # from compute_per_group_metrics()
      top_n         : int = 5,
  ) -> BiasExplanation

  BiasExplanation is a TypedDict / plain dict with keys:
    "group_disparity"      : dict   — highest/lowest positive-rate groups
    "feature_distributions": list   — per-feature group variation
    "feature_importances"  : list   — model feature weights
    "bias_drivers"         : list   — combined driver scores (top N)
    "explanation_text"     : list   — plain-English sentences
    "computation_ms"       : float  — wall-clock time
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
_TOP_N_DEFAULT     = 5
_MIN_STD_THRESHOLD = 1e-8     # skip constant features
_HIGH_VAR_RATIO    = 0.20     # group variation > 20% of global std → notable


# =============================================================================
#  1. Feature Distribution Analysis
# =============================================================================

def _feature_distributions(
    df: pd.DataFrame,
    feature_cols: List[str],
    protected_col: str,
) -> List[Dict[str, Any]]:
    """
    For each numerical feature compute per-group mean and the
    max-minus-min spread across groups.

    Returns list sorted by group_variation descending.
    """
    results: List[Dict[str, Any]] = []
    groups = df[protected_col].unique()

    for col in feature_cols:
        if col == protected_col:
            continue
        series = df[col]
        # Only numerical columns
        if not pd.api.types.is_numeric_dtype(series):
            continue
        # Skip constant features
        if series.std() < _MIN_STD_THRESHOLD:
            continue

        global_std   = float(series.std())
        group_means  = {}
        for g in groups:
            vals = series[df[protected_col] == g].dropna()
            if len(vals) > 0:
                group_means[str(g)] = round(float(vals.mean()), 4)

        if len(group_means) < 2:
            continue

        mean_vals      = list(group_means.values())
        variation      = round(max(mean_vals) - min(mean_vals), 4)
        variation_pct  = round(variation / global_std, 4) if global_std > 0 else 0.0
        notable        = variation_pct >= _HIGH_VAR_RATIO

        results.append({
            "feature"        : col,
            "group_means"    : group_means,
            "group_variation": variation,
            "variation_pct_std": variation_pct,
            "notable"        : notable,
        })

    results.sort(key=lambda x: x["group_variation"], reverse=True)
    return results


# =============================================================================
#  2. Feature Importance Extraction
# =============================================================================

def _extract_feature_importance(
    model: Any,
    feature_cols: List[str],
) -> List[Dict[str, Any]]:
    """
    Extract feature importance from the model using:
      - feature_importances_ (tree models)
      - |coef_|              (linear models — takes first row if multiclass)

    Returns list sorted by importance descending.
    """
    importances: Optional[np.ndarray] = None

    try:
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_).ravel()
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            if coef.ndim > 1:
                coef = coef[0]          # binary classification: first row
            importances = np.abs(coef).ravel()
    except Exception as exc:
        logger.warning("[BiasExplainer] Could not extract feature importances: %s", exc)
        return []

    if importances is None or len(importances) == 0:
        return []

    # Align length to feature_cols (may differ after encoding)
    n = min(len(importances), len(feature_cols))
    total = importances[:n].sum()
    if total < 1e-12:
        total = 1.0   # avoid division by zero

    results = []
    for i, col in enumerate(feature_cols[:n]):
        results.append({
            "feature"        : col,
            "importance"     : round(float(importances[i]), 6),
            "importance_pct" : round(float(importances[i] / total), 4),
        })

    results.sort(key=lambda x: x["importance"], reverse=True)
    return results


# =============================================================================
#  3. Bias Driver Score  (importance × variation)
# =============================================================================

def _compute_bias_drivers(
    feature_distributions: List[Dict[str, Any]],
    feature_importances  : List[Dict[str, Any]],
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    driver_score = importance * group_variation

    If no importances are available, falls back to ranking by
    group_variation alone.
    """
    # Build lookup: feature → importance
    imp_map: Dict[str, float] = {
        fi["feature"]: fi["importance"]
        for fi in feature_importances
    }
    # Normalise variations to [0, 1] for interpretable scoring
    all_vars   = [fd["group_variation"] for fd in feature_distributions]
    max_var    = max(all_vars) if all_vars else 1.0
    if max_var < 1e-12:
        max_var = 1.0

    drivers: List[Dict[str, Any]] = []
    for fd in feature_distributions:
        feat    = fd["feature"]
        var     = fd["group_variation"]
        imp     = imp_map.get(feat, np.nan)
        if np.isnan(imp):
            # No model importance: score purely on variation
            score = round(var / max_var, 4)
            has_imp = False
        else:
            score   = round(float(imp) * (var / max_var), 4)
            has_imp = True

        drivers.append({
            "feature"        : feat,
            "driver_score"   : score,
            "importance"     : round(imp, 6) if has_imp else None,
            "group_variation": var,
            "group_means"    : fd["group_means"],
            "notable"        : fd["notable"],
            "has_importance" : has_imp,
        })

    drivers.sort(key=lambda x: x["driver_score"], reverse=True)
    return drivers[:top_n]


# =============================================================================
#  4. Group-Level Prediction Rate Analysis
# =============================================================================

def _group_prediction_analysis(
    per_group_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Identify highest and lowest positive-prediction-rate groups
    from pre-computed per_group_metrics.
    """
    valid = [
        g for g in per_group_metrics
        if g.get("positive_rate") is not None
    ]
    if not valid:
        return {"groups": [], "highest": None, "lowest": None, "disparity": None}

    highest = max(valid, key=lambda g: g["positive_rate"])
    lowest  = min(valid, key=lambda g: g["positive_rate"])
    disparity = round(highest["positive_rate"] - lowest["positive_rate"], 4)

    return {
        "groups"   : [
            {
                "group"        : g["group"],
                "positive_rate": g["positive_rate"],
                "size"         : g["size"],
            }
            for g in valid
        ],
        "highest"  : {
            "group"        : highest["group"],
            "positive_rate": highest["positive_rate"],
            "size"         : highest["size"],
        },
        "lowest"   : {
            "group"        : lowest["group"],
            "positive_rate": lowest["positive_rate"],
            "size"         : lowest["size"],
        },
        "disparity": disparity,
    }


# =============================================================================
#  5. Explanation Text Generator
# =============================================================================

_SEVERITY_LABELS = [
    (0.40, "dramatically"),
    (0.20, "significantly"),
    (0.10, "noticeably"),
    (0.00, "somewhat"),
]


def _severity_word(disparity: float) -> str:
    for threshold, word in _SEVERITY_LABELS:
        if abs(disparity) >= threshold:
            return word
    return "slightly"


def _generate_explanations(
    group_disparity  : Dict[str, Any],
    bias_drivers     : List[Dict[str, Any]],
    feature_dists    : List[Dict[str, Any]],
    feature_imps     : List[Dict[str, Any]],
    protected_col    : str,
) -> List[str]:
    """
    Produce a ranked list of plain-English, non-technical explanation
    sentences. Each sentence addresses one specific cause.

    Sentence structure is kept to ≤ 20 words where possible.
    """
    sentences: List[str] = []

    # ── Part A: Group prediction disparity ────────────────────────────
    hi = group_disparity.get("highest")
    lo = group_disparity.get("lowest")
    dp = group_disparity.get("disparity")

    if hi and lo and dp is not None:
        severity = _severity_word(dp)
        if dp < 0.02:
            sentences.append(
                f"Prediction rates are nearly equal across all groups of '{protected_col}' "
                f"— no significant disparity detected."
            )
        else:
            hi_pct = round(hi["positive_rate"] * 100, 1)
            lo_pct = round(lo["positive_rate"] * 100, 1)
            sentences.append(
                f"Group '{hi['group']}' receives {severity} more positive predictions "
                f"({hi_pct}%) than group '{lo['group']}' ({lo_pct}%) "
                f"— a gap of {round(dp * 100, 1)} percentage points."
            )

    # ── Part B: Top bias drivers (feature + importance + variation) ────
    for rank, driver in enumerate(bias_drivers[:5]):
        feat    = driver["feature"]
        var     = driver["group_variation"]
        imp     = driver.get("importance")
        notable = driver["notable"]
        means   = driver["group_means"]

        # Find the group with the highest/lowest mean for this feature
        if means:
            high_grp = max(means, key=means.get)
            low_grp  = min(means, key=means.get)
            mean_diff_str = (
                f"Group '{high_grp}' averages {means[high_grp]:.2f} "
                f"vs group '{low_grp}' at {means[low_grp]:.2f}."
            )
        else:
            mean_diff_str = ""

        if imp is not None and notable:
            sentences.append(
                f"Feature '{feat}' strongly influences model predictions "
                f"and varies considerably across groups ({var:.2f} unit spread). "
                f"{mean_diff_str} "
                f"This is a likely structural cause of the observed disparity."
            )
        elif imp is not None and not notable:
            sentences.append(
                f"Feature '{feat}' is important to the model "
                f"but shows only minor group-level variation ({var:.2f} unit spread). "
                f"Its contribution to bias is limited."
            )
        elif imp is None and notable:
            sentences.append(
                f"Feature '{feat}' varies substantially across groups ({var:.2f} unit spread). "
                f"{mean_diff_str} "
                f"This distributional difference may contribute to prediction disparity."
            )
        else:
            # Low driver score — skip to avoid noise
            break

    # ── Part C: Features with high variation but low/no importance ─────
    high_var_not_in_drivers = [
        fd for fd in feature_dists[:10]
        if fd["notable"]
        and fd["feature"] not in {d["feature"] for d in bias_drivers}
    ]
    if high_var_not_in_drivers:
        feat = high_var_not_in_drivers[0]["feature"]
        sentences.append(
            f"Feature '{feat}' also differs across groups "
            f"but has lower model influence — monitor it if bias persists."
        )

    # ── Part D: No importances available fallback ──────────────────────
    if not feature_imps:
        sentences.append(
            "Model does not expose feature importances. "
            "Run the Feature Importance analysis for a full bias driver breakdown."
        )

    # ── Part E: Actionable closing ────────────────────────────────────
    if bias_drivers and bias_drivers[0]["notable"]:
        top_feat = bias_drivers[0]["feature"]
        sentences.append(
            f"To reduce bias, consider auditing '{top_feat}': "
            "check for data collection imbalance, apply fairness-aware reweighting, "
            "or remove this feature from training."
        )

    return sentences


# =============================================================================
#  Main public function
# =============================================================================

def explain_bias(
    df            : pd.DataFrame,
    model         : Any,
    feature_cols  : List[str],
    protected_col : str,
    y_pred        : np.ndarray,
    per_group     : List[Dict[str, Any]],
    top_n         : int = _TOP_N_DEFAULT,
) -> Dict[str, Any]:
    """
    Generate a structured "Why Bias is Happening" explanation.

    Parameters
    ----------
    df            : Full (processed) dataset — used for feature distributions.
    model         : Trained sklearn-compatible model.
    feature_cols  : Column names the model was trained on.
    protected_col : Protected attribute column name.
    y_pred        : Binary predictions aligned with df[protected_col].
    per_group     : Output of compute_per_group_metrics() — already computed
                    in the route, passed in to avoid recomputation.
    top_n         : Number of top bias drivers to surface (default 5).

    Returns
    -------
    {
      "group_disparity"       : {highest, lowest, disparity, groups},
      "feature_distributions" : [...sorted by variation...],
      "feature_importances"   : [...sorted by importance...],
      "bias_drivers"          : [...top_n, sorted by driver_score...],
      "explanation_text"      : [plain-English sentences],
      "computation_ms"        : float,
    }

    Never raises — returns partial results with an "error" key on failure.
    """
    t0 = time.perf_counter()

    try:
        # ── 1. Feature distributions ───────────────────────────────────
        try:
            _df_with_prot = df.copy()
            # Ensure protected column is in the df slice we analyse
            feat_dist = _feature_distributions(_df_with_prot, feature_cols, protected_col)
        except Exception as exc:
            logger.warning("[BiasExplainer] feature_distributions failed: %s", exc)
            feat_dist = []

        # ── 2. Feature importance ──────────────────────────────────────
        try:
            feat_imp = _extract_feature_importance(model, feature_cols)
        except Exception as exc:
            logger.warning("[BiasExplainer] feature_importance failed: %s", exc)
            feat_imp = []

        # ── 3. Bias drivers ────────────────────────────────────────────
        try:
            drivers = _compute_bias_drivers(feat_dist, feat_imp, top_n)
        except Exception as exc:
            logger.warning("[BiasExplainer] bias_drivers failed: %s", exc)
            drivers = []

        # ── 4. Group prediction analysis ───────────────────────────────
        try:
            group_disp = _group_prediction_analysis(per_group)
        except Exception as exc:
            logger.warning("[BiasExplainer] group_prediction_analysis failed: %s", exc)
            group_disp = {"groups": [], "highest": None, "lowest": None, "disparity": None}

        # ── 5. Explanation text ────────────────────────────────────────
        try:
            explanations = _generate_explanations(
                group_disp, drivers, feat_dist, feat_imp, protected_col
            )
        except Exception as exc:
            logger.warning("[BiasExplainer] explanation_text failed: %s", exc)
            explanations = ["Explanation generation failed — check logs for details."]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "[BiasExplainer] Completed in %.1f ms — %d drivers, %d explanations",
            elapsed_ms, len(drivers), len(explanations),
        )

        return {
            "group_disparity"       : group_disp,
            "feature_distributions" : feat_dist[:top_n * 2],   # top 2×N for context
            "feature_importances"   : feat_imp[:top_n],
            "bias_drivers"          : drivers,
            "explanation_text"      : explanations,
            "computation_ms"        : elapsed_ms,
        }

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.error("[BiasExplainer] Unexpected error: %s", exc)
        return {
            "group_disparity"       : {},
            "feature_distributions" : [],
            "feature_importances"   : [],
            "bias_drivers"          : [],
            "explanation_text"      : [
                "Bias explanation could not be generated. "
                "Please ensure the model and dataset are loaded correctly."
            ],
            "computation_ms"        : elapsed_ms,
            "error"                 : str(exc),
        }
