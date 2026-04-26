"""
threshold_optimizer.py  —  Automatic Optimal Probability Threshold Finder
=========================================================================

Sweeps thresholds from 0.01 → 0.99, computes Precision / Recall / F1 /
Accuracy at every point, then selects the best threshold using a two-stage
strategy:

  PRIMARY  : Maximise F1-score
  SECONDARY: If the dataset is class-imbalanced (minority < 20 %), also
             return a *recall-priority* threshold (highest Recall while
             F1 is within 5 % of its maximum).

All edge cases are handled:
  - Division-by-zero (zero_division=0 throughout)
  - Empty probability arrays
  - Constant probability outputs
  - String / boolean class labels
  - Thresholds that produce all-positive or all-negative predictions

Public API
----------
find_optimal_threshold(y_test, y_pred_proba) -> dict
apply_threshold(y_pred_proba, threshold) -> np.ndarray
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

# ── Constants ────────────────────────────────────────────────────────────────
_THRESHOLD_STEPS  = 200          # number of threshold candidates
_IMBALANCE_CUTOFF = 20.0         # minority % below which we call it imbalanced
_RECALL_F1_SLACK  = 0.05         # recall-priority: F1 must stay within this of max
_MIN_SAMPLES      = 5            # minimum samples per class to compute metrics


# ── Public helpers ────────────────────────────────────────────────────────────

def apply_threshold(y_pred_proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert predicted probabilities to binary labels using *threshold*.

    Returns integer (0/1) labels.  Handles NaN / Inf in probabilities safely.
    """
    proba = np.asarray(y_pred_proba, dtype=np.float64)
    # Replace non-finite values with 0.5 (ambiguous → predict negative class)
    proba = np.where(np.isfinite(proba), proba, 0.5)
    return (proba >= threshold).astype(int)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _encode_labels(
    y_test: np.ndarray,
) -> Tuple[np.ndarray, Optional[LabelEncoder]]:
    """
    Encode target labels to {0, 1} integers if they are not already.

    Returns (y_encoded, label_encoder_or_None).
    The positive class is always the last class alphabetically / by sort order
    (mirroring sklearn's LabelEncoder behaviour).
    """
    unique = np.unique(y_test)
    if len(unique) != 2:
        raise ValueError(
            f"Threshold optimisation requires exactly 2 classes; "
            f"found {len(unique)}: {unique.tolist()}"
        )

    # Already numeric 0/1 — no encoding needed
    if set(unique.tolist()).issubset({0, 1, 0.0, 1.0}):
        return y_test.astype(int), None

    le = LabelEncoder()
    y_enc = le.fit_transform(y_test.astype(str))
    return y_enc, le


def _minority_pct(y_enc: np.ndarray) -> float:
    """Return the percentage share of the minority class."""
    _, counts = np.unique(y_enc, return_counts=True)
    return float(min(counts)) / float(len(y_enc)) * 100.0


def _safe_metric(fn, y_true, y_pred, **kw) -> float:
    """Call a sklearn metric function, returning 0.0 on any exception."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(fn(y_true, y_pred, **kw))
    except Exception:
        return 0.0


def _sweep_thresholds(
    y_enc: np.ndarray,
    y_prob: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Evaluate classification metrics at _THRESHOLD_STEPS evenly-spaced
    thresholds between 0.01 and 0.99 (inclusive).

    Returns a list of dicts, one per threshold, with keys:
        threshold, precision, recall, f1, accuracy
    """
    thresholds = np.linspace(0.01, 0.99, _THRESHOLD_STEPS)
    rows: List[Dict[str, float]] = []

    for t in thresholds:
        y_pred = apply_threshold(y_prob, float(t))

        # Skip degenerate predictions (all-same class) but still record the row
        unique_pred = np.unique(y_pred)
        if len(unique_pred) == 1:
            # Precision/Recall/F1 are ill-defined for single-class predictions
            rows.append({
                "threshold": round(float(t), 4),
                "precision": 0.0 if unique_pred[0] == 0 else 1.0,
                "recall":    float(_safe_metric(recall_score,    y_enc, y_pred, zero_division=0)),
                "f1":        0.0,
                "accuracy":  float(_safe_metric(accuracy_score,  y_enc, y_pred)),
            })
            continue

        rows.append({
            "threshold": round(float(t), 4),
            "precision": float(_safe_metric(precision_score, y_enc, y_pred, zero_division=0)),
            "recall":    float(_safe_metric(recall_score,    y_enc, y_pred, zero_division=0)),
            "f1":        float(_safe_metric(f1_score,        y_enc, y_pred, zero_division=0)),
            "accuracy":  float(_safe_metric(accuracy_score,  y_enc, y_pred)),
        })

    return rows


def _best_f1_threshold(curve: List[Dict[str, float]]) -> int:
    """Return the index into *curve* with the highest F1-score."""
    f1_values = [r["f1"] for r in curve]
    return int(np.argmax(f1_values))


def _best_recall_priority_threshold(
    curve: List[Dict[str, float]],
    max_f1: float,
) -> int:
    """
    Among all thresholds whose F1 stays within _RECALL_F1_SLACK of *max_f1*,
    return the index with the highest Recall.
    """
    slack_val = max_f1 * (1.0 - _RECALL_F1_SLACK)
    candidates = [
        (i, r["recall"])
        for i, r in enumerate(curve)
        if r["f1"] >= slack_val
    ]
    if not candidates:
        # Fallback: just take the best-F1 index
        return _best_f1_threshold(curve)
    return max(candidates, key=lambda x: x[1])[0]


def _confusion_matrix_at(
    y_enc: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> List[List[int]]:
    y_pred = apply_threshold(y_prob, threshold)
    return confusion_matrix(y_enc, y_pred).tolist()


def _build_explanation(
    best_threshold: float,
    strategy: str,
    minority_pct: float,
    best_row: Dict[str, float],
    label_encoder: Optional[LabelEncoder],
) -> str:
    """
    Generate a human-readable explanation of why this threshold was chosen.
    """
    class_info = ""
    if label_encoder is not None:
        classes = label_encoder.classes_.tolist()
        class_info = (
            f" (class '{classes[0]}' = 0, class '{classes[1]}' = 1)"
        )

    imbalance_note = ""
    if minority_pct < _IMBALANCE_CUTOFF:
        imbalance_note = (
            f" The dataset is **class-imbalanced** — the minority class "
            f"represents only {minority_pct:.1f}% of samples."
        )

    if strategy == "f1":
        return (
            f"**F1-Score Maximisation** — the threshold **{best_threshold:.3f}**"
            f"{class_info} was selected because it yields the highest F1-score "
            f"({best_row['f1']:.3f}) across all {_THRESHOLD_STEPS} candidate "
            f"thresholds (0.01 → 0.99). F1 balances Precision "
            f"({best_row['precision']:.3f}) and Recall ({best_row['recall']:.3f}) "
            f"equally, making it the most robust single-number choice for "
            f"classification tasks.{imbalance_note}"
        )
    else:  # recall_priority
        return (
            f"**Recall-Priority Strategy** — the threshold **{best_threshold:.3f}**"
            f"{class_info} was chosen because the dataset is class-imbalanced "
            f"(minority = {minority_pct:.1f}%). "
            f"Among all thresholds whose F1 remains within "
            f"{int(_RECALL_F1_SLACK * 100)}% of the global maximum, this "
            f"threshold maximises Recall ({best_row['recall']:.3f}), minimising "
            f"false negatives — which is critical when missing a positive case "
            f"has a higher cost than a false alarm. "
            f"Precision at this threshold is {best_row['precision']:.3f} and "
            f"Accuracy is {best_row['accuracy']:.3f}."
        )


# ── Public API ────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_test: Any,
    y_pred_proba: Any,
    strategy: str = "auto",
) -> Dict[str, Any]:
    """
    Automatically determine the optimal probability threshold for a binary
    classification model.

    Parameters
    ----------
    y_test       : array-like of true labels (int, str, or bool)
    y_pred_proba : array-like of predicted probabilities for the **positive**
                   class (output of model.predict_proba(X)[:, 1])
    strategy     : "auto" | "f1" | "recall_priority"
                   "auto" picks strategy based on class balance.

    Returns
    -------
    dict with keys:
        best_threshold      float
        strategy_used       str
        precision           float
        recall              float
        f1_score            float
        accuracy            float
        confusion_matrix    List[List[int]]
        threshold_curve     List[{threshold, precision, recall, f1, accuracy}]
        class_labels        List[str]   (original class names)
        minority_pct        float
        is_imbalanced       bool
        explanation         str
        recall_priority     dict | None  (populated when auto+imbalanced)
    """
    # ── Validate inputs ───────────────────────────────────────────────────
    y_test  = np.asarray(y_test)
    y_prob  = np.asarray(y_pred_proba, dtype=np.float64).ravel()

    if len(y_test) == 0 or len(y_prob) == 0:
        raise ValueError("y_test and y_pred_proba must not be empty.")
    if len(y_test) != len(y_prob):
        raise ValueError(
            f"Length mismatch: y_test has {len(y_test)} samples, "
            f"y_pred_proba has {len(y_prob)}."
        )
    if not np.all(np.isfinite(y_prob)):
        y_prob = np.where(np.isfinite(y_prob), y_prob, 0.5)

    # ── Encode labels ─────────────────────────────────────────────────────
    y_enc, le = _encode_labels(y_test)
    class_labels: List[str] = (
        le.classes_.tolist() if le is not None
        else ["0", "1"]
    )

    # ── Class imbalance check ─────────────────────────────────────────────
    min_pct      = _minority_pct(y_enc)
    is_imbalanced = min_pct < _IMBALANCE_CUTOFF

    # ── Sweep thresholds ──────────────────────────────────────────────────
    curve = _sweep_thresholds(y_enc, y_prob)

    # ── Select best threshold ─────────────────────────────────────────────
    max_f1          = max(r["f1"] for r in curve)
    f1_idx          = _best_f1_threshold(curve)
    recall_idx      = _best_recall_priority_threshold(curve, max_f1)

    # Determine active strategy
    if strategy == "auto":
        active_strategy = "recall_priority" if is_imbalanced else "f1"
    else:
        active_strategy = strategy

    primary_idx = recall_idx if active_strategy == "recall_priority" else f1_idx
    best_row    = curve[primary_idx]
    best_thresh = best_row["threshold"]

    # ── Confusion matrix at best threshold ───────────────────────────────
    cm = _confusion_matrix_at(y_enc, y_prob, best_thresh)

    # ── Build explanation ─────────────────────────────────────────────────
    explanation = _build_explanation(
        best_thresh, active_strategy, min_pct, best_row, le
    )

    # ── Recall-priority secondary result (when auto & imbalanced) ─────────
    recall_priority_result: Optional[Dict[str, Any]] = None
    if active_strategy == "recall_priority" and f1_idx != recall_idx:
        f1_row = curve[f1_idx]
        recall_priority_result = {
            "f1_best_threshold": f1_row["threshold"],
            "f1_best_f1":        round(f1_row["f1"], 4),
            "f1_best_precision": round(f1_row["precision"], 4),
            "f1_best_recall":    round(f1_row["recall"], 4),
            "note": (
                f"Pure F1-maximisation would choose threshold "
                f"{f1_row['threshold']:.3f} (F1={f1_row['f1']:.3f}). "
                f"The recall-priority threshold {best_thresh:.3f} sacrifices "
                f"{(f1_row['f1'] - best_row['f1']):.3f} F1 points to gain "
                f"{(best_row['recall'] - f1_row['recall']):.3f} Recall points."
            ),
        }

    # ── Downsample curve for frontend (keep at most 100 points) ──────────
    step = max(1, len(curve) // 100)
    curve_display = curve[::step]

    return {
        "best_threshold"    : round(best_thresh, 4),
        "strategy_used"     : active_strategy,
        "precision"         : round(best_row["precision"], 4),
        "recall"            : round(best_row["recall"],    4),
        "f1_score"          : round(best_row["f1"],        4),
        "accuracy"          : round(best_row["accuracy"],  4),
        "confusion_matrix"  : cm,
        "threshold_curve"   : curve_display,
        "class_labels"      : class_labels,
        "minority_pct"      : round(min_pct, 2),
        "is_imbalanced"     : is_imbalanced,
        "explanation"       : explanation,
        "recall_priority"   : recall_priority_result,
        "n_samples"         : int(len(y_test)),
        "n_thresholds_swept": _THRESHOLD_STEPS,
    }
