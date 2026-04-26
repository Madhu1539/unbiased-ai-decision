"""
evaluation.py  —  Production-grade Model Evaluation Pipeline

EvaluationPipeline (RECOMMENDED)
---------------------------------
  pipeline = EvaluationPipeline(model, preprocessor, threshold)
  result   = pipeline.run(X_test_raw, y_test, debug=False)

Centralized Prediction Rules (STRICT)
--------------------------------------
  1. preprocessor.transform(X_test)         ← NEVER fit_transform
  2. y_prob = model.predict_proba(X)[:, 1]  ← NEVER model.predict()
  3. y_pred = (y_prob >= threshold).astype(int)
  4. All metrics from sklearn (zero_division=0)
  5. ROC-AUC and log_loss ALWAYS use y_prob, NEVER y_pred
  6. SAME threshold in predictions, confusion matrix, error analysis
  7. Sanity checks: metrics ∈ [0,1], probs ∈ [0,1], no NaN

Public API
----------
  EvaluationPipeline                          ← recommended
  compute_threshold_metrics(y_test, y_prob, threshold, debug) → dict
  get_y_prob_from_model(model, X_safe)                        → ndarray | None
  validate_data_integrity(y_test, y_prob)                     → dict
  validate_threshold_behavior(y_prob, threshold, ...)         → List[str]
  classification_metrics(y_test, y_pred, model, X_test)       → dict
  regression_metrics(y_test, y_pred)                          → dict
  actual_vs_predicted(y_test, y_pred)                         → dict
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Mismatch tolerance between manual and sklearn metrics ─────────────────────
_MISMATCH_TOL = 1e-4

# ── Minimum test-set size for reliable evaluation ────────────────────────────
_MIN_SAMPLES = 50

# ── Imbalance threshold (minority < this % → warning) ────────────────────────
_IMBALANCE_PCT = 20.0


# ══════════════════════════════════════════════════════════════════════════════
#  EvaluationPipeline  — production-grade, strict, reproducible
# ══════════════════════════════════════════════════════════════════════════════

class EvaluationPipeline:
    """
    Production-grade evaluation pipeline for sklearn binary classifiers.

    Strict rules enforced
    ─────────────────────
    • preprocessor.transform() only — never fit_transform on test data
    • model.predict_proba()[:, 1] only — never model.predict()
    • Single threshold used consistently across ALL outputs
    • All metrics computed via sklearn with zero_division=0
    • Sanity checks raise ValueError on invalid metric ranges or NaN

    Parameters
    ----------
    model        : fitted sklearn-compatible classifier
    preprocessor : fitted sklearn Pipeline/ColumnTransformer (or None)
    threshold    : float  classification cutoff (0.0 – 1.0)
    """

    def __init__(self, model: Any, preprocessor: Any = None, threshold: float = 0.5):
        if model is None:
            raise ValueError("EvaluationPipeline: model must not be None.")
        self.model        = model
        self.preprocessor = preprocessor
        self.threshold    = max(0.0, min(1.0, float(threshold)))

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        X_test: Any,
        y_test: Any,
        *,
        debug: bool = False,
        sweep_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline.

        Parameters
        ----------
        X_test      : array-like  – raw (un-preprocessed) test features
        y_test      : array-like  – true labels
        debug       : bool        – if True, include 50-sample diagnostic outputs
        sweep_steps : int         – number of threshold steps for PR/F1 curves

        Returns
        -------
        dict – complete evaluation result (see _build_result for keys)
        """
        y_test = np.asarray(y_test)
        n = len(y_test)

        # ── STEP 1: Preprocess (transform-only, NEVER fit) ───────────────────
        X_transformed = self._transform(X_test)

        # ── STEP 2: Get probabilities (predict_proba only) ───────────────────
        y_prob = self._get_proba(X_transformed)

        # ── STEP 3: Sanity-check inputs ──────────────────────────────────────
        structural_warnings = self._validate_inputs(y_test, y_prob, n)

        # ── STEP 4: Apply threshold → y_pred ─────────────────────────────────
        y_pred = (y_prob >= self.threshold).astype(int)

        # ── STEP 5: Encode labels → {0, 1} ───────────────────────────────────
        y_enc, y_pred_enc, le = _encode_binary(y_test, y_pred)
        class_labels: List[str] = le.classes_.tolist() if le is not None else ["0", "1"]

        # ── STEP 6: Confusion matrix (sklearn, labels=[0,1]) ─────────────────
        cm_arr = confusion_matrix(y_enc, y_pred_enc, labels=[0, 1])
        if cm_arr.shape == (2, 2):
            tn, fp, fn, tp = (int(x) for x in cm_arr.ravel())
        else:
            tp = int(np.sum((y_enc == 1) & (y_pred_enc == 1)))
            tn = int(np.sum((y_enc == 0) & (y_pred_enc == 0)))
            fp = int(np.sum((y_enc == 0) & (y_pred_enc == 1)))
            fn = int(np.sum((y_enc == 1) & (y_pred_enc == 0)))

        # MANDATORY: TP + TN + FP + FN == n
        cm_sum = tp + tn + fp + fn
        if cm_sum != n:
            raise ValueError(
                f"Confusion matrix sum {cm_sum} != n_samples {n}. "
                "Data integrity violation."
            )

        # ── STEP 7: sklearn metrics (sole source of truth) ───────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            accuracy  = float(accuracy_score(y_enc, y_pred_enc))
            precision = float(precision_score(y_enc, y_pred_enc, average="binary", zero_division=0))
            recall    = float(recall_score(y_enc, y_pred_enc, average="binary", zero_division=0))
            f1        = float(f1_score(y_enc, y_pred_enc, average="binary", zero_division=0))
            sk_report = classification_report(
                y_enc, y_pred_enc, target_names=class_labels, zero_division=0
            )

        # ── STEP 8: Probability-based metrics (y_prob, NOT y_pred) ───────────
        roc_auc_val   = None
        roc_curve_dict = None
        logloss_val   = None
        pr_curve_dict = None

        n_unique = len(np.unique(y_enc))
        if n_unique == 2:
            try:
                roc_auc_val = float(roc_auc_score(y_enc, y_prob))
                fpr, tpr, _ = roc_curve(y_enc, y_prob)
                step = max(1, len(fpr) // 150)
                roc_curve_dict = {
                    "fpr"   : fpr[::step].tolist(),
                    "tpr"   : tpr[::step].tolist(),
                    "auc"   : round(roc_auc_val, 6),
                    "labels": class_labels,
                }
            except Exception as e:
                logger.debug("[EvaluationPipeline] ROC skipped: %s", e)

            try:
                logloss_val = float(log_loss(y_enc, y_prob))
            except Exception as e:
                logger.debug("[EvaluationPipeline] log_loss skipped: %s", e)

            try:
                prec_arr, rec_arr, thr_arr = precision_recall_curve(y_enc, y_prob)
                step_pr = max(1, len(prec_arr) // 150)
                pr_curve_dict = {
                    "precision": prec_arr[::step_pr].tolist(),
                    "recall"   : rec_arr[::step_pr].tolist(),
                    "auc"      : float(np.trapz(prec_arr[::-1], rec_arr[::-1])),
                }
            except Exception as e:
                logger.debug("[EvaluationPipeline] PR curve skipped: %s", e)

        # ── STEP 9: Sanity checks (raises on violation) ───────────────────────
        self._sanity_check_metrics(accuracy, precision, recall, f1, roc_auc_val, y_prob)

        # ── STEP 10: Threshold sweep for curves ───────────────────────────────
        threshold_curve = self._sweep_thresholds(y_enc, y_prob, sweep_steps)

        # ── STEP 11: Class distribution + imbalance ───────────────────────────
        unique_cls, counts = np.unique(y_enc, return_counts=True)
        class_dist = {}
        imbalance_warnings: List[str] = []
        for c, cnt in zip(unique_cls, counts):
            lbl = class_labels[int(c)] if int(c) < len(class_labels) else str(c)
            pct = float(cnt) / n * 100
            class_dist[lbl] = {"count": int(cnt), "pct": round(pct, 2)}
            if pct < _IMBALANCE_PCT:
                imbalance_warnings.append(
                    f"Class '{lbl}' represents only {pct:.1f}% of test samples. "
                    "Accuracy may be misleading — use precision, recall and F1."
                )

        # ── STEP 12: Threshold-behaviour validation ───────────────────────────
        thresh_warnings = validate_threshold_behavior(y_prob, self.threshold, y_enc, y_pred)

        # ── STEP 13: All warnings consolidated ───────────────────────────────
        all_warnings = structural_warnings + imbalance_warnings + thresh_warnings

        # ── STEP 14: Debug payload + root-cause detection ────────────────────
        debug_info = None
        if debug:
            n_dbg = min(50, n)

            # ── Diagnostic values ────────────────────────────────────────────
            y_pred_unique = np.unique(y_pred).tolist()
            y_proba_min   = float(np.min(y_prob))
            y_proba_max   = float(np.max(y_prob))
            y_proba_mean  = float(np.mean(y_prob))
            y_proba_var   = float(np.var(y_prob))

            # ── Root-cause auto-detection ─────────────────────────────────────
            root_causes: List[str] = []
            suggestions: List[str] = []
            _LOW_VAR = 0.01

            if len(y_pred_unique) <= 1:
                root_causes.append("Model predicting single class")
                logger.warning(
                    "[EvaluationPipeline DEBUG] Model predicting single class: "
                    "y_pred unique=%s", y_pred_unique,
                )
                suggestions.append("Lower threshold to 0.3 and recompute metrics")
                suggestions.append("Retrain with class_weight='balanced' or SMOTE")

            if y_proba_max < self.threshold:
                root_causes.append(
                    f"Threshold too high for probability distribution "
                    f"(max_proba={y_proba_max:.4f} < threshold={self.threshold})"
                )
                logger.warning(
                    "[EvaluationPipeline DEBUG] Threshold too high: "
                    "max(y_proba)=%.4f < threshold=%.4f",
                    y_proba_max, self.threshold,
                )
                suggestions.append(
                    f"Try threshold=0.3 (current max_proba={y_proba_max:.4f})"
                )

            if y_proba_var < _LOW_VAR:
                root_causes.append(
                    f"Model outputs nearly constant probabilities "
                    f"(var={y_proba_var:.6f})"
                )
                logger.warning(
                    "[EvaluationPipeline DEBUG] Near-constant probabilities: "
                    "var(y_proba)=%.8f", y_proba_var,
                )
                suggestions.append(
                    "Feature engineering or model upgrade (Random Forest / GBM)"
                )

            # Class imbalance check from class_dist computed in STEP 11
            _minority_pct = (
                min(v["pct"] for v in class_dist.values())
                if class_dist else 100.0
            )
            if _minority_pct < 20.0:
                root_causes.append(
                    f"Class imbalance: minority class is {_minority_pct:.1f}% "
                    f"of test set (threshold 80/20 exceeded)"
                )
                suggestions.append(
                    "Retrain with class_weight='balanced' or apply SMOTE to training data"
                )

            # ── Auto-fix: recompute metrics at threshold=0.3 when f1==0 ──────
            fixed_metrics: Optional[Dict] = None
            if f1 == 0.0 and (len(y_pred_unique) <= 1 or y_proba_max < self.threshold):
                _alt_thresh = 0.3
                _yp_alt = (y_prob >= _alt_thresh).astype(int)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _f1_alt   = float(f1_score(y_enc, _yp_alt, average="binary", zero_division=0))
                    _prec_alt = float(precision_score(y_enc, _yp_alt, average="binary", zero_division=0))
                    _rec_alt  = float(recall_score(y_enc, _yp_alt, average="binary", zero_division=0))
                    _acc_alt  = float(accuracy_score(y_enc, _yp_alt))
                if _f1_alt > 0:
                    fixed_metrics = {
                        "threshold" : _alt_thresh,
                        "accuracy"  : round(_acc_alt,  6),
                        "precision" : round(_prec_alt, 6),
                        "recall"    : round(_rec_alt,  6),
                        "f1_score"  : round(_f1_alt,   6),
                        "note"      : f"Recomputed at threshold={_alt_thresh} (auto-fix)",
                    }
                    logger.info(
                        "[EvaluationPipeline DEBUG] auto-fix threshold=%.2f -> "
                        "precision=%.4f recall=%.4f f1=%.4f",
                        _alt_thresh, _prec_alt, _rec_alt, _f1_alt,
                    )

            debug_info = {
                "threshold"        : self.threshold,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                # Sample arrays (up to 50 rows)
                "y_test"           : [_to_serialisable(v) for v in y_test[:n_dbg]],
                "y_pred"           : y_pred[:n_dbg].tolist(),
                "y_proba"          : np.round(y_prob[:n_dbg], 4).tolist(),
                # Probability statistics
                "y_pred_unique"    : y_pred_unique,
                "y_proba_min"      : round(y_proba_min,  6),
                "y_proba_max"      : round(y_proba_max,  6),
                "y_proba_mean"     : round(y_proba_mean, 6),
                "y_proba_variance" : round(y_proba_var,  8),
                # Root-cause diagnostics
                "root_causes"      : root_causes,
                "suggestions"      : suggestions,
                "fixed_metrics"    : fixed_metrics,
            }
            logger.info(
                "[EvaluationPipeline DEBUG] threshold=%.4f  "
                "TP=%d FP=%d TN=%d FN=%d  "
                "Accuracy=%.4f Precision=%.4f Recall=%.4f F1=%.4f  "
                "ROC-AUC=%s  log_loss=%s  "
                "y_pred_unique=%s  proba_min=%.4f  proba_max=%.4f  proba_var=%.6f  "
                "root_causes=%s",
                self.threshold, tp, fp, tn, fn,
                accuracy, precision, recall, f1,
                f"{roc_auc_val:.4f}" if roc_auc_val is not None else "N/A",
                f"{logloss_val:.4f}" if logloss_val is not None else "N/A",
                y_pred_unique, y_proba_min, y_proba_max, y_proba_var,
                root_causes,
            )

        return {
            # ── Threshold (single source) ──────────────────────────────────
            "threshold"          : round(self.threshold, 4),
            # ── sklearn metrics ────────────────────────────────────────────
            "accuracy"           : round(accuracy,  6),
            "precision"          : round(precision, 6),
            "recall"             : round(recall,    6),
            "f1_score"           : round(f1,        6),
            # ── Probability-based ──────────────────────────────────────────
            "roc_auc"            : round(roc_auc_val, 6) if roc_auc_val is not None else None,
            "log_loss"           : round(logloss_val, 6) if logloss_val is not None else None,
            # ── Confusion matrix ───────────────────────────────────────────
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "confusion_matrix"   : cm_arr.tolist(),
            # ── Curves ────────────────────────────────────────────────────
            "roc_curve"          : roc_curve_dict,
            "pr_curve"           : pr_curve_dict,
            "threshold_curve"    : threshold_curve,
            # ── Metadata ──────────────────────────────────────────────────
            "class_labels"       : class_labels,
            "class_distribution" : class_dist,
            "n_samples"          : n,
            "sklearn_report"     : sk_report,
            # ── Warnings ──────────────────────────────────────────────────
            "threshold_warnings" : thresh_warnings,
            "imbalance_warnings" : imbalance_warnings,
            "warnings"           : all_warnings,
            # ── Debug ──────────────────────────────────────────────────────
            "debug_info"         : debug_info,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _transform(self, X_test: Any) -> np.ndarray:
        """Apply preprocessor.transform() only — NEVER fit_transform."""
        if self.preprocessor is None:
            # No preprocessor: assume X_test is already numeric
            logger.debug("[EvaluationPipeline] No preprocessor — using X_test as-is.")
            _cast = _safe_cast_X(X_test)
            return _cast if _cast is not None else np.asarray(X_test, dtype=object)

        if not hasattr(self.preprocessor, "transform"):
            raise ValueError(
                "EvaluationPipeline: preprocessor must have a .transform() method. "
                "Passing an unfitted object or using fit_transform on test data is forbidden."
            )
        try:
            import pandas as pd
            X_arr = self.preprocessor.transform(X_test)
            # Cast to float64 if possible
            try:
                return np.asarray(X_arr, dtype=np.float64)
            except ValueError:
                return np.asarray(
                    pd.DataFrame(X_arr)
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0)
                    .to_numpy(dtype=np.float64)
                )
        except Exception as e:
            raise RuntimeError(
                f"EvaluationPipeline: preprocessor.transform() failed: {e}. "
                "Ensure the preprocessor was fitted on training data only."
            ) from e

    def _get_proba(self, X_safe: np.ndarray) -> np.ndarray:
        """Extract P(positive class) via predict_proba only — NEVER predict()."""
        y_prob = get_y_prob_from_model(self.model, X_safe)
        if y_prob is None:
            raise ValueError(
                "EvaluationPipeline: model does not support predict_proba() or "
                "decision_function(). Evaluation requires probability outputs."
            )
        y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
        if np.any(~np.isfinite(y_prob)):
            logger.warning("[EvaluationPipeline] Non-finite probabilities replaced with 0.5")
            y_prob = np.where(np.isfinite(y_prob), y_prob, 0.5)
        return y_prob

    def _validate_inputs(
        self, y_test: np.ndarray, y_prob: np.ndarray, n: int
    ) -> List[str]:
        """Structural checks. Returns list of warning strings."""
        w: List[str] = []
        if n < _MIN_SAMPLES:
            w.append(
                f"Evaluation may be unreliable: only {n} test samples "
                f"(recommended minimum: {_MIN_SAMPLES})."
            )
        if len(y_prob) != n:
            raise ValueError(
                f"Length mismatch: y_test has {n} rows but y_prob has {len(y_prob)}."
            )
        out_of_range = int(np.sum((y_prob < 0) | (y_prob > 1)))
        if out_of_range > 0:
            raise ValueError(
                f"{out_of_range} probability values outside [0, 1]. "
                "Probabilities must be in [0, 1]."
            )
        return w

    @staticmethod
    def _sanity_check_metrics(
        accuracy: float, precision: float, recall: float, f1: float,
        roc_auc: Optional[float], y_prob: np.ndarray,
    ) -> None:
        """Raise ValueError if any metric is outside its valid range."""
        checks = {
            "accuracy" : accuracy,
            "precision": precision,
            "recall"   : recall,
            "f1"       : f1,
        }
        if roc_auc is not None:
            checks["roc_auc"] = roc_auc
        for name, val in checks.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Sanity check failed: {name}={val:.6f} is outside [0, 1]. "
                    "This indicates a pipeline bug."
                )
        if np.any(np.isnan(y_prob)):
            raise ValueError("Sanity check failed: NaN values detected in predicted probabilities.")

    @staticmethod
    def _sweep_thresholds(
        y_enc: np.ndarray, y_prob: np.ndarray, steps: int
    ) -> List[Dict[str, float]]:
        """Return precision/recall/f1/accuracy at `steps` evenly-spaced thresholds."""
        curve = []
        for t in np.linspace(0.0, 1.0, steps):
            yp = (y_prob >= t).astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                curve.append({
                    "threshold": round(float(t), 4),
                    "precision": round(float(precision_score(y_enc, yp, average="binary", zero_division=0)), 4),
                    "recall"   : round(float(recall_score(y_enc, yp, average="binary", zero_division=0)), 4),
                    "f1"       : round(float(f1_score(y_enc, yp, average="binary", zero_division=0)), 4),
                    "accuracy" : round(float(accuracy_score(y_enc, yp)), 4),
                })
        return curve




# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_serialisable(v: Any) -> Any:
    """
    Convert a single label value to a JSON-serialisable Python type.
    Numeric → float, String/other → str.
    Prevents map(float, y) from crashing on 'No'/'Yes' labels.
    """
    try:
        f = float(v)
        return f if np.isfinite(f) else str(v)
    except (ValueError, TypeError):
        return str(v)


def _encode_binary(y_test: np.ndarray, y_pred: np.ndarray):
    """
    Encode both y_test and y_pred to integer {0, 1} if they contain string
    labels (e.g. 'No'/'Yes').  LabelEncoder maps alphabetically: No→0, Yes→1.
    Returns (y_test_enc, y_pred_enc, label_encoder_or_None).

    Bug-fix: when y_test has string labels ('No'/'Yes') but y_pred is already
    an integer array [0, 1] (e.g. from `(y_prob >= threshold).astype(int)`),
    the old code cast integers to strings '0'/'1' which were never found in
    le.classes_ (['No','Yes']), silently defaulting every prediction to 0.
    This caused TP=0 → precision=recall=f1=0 despite accuracy > 0.
    """
    unique = np.unique(y_test)
    if set(unique.tolist()).issubset({0, 1, 0.0, 1.0}):
        return y_test.astype(int), y_pred.astype(int), None

    le = LabelEncoder()
    le.fit(y_test.astype(str))
    y_test_enc = le.transform(y_test.astype(str))

    # ── Detect if y_pred is already integer-encoded {0 … n_classes-1} ──────
    # This happens when y_pred = (y_prob >= threshold).astype(int), which
    # produces integers [0, 1] that are NOT the string labels 'No'/'Yes'.
    # In that case we must NOT run them through the string-label LabelEncoder.
    n_classes = len(le.classes_)
    try:
        y_pred_float = np.asarray(y_pred, dtype=float)
        y_pred_int   = y_pred_float.astype(int)
        already_encoded = (
            np.all(y_pred_float == y_pred_int)          # whole numbers only
            and int(y_pred_int.min()) >= 0              # non-negative
            and int(y_pred_int.max()) < n_classes       # within class range
        )
        if already_encoded:
            return y_test_enc, y_pred_int, le
    except (ValueError, TypeError):
        pass

    # ── y_pred contains string labels — encode through LabelEncoder ─────────
    y_pred_str = np.asarray(y_pred).astype(str)
    y_pred_enc = np.array([
        le.transform([v])[0] if v in le.classes_ else 0
        for v in y_pred_str
    ], dtype=int)
    return y_test_enc, y_pred_enc, le


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC: Centralized threshold-controlled metric computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_threshold_metrics(
    y_test: Any,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Centralized Prediction Pipeline
    ================================
    Step 1: Apply threshold to probabilities → y_pred (0/1 integers)
    Step 2: Compute TP/FP/TN/FN from confusion_matrix().ravel()
    Step 3: Compute ALL metrics MANUALLY from TP/FP/TN/FN
    Step 4: Cross-verify with sklearn — warn loudly on any mismatch
    Step 5: Compute ROC-AUC ALWAYS from y_prob (never from y_pred)

    Parameters
    ----------
    y_test    : array-like  – true labels (int, str, or bool)
    y_prob    : array-like  – P(positive class) from predict_proba()[:, 1]
    threshold : float       – classification cutoff (default 0.5)
    debug     : bool        – if True, print TP/FP/TN/FN + sample preds

    Returns
    -------
    dict with keys:
        threshold, accuracy, precision, recall, f1_score,
        tp, fp, tn, fn, confusion_matrix (List[List[int]]),
        roc_auc (float | None), roc_curve (dict | None),
        class_labels (List[str]),
        class_distribution (dict),
        sklearn_report (str),
        debug_info (dict | None)
    """
    y_test = np.asarray(y_test)
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    if len(y_test) == 0:
        raise ValueError("y_test must not be empty.")
    if len(y_prob) != len(y_test):
        raise ValueError(
            f"Length mismatch: y_test={len(y_test)}, y_prob={len(y_prob)}"
        )

    # ── Replace non-finite probabilities ─────────────────────────────────────
    if not np.all(np.isfinite(y_prob)):
        logger.warning("[Evaluation] Non-finite values in y_prob — replacing with 0.5")
        y_prob = np.where(np.isfinite(y_prob), y_prob, 0.5)

    threshold = float(threshold)
    threshold = max(0.0, min(1.0, threshold))

    # ── STEP 1: Apply threshold → y_pred (always 0/1 integers) ──────────────
    y_pred_bin = (y_prob >= threshold).astype(int)

    # ── Encode y_test to {0,1} for sklearn internals ─────────────────────────
    y_test_enc, y_pred_enc, le = _encode_binary(y_test, y_pred_bin)
    class_labels: List[str] = (
        le.classes_.tolist() if le is not None else ["0", "1"]
    )

    # ── STEP 2: Confusion matrix → TP/FP/TN/FN ──────────────────────────────
    # confusion_matrix returns [[TN, FP], [FN, TP]] for binary classification
    cm_arr = confusion_matrix(y_test_enc, y_pred_enc, labels=[0, 1])
    cm_list = cm_arr.tolist()

    if cm_arr.shape == (2, 2):
        tn, fp, fn, tp = cm_arr.ravel()
        tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    else:
        # Fallback for degenerate single-class predictions
        tp = int(np.sum((y_test_enc == 1) & (y_pred_enc == 1)))
        tn = int(np.sum((y_test_enc == 0) & (y_pred_enc == 0)))
        fp = int(np.sum((y_test_enc == 0) & (y_pred_enc == 1)))
        fn = int(np.sum((y_test_enc == 1) & (y_pred_enc == 0)))

    # ── STEP 3: Manual metric formulas from TP/FP/TN/FN ─────────────────────
    total = tp + tn + fp + fn

    accuracy  = (tp + tn) / total                            if total > 0       else 0.0
    precision = tp / (tp + fp)                               if (tp + fp) > 0   else 0.0
    recall    = tp / (tp + fn)                               if (tp + fn) > 0   else 0.0
    f1        = (2 * precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    # ── STEP 4: Cross-verify with sklearn ────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sk_accuracy  = float(accuracy_score(y_test_enc, y_pred_enc))
        sk_precision = float(precision_score(y_test_enc, y_pred_enc, average="binary", zero_division=0))
        sk_recall    = float(recall_score(y_test_enc, y_pred_enc, average="binary", zero_division=0))
        sk_f1        = float(f1_score(y_test_enc, y_pred_enc, average="binary", zero_division=0))
        sk_report    = classification_report(
            y_test_enc, y_pred_enc,
            target_names=class_labels,
            zero_division=0,
        )

    _check_mismatch("accuracy",  accuracy,  sk_accuracy,  threshold)
    _check_mismatch("precision", precision, sk_precision, threshold)
    _check_mismatch("recall",    recall,    sk_recall,    threshold)
    _check_mismatch("f1_score",  f1,        sk_f1,        threshold)

    # ── STEP 5: ROC-AUC — ALWAYS from y_prob, NEVER from y_pred ─────────────
    roc_auc_val:  Optional[float]     = None
    roc_curve_dict: Optional[Dict]   = None

    try:
        roc_auc_val = float(roc_auc_score(y_test_enc, y_prob))
        fpr, tpr, _ = roc_curve(y_test_enc, y_prob)
        step = max(1, len(fpr) // 100)
        roc_curve_dict = {
            "fpr"   : fpr[::step].tolist(),
            "tpr"   : tpr[::step].tolist(),
            "auc"   : roc_auc_val,
            "labels": class_labels,
        }
    except Exception as roc_err:
        logger.debug("[Evaluation] ROC skipped: %s", roc_err)

    # ── Class distribution ────────────────────────────────────────────────────
    unique_cls, counts = np.unique(y_test_enc, return_counts=True)
    class_dist = {
        class_labels[int(c)] if int(c) < len(class_labels) else str(c): {
            "count": int(cnt),
            "pct"  : round(float(cnt) / len(y_test_enc) * 100, 2),
        }
        for c, cnt in zip(unique_cls, counts)
    }

    # ── DEBUG output ──────────────────────────────────────────────────────────
    debug_info = None
    if debug:
        n_sample = min(10, len(y_test))
        sample_actual    = y_test[:n_sample].tolist()
        sample_prob      = y_prob[:n_sample].round(4).tolist()
        sample_pred      = y_pred_bin[:n_sample].tolist()

        debug_info = {
            "threshold"      : threshold,
            "tp"             : tp,
            "fp"             : fp,
            "tn"             : tn,
            "fn"             : fn,
            "sample_actual"  : [_to_serialisable(v) for v in sample_actual],
            "sample_prob"    : sample_prob,
            "sample_pred"    : sample_pred,
        }

        logger.info(
            "[DEBUG] threshold=%.4f  TP=%d  FP=%d  TN=%d  FN=%d\n"
            "        Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f\n"
            "        Sample actual:  %s\n"
            "        Sample prob:    %s\n"
            "        Sample pred:    %s",
            threshold, tp, fp, tn, fn,
            accuracy, precision, recall, f1,
            sample_actual, sample_prob, sample_pred,
        )

    # ── STEP 6: Threshold edge-case validation ───────────────────────────────
    thresh_warnings = validate_threshold_behavior(
        y_prob, threshold, y_test_enc, y_pred_bin
    )

    # ── Performance guard ─────────────────────────────────────────────────────
    perf_warnings: List[str] = []
    if len(y_test) > 100_000:
        perf_warnings.append(
            f"Large dataset ({len(y_test):,} samples). Metric computation may be slow. "
            "Consider using sampled evaluation for interactive use."
        )

    return {
        "threshold"         : round(threshold, 4),
        # Manual metrics (PRIMARY)
        "accuracy"          : round(accuracy,  6),
        "precision"         : round(precision, 6),
        "recall"            : round(recall,    6),
        "f1_score"          : round(f1,        6),
        # Raw confusion matrix components
        "tp"                : tp,
        "fp"                : fp,
        "tn"                : tn,
        "fn"                : fn,
        "confusion_matrix"  : cm_list,
        # sklearn cross-verification values
        "sklearn_accuracy"  : round(sk_accuracy,  6),
        "sklearn_precision" : round(sk_precision, 6),
        "sklearn_recall"    : round(sk_recall,    6),
        "sklearn_f1"        : round(sk_f1,        6),
        "sklearn_report"    : sk_report,
        # ROC — probability based (threshold-independent)
        "roc_auc"           : round(roc_auc_val, 6) if roc_auc_val is not None else None,
        "roc_curve"         : roc_curve_dict,
        # Meta
        "class_labels"      : class_labels,
        "class_distribution": class_dist,
        "n_samples"         : int(len(y_test)),
        "debug_info"        : debug_info,
        # Validation
        "threshold_warnings": thresh_warnings,
        "perf_warnings"     : perf_warnings,
    }


def _check_mismatch(name: str, manual: float, sklearn_val: float, threshold: float):
    """Warn loudly if manual and sklearn values differ beyond tolerance."""
    diff = abs(manual - sklearn_val)
    if diff > _MISMATCH_TOL:
        logger.warning(
            "[Evaluation] ⚠️  METRIC MISMATCH '%s' at threshold=%.4f: "
            "manual=%.6f  sklearn=%.6f  diff=%.6f",
            name, threshold, manual, sklearn_val, diff,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Backward-compatible wrappers
# ══════════════════════════════════════════════════════════════════════════════

def classification_metrics(
    y_test, y_pred, model=None, X_test=None, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper.

    If model supports predict_proba AND X_test is provided → uses the full
    centralized pipeline (y_prob → threshold → manual metrics).

    Otherwise (model.predict() only) → falls back to direct sklearn computation
    with binary averaging, labelling the result clearly.
    """
    # ── Try centralized pipeline first ───────────────────────────────────────
    if model is not None and X_test is not None and hasattr(model, "predict_proba"):
        try:
            X_safe = _safe_cast_X(X_test)
            if X_safe is not None:
                y_prob = model.predict_proba(X_safe)[:, 1]
                result = compute_threshold_metrics(
                    y_test, y_prob, threshold=threshold
                )
                # Return compatible shape (keeps roc_curve key)
                return result
        except Exception as e:
            logger.warning(
                "[Evaluation] Centralized pipeline failed, falling back: %s", e
            )

    # ── Fallback: direct sklearn from y_pred ─────────────────────────────────
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    y_test_enc, y_pred_enc, le = _encode_binary(y_test, y_pred)

    cm_arr  = confusion_matrix(y_test_enc, y_pred_enc, labels=[0, 1])
    cm_list = cm_arr.tolist()

    if cm_arr.shape == (2, 2):
        tn, fp, fn, tp_val = cm_arr.ravel()
        tn, fp, fn, tp_val = int(tn), int(fp), int(fn), int(tp_val)
    else:
        tp_val = fp = tn = fn = 0

    total     = tp_val + tn + fp + fn
    accuracy  = (tp_val + tn) / total if total > 0 else 0.0
    precision = tp_val / (tp_val + fp) if (tp_val + fp) > 0 else 0.0
    recall    = tp_val / (tp_val + fn) if (tp_val + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    class_labels = le.classes_.tolist() if le is not None else ["0", "1"]

    metrics: Dict[str, Any] = {
        "threshold"        : threshold,
        "accuracy"         : round(accuracy,  6),
        "precision"        : round(precision, 6),
        "recall"           : round(recall,    6),
        "f1_score"         : round(f1,        6),
        "tp"               : tp_val,
        "fp"               : fp,
        "tn"               : tn,
        "fn"               : fn,
        "confusion_matrix" : cm_list,
        "class_labels"     : class_labels,
        "n_samples"        : int(len(y_test)),
    }

    # ROC for binary classification (fallback: try model if available)
    unique_classes = np.unique(y_test_enc)
    if len(unique_classes) == 2 and model is not None and X_test is not None:
        X_safe = _safe_cast_X(X_test)
        if X_safe is not None:
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_safe)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_prob = model.decision_function(X_safe)
                else:
                    y_prob = None

                if y_prob is not None:
                    roc_auc = float(roc_auc_score(y_test_enc, y_prob))
                    fpr, tpr, _ = roc_curve(y_test_enc, y_prob)
                    step = max(1, len(fpr) // 100)
                    metrics["roc_auc"] = round(roc_auc, 6)
                    metrics["roc_curve"] = {
                        "fpr"   : fpr[::step].tolist(),
                        "tpr"   : tpr[::step].tolist(),
                        "auc"   : roc_auc,
                        "labels": class_labels,
                    }
            except Exception:
                pass

    return metrics


def _safe_cast_X(X_test) -> Optional[np.ndarray]:
    """Safely cast X_test to float64. Returns None on failure."""
    import pandas as pd
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


# ══════════════════════════════════════════════════════════════════════════════
#  Model compatibility layer — public API
# ══════════════════════════════════════════════════════════════════════════════

def get_y_prob_from_model(model: Any, X_safe: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract P(positive class) from *any* sklearn-compatible classifier.

    Priority
    --------
    1. predict_proba()      → already in [0, 1]
    2. decision_function()  → normalized to [0, 1] via min-max scaling
    3. None                 → model does not support probability scoring

    Notes
    -----
    - decision_function normalization handles the degenerate case where all
      scores are identical (returns 0.5 across the board).
    - The caller should raise a clear error if None is returned.
    """
    # PRIMARY: predict_proba
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_safe)[:, 1].astype(np.float64)
            # Guard against non-finite values
            if np.any(~np.isfinite(proba)):
                proba = np.where(np.isfinite(proba), proba, 0.5)
            return proba
        except Exception as e:
            logger.warning("[Evaluation] predict_proba failed (%s) — trying decision_function", e)

    # FALLBACK: decision_function (SVM without probability=True, etc.)
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_safe).astype(np.float64)
            s_min, s_max = float(scores.min()), float(scores.max())
            if s_max > s_min:
                # Min-max normalisation → [0, 1]
                proba = (scores - s_min) / (s_max - s_min)
            else:
                # Constant scores — all predictions are equally uncertain
                logger.warning(
                    "[Evaluation] decision_function returned constant scores — "
                    "falling back to p=0.5 for all samples."
                )
                proba = np.full_like(scores, 0.5)
            return proba
        except Exception as e:
            logger.warning("[Evaluation] decision_function failed: %s", e)

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Data integrity validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_data_integrity(
    y_test: Any,
    y_prob: Any,
    n_large: int = 100_000,
) -> Dict[str, Any]:
    """
    Validate inputs before computing metrics.  Returns a structured report.

    Returns
    -------
    dict:
        valid    bool
        errors   List[str]   – blocking issues that prevent computation
        warnings List[str]   – non-blocking issues
    """
    errors:   List[str] = []
    warnings: List[str] = []

    y_test = np.asarray(y_test)
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    # Blocking checks
    if len(y_test) == 0:
        errors.append("y_test is empty — no data to evaluate.")
    if len(y_prob) == 0:
        errors.append("y_prob is empty — no probability scores available.")
    if len(y_test) > 0 and len(y_prob) > 0 and len(y_test) != len(y_prob):
        errors.append(
            f"Length mismatch: y_test has {len(y_test)} samples but y_prob has "
            f"{len(y_prob)} — they must be equal."
        )

    # Non-blocking checks
    if len(y_prob) > 0:
        n_bad = int(np.sum(~np.isfinite(y_prob)))
        if n_bad > 0:
            warnings.append(
                f"{n_bad} non-finite values in y_prob (NaN/Inf) — will be replaced "
                "with 0.5 (ambiguous prediction) before metric computation."
            )

    if len(y_test) > 0:
        n_classes = len(np.unique(y_test))
        if n_classes < 2:
            warnings.append(
                f"Only {n_classes} unique class(es) in y_test — "
                "binary classification metrics require at least 2 classes."
            )
        elif n_classes > 2:
            warnings.append(
                f"Multiclass target detected ({n_classes} classes). "
                "Threshold-based metrics treat this as binary (class 1 vs rest)."
            )

        if len(y_test) > n_large:
            warnings.append(
                f"Large dataset: {len(y_test):,} samples. "
                "Consider sampled evaluation for interactive threshold exploration."
            )

    return {
        "valid"   : len(errors) == 0,
        "errors"  : errors,
        "warnings": warnings,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Threshold edge-case behaviour validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_threshold_behavior(
    y_prob: np.ndarray,
    threshold: float,
    y_test_enc: np.ndarray,
    y_pred_bin: np.ndarray,
) -> List[str]:
    """
    Verify that predictions at extreme thresholds match expected behaviour:
      - threshold ≈ 0.0  → all predictions = 1, TN = 0
      - threshold ≈ 1.0  → all predictions = 0, TP = 0

    Logs a WARNING and returns a human-readable list if behaviour deviates.
    """
    warn_list: List[str] = []

    if threshold <= 0.01:
        n_neg_pred = int(np.sum(y_pred_bin == 0))
        tn_count = int(np.sum((y_test_enc == 0) & (y_pred_bin == 0)))
        if n_neg_pred > 0:
            warn_list.append(
                f"At threshold={threshold:.3f} (≈0), expected all predictions=1 "
                f"but got {n_neg_pred} negative predictions."
            )
        if tn_count > 0:
            warn_list.append(
                f"At threshold≈0, expected TN=0 but got TN={tn_count}. "
                "Check if y_prob contains values exactly equal to 0."
            )

    if threshold >= 0.99:
        n_pos_pred = int(np.sum(y_pred_bin == 1))
        tp_count = int(np.sum((y_test_enc == 1) & (y_pred_bin == 1)))
        if n_pos_pred > 0:
            warn_list.append(
                f"At threshold={threshold:.3f} (≈1), expected all predictions=0 "
                f"but got {n_pos_pred} positive predictions."
            )
        if tp_count > 0:
            warn_list.append(
                f"At threshold≈1, expected TP=0 but got TP={tp_count}. "
                "Check if y_prob contains values exactly equal to 1."
            )

    for w in warn_list:
        logger.warning("[ThresholdValidation] %s", w)

    return warn_list


# ══════════════════════════════════════════════════════════════════════════════
#  Regression metrics (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(y_test, y_pred) -> Dict[str, Any]:
    """Compute regression evaluation metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {
        "rmse"    : rmse,
        "mae"     : float(mean_absolute_error(y_test, y_pred)),
        "r2_score": float(r2_score(y_test, y_pred)),
        "mse"     : float(mean_squared_error(y_test, y_pred)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Actual vs predicted (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def actual_vs_predicted(y_test, y_pred, max_samples: int = 200) -> Dict[str, Any]:
    """
    Return actual vs predicted values capped for frontend rendering.
    For regression: always emits float (not str) so Recharts ScatterChart
    receives proper numeric values on both axes.
    """
    n = min(len(y_test), max_samples)

    def _to_float(v) -> Any:
        """Convert to float for regression. Fallback to None on failure."""
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except (ValueError, TypeError):
            return None

    return {
        "actual"   : [_to_float(v) for v in y_test[:n]],
        "predicted": [_to_float(v) for v in y_pred[:n]],
    }
