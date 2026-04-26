"""
eda.py  —  Exploratory Data Analysis utilities
Performance-optimized for large datasets (50,000+ rows).

Key optimisations:
  - Intelligent sampling for correlation & describe operations (>10 k rows)
  - Vectorised operations throughout (no Python-level row iteration)
  - Results include 'sampled' flag so the frontend can show a note
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Sampling thresholds ───────────────────────────────────────────────
_DESCRIBE_SAMPLE  = 10_000   # rows used for describe() on large frames
_CORR_SAMPLE      = 10_000   # rows used for correlation matrix
_SCATTER_SAMPLE   = 1_000    # scatter / feature-vs-target points sent to UI


def _maybe_sample(df: pd.DataFrame, n: int, seed: int = 42) -> Tuple[pd.DataFrame, bool]:
    """Return (sample, was_sampled).  No-op when frame fits within n rows."""
    if len(df) > n:
        return df.sample(n=n, random_state=seed), True
    return df, False


# ─────────────────────────────────────────────────────────────────────
def summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Descriptive statistics for numeric and categorical columns.
    Uses a sample for describe() on large datasets to stay fast;
    missing/unique counts are always computed on the full frame.
    """
    def _clean(v: Any) -> Any:
        if v is pd.NA or v is pd.NaT:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    sample_df, was_sampled = _maybe_sample(df, _DESCRIBE_SAMPLE)

    desc = sample_df.describe(include="all")
    try:
        desc_clean = desc.map(_clean)
    except AttributeError:          # pandas < 2.1
        desc_clean = desc.applymap(_clean)

    # Full-frame counts are cheap — always use the full dataset
    missing       = {col: int(cnt) for col, cnt in df.isnull().sum().items()}
    unique_counts = {col: int(cnt) for col, cnt in df.nunique().items()}

    return {
        "numeric"     : desc_clean.select_dtypes(include=[np.number]).to_dict(),
        "all"         : desc_clean.to_dict(),
        "shape"       : {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "dtypes"      : {col: str(dt) for col, dt in df.dtypes.items()},
        "missing"     : missing,
        "unique_counts": unique_counts,
        "sampled"     : was_sampled,
        "sample_size" : len(sample_df),
    }


def histogram_data(df: pd.DataFrame, column: str, bins: int = 20) -> Dict[str, Any]:
    """Histogram bin edges and counts for a single column."""
    series = df[column].dropna()
    if series.dtype == object:
        vc = series.value_counts().head(30)
        return {"type": "categorical", "labels": list(vc.index), "counts": vc.values.tolist()}
    counts, edges = np.histogram(series, bins=bins)
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    return {"type": "numeric", "labels": labels, "counts": counts.tolist()}


def correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Pearson correlation matrix for all numeric columns.
    Samples to 10 k rows for large datasets — statistically very accurate
    and reduces compute time from minutes to milliseconds.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or num_df.shape[1] < 2:
        return {
            "columns"  : list(num_df.columns),
            "matrix"   : [],
            "min_val"  : 0.0,
            "max_val"  : 0.0,
            "sampled"  : False,
            "sample_size": len(df),
        }

    sample_num, was_sampled = _maybe_sample(num_df, _CORR_SAMPLE)

    try:
        corr = sample_num.corr(method="pearson", numeric_only=False)
    except TypeError:
        corr = sample_num.corr(method="pearson")

    cols = list(corr.columns)

    def _safe(v: Any) -> Optional[float]:
        try:
            f = float(v)
            return None if (f != f) else round(f, 4)
        except (TypeError, ValueError):
            return None

    matrix = [[_safe(corr.at[r, c]) for c in cols] for r in cols]

    off_diag = [
        v for i, row in enumerate(matrix)
        for j, v in enumerate(row)
        if v is not None and i != j
    ]
    min_val = round(float(min(off_diag)), 4) if off_diag else 0.0
    max_val = round(float(max(off_diag)), 4) if off_diag else 0.0

    return {
        "columns"    : cols,
        "matrix"     : matrix,
        "min_val"    : min_val,
        "max_val"    : max_val,
        "sampled"    : was_sampled,
        "sample_size": len(sample_num),
    }


def value_counts_all(df: pd.DataFrame) -> Dict[str, Any]:
    """Top-20 value counts for each categorical column."""
    result = {}
    for col in df.select_dtypes(include="object").columns:
        vc = df[col].value_counts().head(20)
        result[col] = {"labels": list(vc.index), "counts": vc.values.tolist()}
    return result


def feature_vs_target(
    df: pd.DataFrame, feature_col: str, target_col: str
) -> Dict[str, Any]:
    """Average target value grouped by a categorical feature (for EDA scatter/box)."""
    if df[feature_col].dtype == object:
        grouped = df.groupby(feature_col, sort=False)[target_col].mean().reset_index()
        return {
            "labels": list(grouped[feature_col]),
            "means" : list(grouped[target_col]),
        }
    # Numeric scatter: sample for frontend performance
    n_samples = min(_SCATTER_SAMPLE, len(df))
    sample    = df[[feature_col, target_col]].dropna().sample(n_samples, random_state=42)
    return {
        "x": sample[feature_col].tolist(),
        "y": sample[target_col].tolist(),
    }


# ────────────────────────────────────────────────────────────────────── #
#  Class Distribution Analysis  (read-only — NO data modification)       #
#  Severity thresholds aligned with Class Imbalance step (imbalance.py)  #
#    ≥ 0.8  → Well Balanced                                              #
#    0.5–0.8 → Slight Imbalance (Acceptable)                             #
#    0.3–0.5 → Moderate Imbalance                                        #
#    < 0.3  → Severe Imbalance                                           #
# ────────────────────────────────────────────────────────────────────── #

def _classify_severity(ratio: float) -> Dict[str, str]:
    """Unified severity classification — must match imbalance.py thresholds."""
    if ratio >= 0.8:
        return {
            "severity":     "well_balanced",
            "status":       "Well Balanced",
            "color":        "emerald",
            "range":        "≥ 0.8",
            "description":  "Dataset is well-balanced. Resampling is not recommended.",
        }
    if ratio >= 0.5:
        return {
            "severity":     "slight",
            "status":       "Slight Imbalance",
            "color":        "sky",
            "range":        "0.5–0.8",
            "description":  "Slight imbalance — usually acceptable. Run baseline evaluation to confirm.",
        }
    if ratio >= 0.3:
        return {
            "severity":     "moderate",
            "status":       "Moderate Imbalance",
            "color":        "amber",
            "range":        "0.3–0.5",
            "description":  "Moderate imbalance. Class weighting or SMOTE recommended.",
        }
    return {
        "severity":     "severe",
        "status":       "Severe Imbalance",
        "color":        "red",
        "range":        "< 0.3",
        "description":  "Severe imbalance. SMOTE or hybrid methods strongly recommended.",
    }


def class_distribution(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Analyse the distribution of the target variable.

    Constraints
    -----------
    * Pure analysis — the DataFrame is NEVER modified.
    * Works on the raw / unprocessed data so the distribution reflects
      the original dataset faithfully.
    * Severity thresholds aligned with the Class Imbalance pipeline step.
    """
    series = df[target_col].dropna()
    total  = int(len(series))

    if total == 0:
        raise ValueError(f"Target column '{target_col}' contains no non-null values.")

    value_counts: pd.Series = series.value_counts()
    num_classes              = int(len(value_counts))

    counts: Dict[str, int] = {
        str(k): int(v) for k, v in value_counts.items()
    }
    percentages: Dict[str, float] = {
        str(k): round(float(v) / total * 100, 2)
        for k, v in value_counts.items()
    }

    majority_class = str(value_counts.index[0])
    minority_class = str(value_counts.index[-1])
    majority_count = int(value_counts.iloc[0])
    minority_count = int(value_counts.iloc[-1])

    imbalance_ratio = round(minority_count / majority_count, 4) if majority_count > 0 else 1.0
    minority_pct    = round(minority_count / total * 100, 2)

    # Unified severity classification (matches imbalance.py)
    sev_info   = _classify_severity(imbalance_ratio)
    severity   = sev_info["severity"]
    status     = sev_info["status"]
    is_balanced = severity in ("well_balanced", "slight")

    if severity == "well_balanced":
        insight = (
            f"The dataset is well-balanced (ratio {imbalance_ratio:.2f} ≥ 0.8). "
            f"The minority class '{minority_class}' represents {minority_pct:.1f}% of the data. "
            f"No resampling is required — standard training procedures will work well."
        )
        recommendations: List[str] = [
            "Use standard training without class balancing.",
            "If model recall is poor despite the balanced ratio, consider class_weight='balanced'.",
        ]
    elif severity == "slight":
        insight = (
            f"The dataset has slight imbalance (ratio {imbalance_ratio:.2f}, range 0.5–0.8). "
            f"The minority class '{minority_class}' represents {minority_pct:.1f}% of the data. "
            f"This is often acceptable — run a baseline evaluation in the Class Imbalance step "
            f"to confirm whether balancing is needed."
        )
        recommendations = [
            "Run the baseline model evaluation (Class Imbalance step) to check minority recall.",
            "If minority recall < 0.6, apply class_weight='balanced' as a first step.",
            "Only apply SMOTE if class_weight does not improve minority recall.",
        ]
    elif severity == "moderate":
        insight = (
            f"The dataset has moderate imbalance (ratio {imbalance_ratio:.2f}, range 0.3–0.5). "
            f"The minority class '{minority_class}' represents only {minority_pct:.1f}% of the data. "
            f"Without intervention, models may be biased toward the majority class "
            f"'{majority_class}', leading to poor recall for minority samples."
        )
        recommendations = [
            "Try class_weight='balanced' first — often sufficient for moderate imbalance.",
            "If minority recall remains below 0.6, consider SMOTE oversampling.",
            "Prefer F1-score, Recall, and PR-AUC over accuracy as evaluation metrics.",
            "Use the Class Imbalance step for baseline evaluation before deciding.",
        ]
    else:  # severe
        insight = (
            f"The dataset has severe imbalance (ratio {imbalance_ratio:.2f} < 0.3). "
            f"The minority class '{minority_class}' represents only {minority_pct:.1f}% of the data. "
            f"Without intervention, the model will almost certainly fail to detect the minority class, "
            f"which is especially harmful in high-stakes predictions (fraud, diagnosis, etc.)."
        )
        recommendations = [
            "Apply SMOTE+Tomek (hybrid) for both oversampling and boundary cleaning.",
            "For very few minority samples (< 6), use class_weight instead of SMOTE.",
            "Always evaluate with F1-score, Recall (minority), and PR-AUC — NOT accuracy.",
            "Consider collecting more minority class data if possible.",
            "Use threshold tuning after training to optimise the decision boundary.",
        ]

    return {
        "target_column"  : target_col,
        "total_samples"  : total,
        "num_classes"    : num_classes,
        "counts"         : counts,
        "percentages"    : percentages,
        "majority_class" : majority_class,
        "minority_class" : minority_class,
        "majority_count" : majority_count,
        "minority_count" : minority_count,
        "minority_pct"   : minority_pct,
        "imbalance_ratio": imbalance_ratio,
        "severity"       : severity,
        "severity_color" : sev_info["color"],
        "severity_range" : sev_info["range"],
        "status"         : status,
        "is_balanced"    : is_balanced,
        "insight"        : insight,
        "recommendations": recommendations,
    }


# ══════════════════════════════════════════════════════════════════════ #
#  EDA v2 — ML Readiness Analysis Functions                              #
#  All are pure (no DataFrame mutation) and sampling-aware.             #
# ══════════════════════════════════════════════════════════════════════ #

def _iqr_outlier_count(series: pd.Series) -> int:
    """Return number of outliers using the IQR fence method."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    fence_lo, fence_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < fence_lo) | (series > fence_hi)).sum())


def compute_data_quality(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """
    Missing values, duplicates, outliers (IQR), and skewness.
    Pure analysis — DataFrame is never modified.
    """
    n = len(df)

    # Missing values
    missing_series = df.isnull().sum()
    total_missing  = int(missing_series.sum())
    total_cells    = n * len(df.columns)
    missing_pct    = round(100 * total_missing / total_cells, 2) if total_cells > 0 else 0.0
    missing_cols   = [
        {"name": col, "missing_count": int(cnt), "missing_percent": round(100 * cnt / n, 2)}
        for col, cnt in missing_series.items() if cnt > 0
    ]
    missing_cols.sort(key=lambda x: -x["missing_percent"])

    # Duplicates
    dup_count = int(df.duplicated().sum())
    dup_pct   = round(100 * dup_count / n, 2) if n > 0 else 0.0

    # Outliers (IQR) — numeric features only, skip target
    num_cols = df.select_dtypes(include=[np.number]).columns
    outlier_results = []
    for col in num_cols:
        if col == target_col:
            continue
        s = df[col].dropna()
        if len(s) < 4:
            continue
        cnt = _iqr_outlier_count(s)
        if cnt > 0:
            outlier_results.append({
                "name":          col,
                "outlier_count": cnt,
                "outlier_pct":   round(100 * cnt / len(s), 2),
            })
    outlier_results.sort(key=lambda x: -x["outlier_count"])

    # Skewness — flag |skew| > 1
    skewness_results = []
    for col in num_cols:
        if col == target_col:
            continue
        s = df[col].dropna()
        if len(s) < 3:
            continue
        try:
            skew_val = float(s.skew())
            if abs(skew_val) > 1.0:
                skewness_results.append({
                    "feature":    col,
                    "skew_value": round(skew_val, 4),
                    "direction":  "right-skewed" if skew_val > 0 else "left-skewed",
                    "severity":   "high" if abs(skew_val) > 2.0 else "moderate",
                })
        except Exception:
            pass
    skewness_results.sort(key=lambda x: -abs(x["skew_value"]))

    return {
        "missing_values": {
            "total_missing":      total_missing,
            "missing_percentage": missing_pct,
            "columns":            missing_cols,
        },
        "duplicates": {
            "count":      dup_count,
            "percentage": dup_pct,
        },
        "outliers": {
            "columns": outlier_results,
        },
        "skewness": skewness_results,
    }


def compute_feature_diagnostics(
    df: pd.DataFrame, target_col: str = None
) -> List[Dict[str, Any]]:
    """
    Per-feature diagnostic flags with human-readable reasons.
    Flags: Zero Variance | Near Constant | High Cardinality |
           Potential ID | High Risk Leakage | Very Strong Predictor (Check)
    """
    n = len(df)
    num_df = df.select_dtypes(include=[np.number])
    _leakage_keywords = {"target", "label", "output", "result", "prediction", "score", "outcome"}
    results = []

    for col in df.columns:
        if col == target_col:
            continue

        series   = df[col]
        dtype    = str(series.dtype)
        n_unique = int(series.nunique(dropna=True))
        flags:   List[str] = []
        reasons: List[str] = []

        # ── Zero Variance / Near Constant ──────────────────────────────
        if n_unique <= 1:
            flags.append("Zero Variance")
            reasons.append("Column has only one unique value — carries no information.")
        elif n > 0:
            top_freq = series.value_counts(normalize=True, dropna=True).iloc[0]
            if top_freq >= 0.95:
                flags.append("Near Constant")
                reasons.append(
                    f"{round(top_freq * 100, 1)}% of values are identical "
                    "— nearly constant, very low signal."
                )

        # High Cardinality — only meaningful for categorical/text columns, not continuous numerics
        _is_categorical_type = (
            series.dtype == object
            or isinstance(series.dtype, (pd.CategoricalDtype, pd.StringDtype))
            or str(series.dtype) in ("category", "string")
        )
        if _is_categorical_type and n_unique > 20 and n > 0 and (n_unique / n) > 0.1:
            flags.append("High Cardinality")
            reasons.append(
                f"{n_unique:,} unique values ({round(100 * n_unique / n, 1)}% of rows) "
                "— requires target encoding or frequency encoding."
            )

        # Potential ID — safe dtype check (handles pd.StringDtype / nullable int)
        try:
            _is_int = np.issubdtype(series.dtype, np.integer)
        except TypeError:
            _is_int = False
        is_cat_or_int = (
            series.dtype == object
            or isinstance(series.dtype, pd.StringDtype)
            or _is_int
        )
        if is_cat_or_int and n > 0 and n_unique == n:
            flags.append("Potential ID")
            reasons.append(
                "Every row has a unique value — this is likely an ID column "
                "and will not generalise to unseen data."
            )

        # ── Leakage detection ──────────────────────────────────────────
        col_lower    = col.lower()
        name_leakage = any(kw in col_lower for kw in _leakage_keywords)
        corr_leakage = False
        corr_val_abs = None
        if target_col and col in num_df.columns and target_col in df.columns:
            try:
                tgt = df[target_col]
                if pd.api.types.is_numeric_dtype(tgt):
                    cv = abs(float(df[col].corr(tgt)))
                    if not np.isnan(cv) and cv > 0.95:
                        corr_leakage = True
                        corr_val_abs = round(cv, 4)
            except Exception:
                pass

        if name_leakage and corr_leakage:
            flags.append("High Risk Leakage")
            reasons.append(
                f"Suspicious name (contains keyword) AND correlation with target = {corr_val_abs}. "
                "Very likely leakage — remove before training."
            )
        elif corr_leakage:
            flags.append("Very Strong Predictor (Check)")
            reasons.append(
                f"Correlation with target = {corr_val_abs}. "
                "Could be legitimate or post-event data — verify carefully."
            )
        elif name_leakage:
            flags.append("High Risk Leakage")
            reasons.append(
                "Column name contains a leakage keyword (target/label/output/result/score). "
                "Inspect and remove if it encodes the target."
            )

        # ── Skewness value (for display) ───────────────────────────────
        skew_val = None
        if col in num_df.columns:
            try:
                s = df[col].dropna()
                if len(s) >= 3:
                    skew_val = round(float(s.skew()), 4)
            except Exception:
                pass

        results.append({
            "feature":       col,
            "type":          dtype,
            "unique_values": n_unique,
            "skewness":      skew_val,
            "flags":         flags,
            "reasons":       reasons,
        })

    return results


def compute_feature_relationships(
    df: pd.DataFrame, target_col: str
) -> Dict[str, Any]:
    """
    Numerical → Pearson correlation with target.
    Categorical → Cramér's V (categorical target) or mean-diff (numeric target).
    """
    if target_col not in df.columns:
        return {"numerical_vs_target": [], "categorical_vs_target": []}

    tgt = df[target_col]
    num_results, cat_results = [], []

    # Numerical vs target
    if pd.api.types.is_numeric_dtype(tgt):
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == target_col:
                continue
            try:
                corr_val = float(df[col].corr(tgt))
                if not np.isnan(corr_val):
                    num_results.append({"feature": col, "correlation": round(corr_val, 4)})
            except Exception:
                pass
        num_results.sort(key=lambda x: -abs(x["correlation"]))

    # Categorical vs target
    n_rows = len(df)
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if col == target_col:
            continue
        try:
            n_unique_col = int(df[col].nunique(dropna=True))
            if pd.api.types.is_numeric_dtype(tgt):
                # Mean-diff impact (safe for any cardinality)
                grp    = df.groupby(col, sort=False)[target_col].mean()
                impact = round(float(grp.max() - grp.min()), 4)
                method = "mean_diff"
            elif n_unique_col > 20 or (n_rows > 0 and n_unique_col / n_rows > 0.1):
                # High-cardinality guard: Cramér's V undefined/unstable — use frequency impact
                freq   = df[col].value_counts(normalize=True)
                impact = round(float(freq.iloc[0] - freq.iloc[-1]), 4) if len(freq) > 1 else 0.0
                method = "frequency"
            else:
                # Cramér's V (only for well-behaved low-cardinality categoricals)
                ct   = pd.crosstab(df[col], tgt)
                n_ct = ct.values.sum()
                if n_ct == 0:
                    continue
                expected = (ct.sum(axis=0).values * ct.sum(axis=1).values.reshape(-1, 1)) / n_ct
                chi2   = float(((ct.values - expected) ** 2 / (expected + 1e-9)).sum())
                phi2   = chi2 / n_ct
                r, k   = ct.shape
                impact = round(float(np.sqrt(phi2 / max(min(k - 1, r - 1), 1))), 4)
                method = "cramers_v"
            cat_results.append({"feature": col, "impact_score": impact, "method": method})
        except Exception:
            pass
    cat_results.sort(key=lambda x: -abs(x["impact_score"]))

    return {
        "numerical_vs_target":   num_results[:20],
        "categorical_vs_target": cat_results[:20],
    }


def compute_highly_correlated_pairs(
    df: pd.DataFrame, threshold: float = 0.9
) -> List[Dict[str, Any]]:
    """Return feature pairs with |correlation| > threshold with drop recommendation."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return []
    sample_df, _ = _maybe_sample(num_df, _CORR_SAMPLE)
    try:
        corr = sample_df.corr(method="pearson", numeric_only=False)
    except TypeError:
        corr = sample_df.corr(method="pearson")

    cols  = list(corr.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if not np.isnan(val) and abs(val) > threshold:
                abs_val = abs(val)
                risk    = "Critical" if abs_val > 0.97 else "High" if abs_val > 0.95 else "Moderate"
                pairs.append({
                    "feature_1":           cols[i],
                    "feature_2":           cols[j],
                    "correlation":         round(float(val), 4),
                    "risk":                risk,
                    "recommended_action":  (
                        f"Drop '{cols[j]}' (or '{cols[i]}') — "
                        "keeping both introduces multicollinearity."
                    ),
                })
    pairs.sort(key=lambda x: -abs(x["correlation"]))
    return pairs


def compute_ml_readiness_score(
    quality: Dict[str, Any],
    diagnostics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """0–100 quality score with per-category breakdown."""
    missing_pen  = 0
    dup_pen      = 0
    outlier_pen  = 0
    skewness_pen = 0
    leakage_pen  = 0

    for col in quality["missing_values"]["columns"]:
        missing_pen += 3 if col["missing_percent"] > 30 else 1

    dup_pct = quality["duplicates"]["percentage"]
    dup_pen = (10 if dup_pct > 5 else 5) if dup_pct > 1 else 0

    for col in quality["outliers"]["columns"]:
        if col["outlier_pct"] > 5:
            outlier_pen += 1

    for sk in quality["skewness"]:
        if sk["severity"] == "high":
            skewness_pen += 1

    flag_counts: Dict[str, int] = {}
    for feat in diagnostics:
        for flag in feat["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    card_pen    = min(flag_counts.get("High Cardinality", 0) * 2, 10)
    zero_pen    = flag_counts.get("Zero Variance", 0) * 3
    leakage_pen = (
        flag_counts.get("High Risk Leakage", 0) * 15 +
        flag_counts.get("Very Strong Predictor (Check)", 0) * 5
    )

    total_deduction = missing_pen + dup_pen + outlier_pen + skewness_pen + leakage_pen + card_pen + zero_pen
    score = max(0, min(100, 100 - total_deduction))

    return {
        "score": score,
        "breakdown": {
            "missing_penalty":    missing_pen,
            "duplicate_penalty":  dup_pen,
            "outlier_penalty":    outlier_pen,
            "skewness_penalty":   skewness_pen,
            "leakage_penalty":    leakage_pen,
            "cardinality_penalty": card_pen,
            "variance_penalty":   zero_pen,
        },
    }


def generate_suggested_actions(
    quality: Dict[str, Any],
    diagnostics: List[Dict[str, Any]],
    target_analysis: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate structured, prioritised ML recommendations.
    Priority: Leakage > Missing > Duplicates > Imbalance > Outliers > Encoding > Cardinality
    Each action has: priority, emoji, title, detail, fix.
    """
    actions: List[Dict[str, Any]] = []

    # 1. DATA LEAKAGE — highest risk
    high_risk = [f for f in diagnostics if "High Risk Leakage" in f["flags"]]
    strong_pred = [f for f in diagnostics if "Very Strong Predictor (Check)" in f["flags"]]
    zero_var  = [f for f in diagnostics if "Zero Variance" in f["flags"]]

    for feat in high_risk:
        reason = next((r for r in feat.get("reasons", []) if "leakage" in r.lower() or "keyword" in r.lower() or "correlation" in r.lower()), "")
        actions.append({
            "priority": 1,
            "level":    "critical",
            "emoji":    "🚨",
            "title":    f"High Risk Leakage — {feat['feature']}",
            "detail":   reason or "Suspicious name and/or very high correlation with target.",
            "fix":      "Remove this feature before training. It likely encodes the target outcome.",
        })

    for feat in strong_pred:
        reason = next((r for r in feat.get("reasons", []) if "correlation" in r.lower()), "")
        actions.append({
            "priority": 1,
            "level":    "warn",
            "emoji":    "⚡",
            "title":    f"Very Strong Predictor — {feat['feature']}",
            "detail":   reason or "Correlation with target exceeds 0.95.",
            "fix":      "Verify this feature is available at inference time. If post-event data, remove it.",
        })

    # 2. MISSING VALUES
    critical_miss = [c for c in quality["missing_values"]["columns"] if c["missing_percent"] > 30]
    moderate_miss = [c for c in quality["missing_values"]["columns"] if 5 < c["missing_percent"] <= 30]
    low_miss      = [c for c in quality["missing_values"]["columns"] if 0 < c["missing_percent"] <= 5]

    if critical_miss:
        names = ", ".join(c["name"] for c in critical_miss[:3])
        actions.append({
            "priority": 2,
            "level":    "critical",
            "emoji":    "⚠",
            "title":    f"{len(critical_miss)} column(s) with >30% missing values",
            "detail":   f"Affected: {names}. High missingness degrades model reliability.",
            "fix":      "Consider dropping these columns or using advanced imputation (e.g., KNN imputer).",
        })
    if moderate_miss:
        names = ", ".join(c["name"] for c in moderate_miss[:3])
        actions.append({
            "priority": 2,
            "level":    "warn",
            "emoji":    "📋",
            "title":    f"{len(moderate_miss)} column(s) with 5–30% missing values",
            "detail":   f"Affected: {names}.",
            "fix":      "Impute with median (numeric) or mode (categorical). Consider a 'missing' indicator column.",
        })
    if low_miss:
        names = ", ".join(c["name"] for c in low_miss[:3])
        actions.append({
            "priority": 2,
            "level":    "info",
            "emoji":    "📌",
            "title":    f"{len(low_miss)} column(s) with <5% missing values",
            "detail":   f"Affected: {names}.",
            "fix":      "Simple median/mode imputation is sufficient.",
        })

    # 3. DUPLICATES
    dup_pct = quality["duplicates"]["percentage"]
    if dup_pct > 1:
        actions.append({
            "priority": 3,
            "level":    "warn" if dup_pct <= 5 else "critical",
            "emoji":    "♻",
            "title":    f"{quality['duplicates']['count']:,} duplicate rows ({dup_pct:.1f}%)",
            "detail":   "Duplicate rows cause overfitting and inflate cross-validation scores.",
            "fix":      "Run df.drop_duplicates() in the preprocessing step before splitting.",
        })

    # 4. CLASS IMBALANCE
    if target_analysis:
        ratio = target_analysis.get("imbalance_ratio", 1.0)
        if ratio < 0.3:
            actions.append({
                "priority": 4,
                "level":    "critical",
                "emoji":    "⚖",
                "title":    f"Severe class imbalance (ratio = {ratio:.2f})",
                "detail":   "The minority class represents less than 23% of training data. Models will be biased toward the majority class.",
                "fix":      "Apply SMOTE during training (Class Imbalance step). For KNN/Logistic Regression, scale BEFORE SMOTE. Tree-based models skip scaling.",
            })
        elif ratio < 0.6:
            actions.append({
                "priority": 4,
                "level":    "warn",
                "emoji":    "⚖",
                "title":    f"Moderate class imbalance (ratio = {ratio:.2f})",
                "detail":   "Minority class may be underrepresented. Evaluate minority recall in baseline.",
                "fix":      "Try class_weight='balanced' first. Use SMOTE only if minority recall < 0.6.",
            })

    # 5. OUTLIERS
    high_outlier = [c for c in quality["outliers"]["columns"] if c["outlier_pct"] > 5]
    if high_outlier:
        names = ", ".join(c["name"] for c in high_outlier[:3])
        actions.append({
            "priority": 5,
            "level":    "warn",
            "emoji":    "📊",
            "title":    f"{len(high_outlier)} feature(s) with >5% outliers",
            "detail":   f"Affected: {names}. Outliers can distort linear models significantly.",
            "fix":      "Apply IQR clipping or RobustScaler. Tree-based models are naturally resistant.",
        })

    # 6. SKEWNESS → scaling/encoding signal
    high_skew = [s for s in quality["skewness"] if s["severity"] == "high"]
    if high_skew:
        names = ", ".join(s["feature"] for s in high_skew[:3])
        actions.append({
            "priority": 6,
            "level":    "warn",
            "emoji":    "📐",
            "title":    f"{len(high_skew)} highly-skewed feature(s) (|skew| > 2)",
            "detail":   f"Affected: {names}. High skew degrades linear model convergence.",
            "fix":      "Apply log1p (right-skewed) or square transform (left-skewed) in preprocessing.",
        })

    # 7. ZERO VARIANCE
    if zero_var:
        names = ", ".join(f["feature"] for f in zero_var[:3])
        actions.append({
            "priority": 6,
            "level":    "critical",
            "emoji":    "🗑",
            "title":    f"{len(zero_var)} zero-variance feature(s)",
            "detail":   f"Affected: {names}. These columns carry no information.",
            "fix":      "Drop before training — they will never contribute to predictions.",
        })

    # 8. CARDINALITY
    high_card = [f for f in diagnostics if "High Cardinality" in f["flags"]]
    id_cols   = [f for f in diagnostics if "Potential ID" in f["flags"]]
    if high_card:
        names = ", ".join(f["feature"] for f in high_card[:3])
        actions.append({
            "priority": 7,
            "level":    "warn",
            "emoji":    "🏷",
            "title":    f"{len(high_card)} high-cardinality feature(s)",
            "detail":   f"Affected: {names}. One-hot encoding will create too many columns.",
            "fix":      "Use target encoding (for tree models) or frequency encoding (for linear models).",
        })
    if id_cols:
        names = ", ".join(f["feature"] for f in id_cols[:3])
        actions.append({
            "priority": 7,
            "level":    "warn",
            "emoji":    "🔑",
            "title":    f"{len(id_cols)} likely ID column(s)",
            "detail":   f"Affected: {names}. ID columns are unique per row and won't generalise.",
            "fix":      "Remove from feature set before training.",
        })

    # Sort by priority
    actions.sort(key=lambda a: a["priority"])

    if not actions:
        actions.append({
            "priority": 0,
            "level":    "good",
            "emoji":    "✅",
            "title":    "Dataset appears ML-ready",
            "detail":   "No critical issues detected in the automated analysis.",
            "fix":      "Proceed to Feature Engineering and Model Training.",
        })

    return actions


def full_ml_readiness_analysis(
    df: pd.DataFrame, target_col: str = None
) -> Dict[str, Any]:
    """
    Master function called by GET /api/eda/v2/analysis.
    Aggregates all sub-analyses into the full ML readiness payload.
    """
    n_rows    = int(len(df))
    n_cols    = int(len(df.columns))
    mem_mb    = round(float(df.memory_usage(deep=True).sum()) / (1024 ** 2), 3)
    num_count = int(len(df.select_dtypes(include=[np.number]).columns))
    cat_count = int(len(df.select_dtypes(include=["object", "category"]).columns))
    bool_count= int(len(df.select_dtypes(include=["bool"]).columns))

    overview = {
        "num_rows":         n_rows,
        "num_columns":      n_cols,
        "memory_usage_mb":  mem_mb,
        "feature_types": {
            "numerical":   num_count,
            "categorical": cat_count,
            "boolean":     bool_count,
        },
    }

    quality         = compute_data_quality(df, target_col)
    diagnostics     = compute_feature_diagnostics(df, target_col)
    relationships   = (
        compute_feature_relationships(df, target_col)
        if target_col else {"numerical_vs_target": [], "categorical_vs_target": []}
    )
    high_corr_pairs = compute_highly_correlated_pairs(df)

    # Target analysis — only for classification (≤20 unique values)
    target_analysis = None
    if target_col and target_col in df.columns:
        try:
            n_unique_tgt = int(df[target_col].nunique())
            if n_unique_tgt <= 20:
                raw = class_distribution(df, target_col)
                target_analysis = {
                    "target_column":      target_col,
                    "class_distribution": [
                        {"class": cls, "count": raw["counts"][cls], "percentage": raw["percentages"][cls]}
                        for cls in raw["counts"]
                    ],
                    "imbalance_ratio": raw["imbalance_ratio"],
                    "interpretation":  raw["status"],
                    "severity":        raw.get("severity", ""),
                    "insight":         raw.get("insight", ""),
                }
        except Exception:
            target_analysis = None

    score_result = compute_ml_readiness_score(quality, diagnostics)
    actions      = generate_suggested_actions(quality, diagnostics, target_analysis)

    return {
        "overview":              overview,
        "data_quality":          quality,
        "target_analysis":       target_analysis,
        "feature_relationships": relationships,
        "correlation_analysis":  {"highly_correlated_pairs": high_corr_pairs},
        "feature_diagnostics":   diagnostics,
        "suggested_actions":     actions,
        "data_quality_score":    score_result,
    }
