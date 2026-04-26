"""
feature_engineering.py — Intelligent Feature Engineering Engine
===============================================================
Automatically analyzes a dataset and generates meaningful new features:

  1. Column Analysis     — classify as numeric / categorical / binary
  2. Categorized Features — domain-specific and quantile binning
  3. Interaction Features — product of semantically related numeric pairs
  4. Risk Score           — sum of binary risk-condition columns
  5. Manual Feature       — user-defined add / multiply / ratio / bin

Domain rules cover healthcare (BMI, glucose, cholesterol, age, BP …)
but fall back to generic quantile binning for unknown numeric columns.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Normalization helper ───────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[\s_\-./]", "", str(s).lower())


# ── Domain-specific binning rules ─────────────────────────────────────
DOMAIN_BINS: Dict[str, Dict] = {
    "bmi": {
        "bins"  : [0, 18.5, 25, 30, float("inf")],
        "labels": ["Underweight", "Normal", "Overweight", "Obese"],
        "desc"  : "WHO BMI classification (kg/m²)",
    },
    "glucose": {
        "bins"  : [0, 100, 126, float("inf")],
        "labels": ["Normal", "Prediabetic", "Diabetic"],
        "desc"  : "Fasting blood glucose thresholds (mg/dL)",
    },
    "bloodsugar": {
        "bins"  : [0, 100, 126, float("inf")],
        "labels": ["Normal", "Prediabetic", "Diabetic"],
        "desc"  : "Blood sugar classification (mg/dL)",
    },
    "age": {
        "bins"  : [0, 30, 45, 60, float("inf")],
        "labels": ["Young", "Middle-Aged", "Senior", "Elderly"],
        "desc"  : "Age group classification",
    },
    "cholesterol": {
        "bins"  : [0, 200, 240, float("inf")],
        "labels": ["Normal", "Borderline", "High"],
        "desc"  : "Total cholesterol classification (mg/dL)",
    },
    "bloodpressure": {
        "bins"  : [0, 80, 90, 120, float("inf")],
        "labels": ["Normal", "Elevated", "Stage1", "Stage2"],
        "desc"  : "Diastolic blood pressure classification (mmHg)",
    },
    "heartrate": {
        "bins"  : [0, 60, 100, float("inf")],
        "labels": ["Low", "Normal", "Elevated"],
        "desc"  : "Resting heart rate classification (bpm)",
    },
    "insulin": {
        "bins"  : [0, 16, 166, float("inf")],
        "labels": ["Low", "Normal", "High"],
        "desc"  : "Insulin level classification (μU/mL)",
    },
    "hba1c": {
        "bins"  : [0, 5.7, 6.5, float("inf")],
        "labels": ["Normal", "Prediabetic", "Diabetic"],
        "desc"  : "HbA1c classification (%)",
    },
    "skinthickness": {
        "bins"  : [0, 20, 40, float("inf")],
        "labels": ["Low", "Normal", "High"],
        "desc"  : "Skin thickness classification (mm)",
    },
    "dpf": {
        "bins"  : [0, 0.3, 0.6, float("inf")],
        "labels": ["Low", "Medium", "High"],
        "desc"  : "Diabetes Pedigree Function risk level",
    },
    "pregnancies": {
        "bins"  : [0, 1, 4, float("inf")],
        "labels": ["None", "Low", "High"],
        "desc"  : "Pregnancy count classification",
    },
    "income": {
        "bins"  : [0, 25000, 75000, float("inf")],
        "labels": ["Low", "Middle", "High"],
        "desc"  : "Income bracket classification",
    },
    "salary": {
        "bins"  : [0, 25000, 75000, float("inf")],
        "labels": ["Low", "Middle", "High"],
        "desc"  : "Salary bracket classification",
    },
}

# Keyword pairs for semantic interaction detection
_RELATED_PAIRS = [
    ("bmi",         "bloodpressure"),
    ("bmi",         "glucose"),
    ("bmi",         "cholesterol"),
    ("bmi",         "insulin"),
    ("cholesterol", "age"),
    ("age",         "glucose"),
    ("age",         "diabetes"),
    ("glucose",     "insulin"),
    ("glucose",     "hba1c"),
    ("heartrate",   "bloodpressure"),
    ("stress",      "sleep"),
    ("smoking",     "cholesterol"),
    ("age",         "heartrate"),
    ("skinthickness","bmi"),
]

# Binary risk-condition keywords
_RISK_KEYWORDS = [
    "diabetes", "hypertension", "smoking", "stroke", "cancer",
    "obesity", "depression", "anxiety", "disease", "disorder",
    "attack", "failure", "condition", "risk", "chronic",
]


# ── Column Analysis ────────────────────────────────────────────────────

def analyze_columns(df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
    """
    Classify every non-target column as numeric / categorical / binary
    and compute basic statistics.
    """
    results = []
    for col in df.columns:
        if col == target_col:
            continue
        series   = df[col].dropna()
        n_unique = int(series.nunique())
        is_num   = np.issubdtype(df[col].dtype, np.number)

        if n_unique <= 2:
            col_type = "binary"
        elif not is_num or n_unique <= 15:
            col_type = "categorical"
        else:
            col_type = "numeric"

        entry: Dict[str, Any] = {
            "column"        : col,
            "type"          : col_type,
            "dtype"         : str(df[col].dtype),
            "unique_values" : n_unique,
            "missing_pct"   : round(df[col].isnull().mean() * 100, 2),
            "sample_values" : [str(v) for v in series.unique()[:5].tolist()],
        }
        if is_num and len(series) > 0:
            entry["min"]  = round(float(series.min()),  3)
            entry["max"]  = round(float(series.max()),  3)
            entry["mean"] = round(float(series.mean()), 3)
            entry["std"]  = round(float(series.std()),  3)
        results.append(entry)
    return results


# ── Binning helpers ────────────────────────────────────────────────────

def _apply_domain_bin(df: pd.DataFrame, col: str, rule: Dict) -> Tuple[pd.Series, str]:
    binned = pd.cut(
        df[col], bins=rule["bins"], labels=rule["labels"],
        right=False, include_lowest=True,
    )
    logic = " | ".join(
        f"{rule['bins'][i]}–{rule['bins'][i+1]} → {rule['labels'][i]}"
        for i in range(len(rule["labels"]))
    )
    return binned.astype(str), logic


def _apply_quantile_bin(df: pd.DataFrame, col: str, n_bins: int = 4) -> Tuple[pd.Series, str]:
    labels = [f"Q{i+1}" for i in range(n_bins)]
    try:
        binned = pd.qcut(df[col], q=n_bins, labels=labels, duplicates="drop")
    except Exception:
        binned = pd.cut(df[col], bins=n_bins, labels=labels)
    return binned.astype(str), f"{n_bins} equal-frequency quantile buckets"


def _match_domain_rule(col: str) -> Optional[str]:
    """Return the matching DOMAIN_BINS key for a column name, or None."""
    col_n = _norm(col)
    return next((k for k in DOMAIN_BINS if k in col_n), None)


# ── Auto Feature Generation ────────────────────────────────────────────

def generate_auto_features(df: pd.DataFrame, target_col: str) -> List[Dict[str, Any]]:
    """
    Propose auto-generated features in three categories:
      1. Categorized  — numeric columns binned into meaningful groups
      2. Interaction  — product of semantically related numeric pairs
      3. Risk Score   — sum of binary risk-condition indicators

    Returns a list of proposal dicts, each containing:
      name, feature_type, source_columns, description, logic,
      enabled (bool), preview_values, is_domain_rule (bool)
    """
    col_info     = analyze_columns(df, target_col)
    numeric_cols = [c["column"] for c in col_info if c["type"] == "numeric"]
    binary_cols  = [c["column"] for c in col_info if c["type"] == "binary"]
    proposals: List[Dict[str, Any]] = []

    # ── 1. Categorized Features ──────────────────────────────────────
    for col in numeric_cols:
        matched = _match_domain_rule(col)
        if matched:
            rule       = DOMAIN_BINS[matched]
            series_bin, logic = _apply_domain_bin(df, col, rule)
            desc       = rule["desc"]
            auto_on    = True
        else:
            series_bin, logic = _apply_quantile_bin(df, col)
            desc       = f"Quantile-based grouping of '{col}' into 4 equal-frequency buckets"
            auto_on    = False

        counts = series_bin.value_counts().to_dict()
        proposals.append({
            "name"           : f"{col}_Category",
            "feature_type"   : "categorical",
            "source_columns" : [col],
            "description"    : desc,
            "logic"          : logic,
            "enabled"        : auto_on,
            "preview_values" : {str(k): int(v) for k, v in counts.items()},
            "is_domain_rule" : auto_on,
        })

    # ── 2. Interaction Features ──────────────────────────────────────
    norm_map = {_norm(c): c for c in numeric_cols}   # normalised → real col name
    added, seen = 0, set()
    MAX_INTERACTIONS = 6

    def _find_col(keyword: str) -> Optional[str]:
        """Find actual col name by keyword substring."""
        return next((norm_map[k] for k in norm_map if keyword in k), None)

    # Priority: domain-defined semantic pairs
    for k1, k2 in _RELATED_PAIRS:
        if added >= MAX_INTERACTIONS:
            break
        c1 = _find_col(k1)
        c2 = _find_col(k2)
        if c1 and c2 and c1 != c2:
            key = tuple(sorted([c1, c2]))
            if key not in seen:
                seen.add(key)
                try:
                    ivals = pd.to_numeric(df[c1], errors="coerce") * \
                            pd.to_numeric(df[c2], errors="coerce")
                    proposals.append({
                        "name"           : f"{c1}_x_{c2}",
                        "feature_type"   : "interaction",
                        "source_columns" : [c1, c2],
                        "description"    : (
                            f"Multiplicative interaction between '{c1}' and '{c2}' "
                            "— captures combined physiological or domain effect"
                        ),
                        "logic"          : f"{c1} × {c2}",
                        "enabled"        : True,
                        "preview_values" : {
                            "min" : round(float(ivals.min()),  3),
                            "max" : round(float(ivals.max()),  3),
                            "mean": round(float(ivals.mean()), 3),
                        },
                        "is_domain_rule" : True,
                    })
                    added += 1
                except Exception:
                    pass

    # Fill remaining slots with high-correlation pairs (when dataset is small enough)
    if added < MAX_INTERACTIONS and len(numeric_cols) >= 2:
        try:
            corr   = df[numeric_cols].corr().abs()
            pairs  = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
            )
            for (c1, c2), _ in pairs.items():
                if added >= MAX_INTERACTIONS:
                    break
                key = tuple(sorted([c1, c2]))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    ivals = pd.to_numeric(df[c1], errors="coerce") * \
                            pd.to_numeric(df[c2], errors="coerce")
                    proposals.append({
                        "name"           : f"{c1}_x_{c2}",
                        "feature_type"   : "interaction",
                        "source_columns" : [c1, c2],
                        "description"    : (
                            f"Multiplicative interaction between '{c1}' and '{c2}' "
                            "(selected by correlation strength)"
                        ),
                        "logic"          : f"{c1} × {c2}",
                        "enabled"        : False,
                        "preview_values" : {
                            "min" : round(float(ivals.min()),  3),
                            "max" : round(float(ivals.max()),  3),
                            "mean": round(float(ivals.mean()), 3),
                        },
                        "is_domain_rule" : False,
                    })
                    added += 1
                except Exception:
                    pass
        except Exception:
            pass

    # ── 3. Risk Score ────────────────────────────────────────────────
    def _to_binary_01(s: pd.Series) -> pd.Series:
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.7:
            return (num > 0).astype(int)
        pos = {"1", "yes", "true", "positive", "y"}
        return s.map(lambda x: 1 if str(x).lower() in pos else 0)

    risk_cols = [
        c for c in binary_cols
        if any(kw in _norm(c) for kw in _RISK_KEYWORDS)
    ]
    # Also include plain 0/1 binary columns not already in risk_cols
    for col in binary_cols:
        if col not in risk_cols:
            vals = set(str(v).lower() for v in df[col].dropna().unique())
            if vals <= {"0", "1"}:
                risk_cols.append(col)

    if len(risk_cols) >= 2:
        risk_series = sum(_to_binary_01(df[rc]) for rc in risk_cols)
        proposals.append({
            "name"           : "Risk_Score",
            "feature_type"   : "risk_score",
            "source_columns" : risk_cols,
            "description"    : (
                f"Cumulative count of {len(risk_cols)} risk conditions present "
                f"({', '.join(risk_cols)}). Higher = more at-risk."
            ),
            "logic"          : f"Sum of ({', '.join(risk_cols)})",
            "enabled"        : True,
            "preview_values" : {
                str(k): int(v)
                for k, v in risk_series.value_counts().sort_index().items()
            },
            "is_domain_rule" : True,
        })

    return proposals


# ── Apply Feature Specs to DataFrame ──────────────────────────────────

def _to_binary_01(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.7:
        return (num > 0).astype(int)
    pos = {"1", "yes", "true", "positive", "y"}
    return s.map(lambda x: 1 if str(x).lower() in pos else 0)


def apply_features(
    df: pd.DataFrame,
    feature_specs: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply a list of feature specs (only those with enabled=True).
    Returns (augmented_df, log_messages).
    Original features are NEVER removed.
    """
    df   = df.copy()
    logs = []

    for spec in feature_specs:
        if not spec.get("enabled", True):
            continue
        name     = spec["name"]
        ftype    = spec.get("feature_type", "manual")
        src_cols = spec.get("source_columns", [])

        # Skip if source columns don't exist
        missing = [c for c in src_cols if c not in df.columns]
        if missing:
            logs.append(f"⚠️ Skipped '{name}': source column(s) not found: {missing}")
            continue

        try:
            if ftype == "categorical":
                col     = src_cols[0]
                matched = _match_domain_rule(col)
                if matched:
                    series, _ = _apply_domain_bin(df, col, DOMAIN_BINS[matched])
                else:
                    series, _ = _apply_quantile_bin(df, col)
                le = LabelEncoder()
                df[name] = le.fit_transform(series.fillna("Unknown").astype(str))
                logs.append(f"✅ '{name}': binned '{col}' → label-encoded integer.")

            elif ftype == "interaction":
                c1, c2   = src_cols[0], src_cols[1]
                val      = (pd.to_numeric(df[c1], errors="coerce") *
                            pd.to_numeric(df[c2], errors="coerce"))
                df[name] = val.fillna(0)
                logs.append(f"✅ '{name}': {c1} × {c2}.")

            elif ftype == "risk_score":
                df[name] = sum(_to_binary_01(df[rc]) for rc in src_cols)
                logs.append(f"✅ '{name}': sum of {len(src_cols)} risk indicators.")

            elif ftype == "manual":
                df = _apply_manual(df, name, src_cols, spec.get("operation", "add"), spec)
                logs.append(f"✅ Manual '{name}' created ({spec.get('operation')}).")

        except Exception as exc:
            logs.append(f"⚠️ Failed to create '{name}': {exc}")

    return df, logs


# ── Manual Feature Creation ────────────────────────────────────────────

def _apply_manual(
    df: pd.DataFrame,
    name: str,
    src_cols: List[str],
    operation: str,
    spec: Dict,
) -> pd.DataFrame:
    nums = [pd.to_numeric(df[c], errors="coerce") for c in src_cols]

    if operation == "add":
        result = sum(nums)
    elif operation == "multiply":
        result = nums[0].copy()
        for s in nums[1:]:
            result = result * s
    elif operation == "ratio":
        if len(nums) < 2:
            raise ValueError("Ratio requires exactly 2 source columns.")
        result = nums[0] / nums[1].replace(0, np.nan)
    elif operation == "bin":
        n_bins = int(spec.get("bin_count", 4))
        labels = spec.get("bin_labels") or [f"Q{i+1}" for i in range(n_bins)]
        try:
            binned = pd.qcut(nums[0], q=n_bins, labels=labels[:n_bins], duplicates="drop")
        except Exception:
            binned = pd.cut(nums[0], bins=n_bins, labels=labels[:n_bins])
        le = LabelEncoder()
        df[name] = le.fit_transform(binned.astype(str).fillna("Unknown"))
        df[name] = df[name].fillna(0)
        return df
    else:
        raise ValueError(f"Unknown manual operation: '{operation}'.")

    df[name] = result.fillna(0)
    return df


def preview_manual_feature(
    df: pd.DataFrame,
    name: str,
    src_cols: List[str],
    operation: str,
    spec: Dict,
) -> Dict[str, Any]:
    """Compute a manual feature preview without persisting anything."""
    try:
        df_copy = _apply_manual(df.copy(), name, src_cols, operation, spec)
        series  = df_copy[name]
        return {
            "ok"          : True,
            "name"        : name,
            "sample"      : [str(round(v, 4)) if isinstance(v, float) else str(v)
                             for v in series.head(10).tolist()],
            "min"         : round(float(series.min()),  4) if series.dtype != object else None,
            "max"         : round(float(series.max()),  4) if series.dtype != object else None,
            "mean"        : round(float(series.mean()), 4) if series.dtype != object else None,
            "unique_count": int(series.nunique()),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
