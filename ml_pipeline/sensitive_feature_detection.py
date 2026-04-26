"""
sensitive_feature_detection.py — Intelligent Sensitive Feature Classifier
=========================================================================
Implements strict Responsible AI principles to determine which columns
qualify as protected (sensitive) attributes eligible for bias & fairness
analysis.

Classification Categories
--------------------------
SENSITIVE             — Direct human identity attributes (gender, age, race …)
POTENTIALLY_SENSITIVE — Proxy attributes (income, education, location …)
NON_SENSITIVE         — Business / medical measurements / technical / ID columns
UNKNOWN               — Cannot determine; excluded by default (prefer False Negative)

Hard Constraints
----------------
  - NEVER classify business / numeric-performance columns as sensitive
  - NEVER allow fairness on arbitrary continuous numeric columns
  - NEVER include ID / index columns
  - Prefer FALSE NEGATIVE (block) over FALSE POSITIVE (wrong bias signal)
"""
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ── Keyword Dictionaries ───────────────────────────────────────────────

# SENSITIVE — Direct legally-protected human identity attributes
_SENSITIVE_KEYWORDS: Dict[str, List[str]] = {
    "gender": [
        "gender", "sex", "gend", "male", "female",
        "transgender", "non_binary", "nonbinary",
    ],
    "age": [
        "age", "age_group", "agegroup", "age_band", "ageband",
        "dob", "birth", "birthdate", "birth_year",
        "age_range", "age_category", "age_bracket",
    ],
    "race": [
        "race", "ethnicity", "ethnic", "caste", "tribe",
        "racial", "minority", "indigenous", "colour", "color",
    ],
    "religion": [
        "religion", "faith", "religious", "denomination",
        "sect", "community",
    ],
    "nationality": [
        "nationality", "citizen", "citizenship",
        "national_origin", "country_of_origin",
        "immigrant", "migrant", "native",
    ],
    "disability": [
        "disability", "disabled", "handicap",
        "impairment", "special_needs", "accessibility",
    ],
}

# POTENTIALLY SENSITIVE — Proxy attributes that can correlate with protected groups
_POTENTIALLY_SENSITIVE_KEYWORDS: Dict[str, List[str]] = {
    "income": [
        "income", "salary", "wage", "pay", "earning",
        "compensation", "annual_income", "household_income",
        "capital_gain", "capital_loss", "fnlwgt",
    ],
    "education": [
        "education", "degree", "qualification", "literacy",
        "school", "college", "university", "study", "academic",
        "education_num", "education_level",
    ],
    "location": [
        "location", "city", "state", "zip", "postal",
        "postcode", "region", "district", "area",
        "urban", "rural", "neighborhood", "province", "country",
    ],
    "occupation": [
        "occupation", "job", "profession", "employment",
        "career", "work", "role", "designation", "workclass",
        "work_class",
    ],
    "marital_status": [
        "marital", "married", "single", "divorced",
        "widowed", "spouse", "relationship",
    ],
}

# NON-SENSITIVE — Business/medical/technical/ID data — MUST NOT be used
_NON_SENSITIVE_KEYWORDS: List[str] = [
    # Business / operational performance
    "sales", "revenue", "profit", "price", "cost", "amount",
    "value", "tv", "radio", "newspaper", "budget", "spend",
    "expense", "purchase", "transaction", "balance", "credit",
    "loan", "debt", "payment", "fee", "tax", "interest",
    # Medical measurements (raw physiological numbers)
    "cholesterol", "glucose", "blood_pressure", "bp",
    "sugar", "heartrate", "heart_rate", "pulse", "insulin",
    "bmi", "temperature", "oxygen", "spo2", "hba1c",
    "triglyceride", "ldl", "hdl", "albumin",
    # ID / index columns
    "id", "index", "unnamed", "uuid", "key",
    "row_number", "record_id", "customer_id", "user_id",
    "account_id", "order_id", "product_id", "item_id",
    "patient_id", "employee_id", "serial",
    # Technical / system-generated
    "timestamp", "created_at", "updated_at", "modified",
    "created", "date", "time", "hour", "minute",
    # Behavioral / system analytics
    "click", "session", "pageview", "log", "event",
    "rating", "review", "feedback", "score",
]

# Values that strongly confirm a column is about gender
_GENDER_VALUE_HINTS = {
    "m", "f", "male", "female", "man", "woman",
    "boy", "girl", "other", "non-binary",
    "0", "1",  # binary-encoded gender
}

_BINARY_HINT_MAX_UNIQUE = 5   # for value-level confirmation checks


# ── Normalisation Helper ──────────────────────────────────────────────

def _norm(text: str) -> str:
    """Collapse separators and lowercase for fuzzy keyword matching."""
    return re.sub(r"[\s_\-./]", "", str(text).lower())


def _match_dict(col_norm: str, kw_dict: Dict[str, List[str]]) -> Tuple[str, bool]:
    """Return (matched_type, found) for the first hit in kw_dict."""
    for ktype, keywords in kw_dict.items():
        for kw in keywords:
            kw_n = _norm(kw)
            if kw_n in col_norm or col_norm == kw_n:
                return ktype, True
    return "", False


def _match_non_sensitive(col_norm: str) -> bool:
    return any(_norm(kw) in col_norm or col_norm == _norm(kw) for kw in _NON_SENSITIVE_KEYWORDS)


def _safe_samples(series: pd.Series, n: int = 6) -> List[Any]:
    return [str(v) for v in series.dropna().unique()[:n].tolist()]


# ── Per-Column Classifier ─────────────────────────────────────────────

def classify_column(col: str, series: pd.Series) -> Dict[str, Any]:
    """
    Classify a single column.

    Priority order
    --------------
    1. Non-sensitive hard-block (keyword match)
    2. ID / unique-value detection
    3. SENSITIVE keyword match  (+ optional value-level confirmation for gender)
    4. POTENTIALLY SENSITIVE keyword match
    5. High-cardinality continuous numeric → block
    6. UNKNOWN → block (prefer False Negative)
    """
    col_norm = _norm(col)
    n_unique = int(series.nunique())
    n_total  = max(int(len(series.dropna())), 1)
    samples  = _safe_samples(series)
    sample_lc = [s.lower() for s in samples]

    def _result(category, stype, confidence, reason, eligible):
        return {
            "column"                : col,
            "category"              : category,   # SENSITIVE | POTENTIALLY_SENSITIVE | NON_SENSITIVE | UNKNOWN
            "type"                  : stype,
            "confidence"            : confidence,  # HIGH | MEDIUM | LOW
            "reason"                : reason,
            "eligible_for_fairness" : eligible,
            "unique_values"         : n_unique,
            "sample_values"         : samples,
        }

    # ── 1. Non-sensitive hard block ────────────────────────────────
    if _match_non_sensitive(col_norm):
        return _result(
            "NON_SENSITIVE", "non_sensitive", "HIGH",
            f"'{col}' matches a known business/medical/ID/technical keyword. "
            "Excluded: comparing predictions across this feature has no ethical fairness meaning.",
            False,
        )

    # ── 2. ID / index detection ────────────────────────────────────
    if n_unique == n_total and n_total > 40:
        return _result(
            "NON_SENSITIVE", "id_column", "HIGH",
            f"Every value in '{col}' is unique ({n_total} rows) — this is an ID or index column.",
            False,
        )

    # ── 3. SENSITIVE keyword match ─────────────────────────────────
    stype, found = _match_dict(col_norm, _SENSITIVE_KEYWORDS)
    if found:
        # For gender specifically, validate sample values when cardinality is low
        value_ok = True
        if stype == "gender" and n_unique <= _BINARY_HINT_MAX_UNIQUE:
            value_ok = bool(set(sample_lc) & _GENDER_VALUE_HINTS)

        if value_ok:
            return _result(
                "SENSITIVE", stype, "HIGH",
                f"'{col}' directly represents '{stype}', a legally protected human identity attribute.",
                True,
            )

    # ── 4. POTENTIALLY SENSITIVE keyword match ─────────────────────
    pstype, found_ps = _match_dict(col_norm, _POTENTIALLY_SENSITIVE_KEYWORDS)
    if found_ps:
        return _result(
            "POTENTIALLY_SENSITIVE", pstype, "MEDIUM",
            f"'{col}' may proxy protected groups via '{pstype}'. "
            "Include only when fairness context explicitly requires it.",
            True,
        )

    # ── 5. High-cardinality continuous numeric → block ─────────────
    # Use pd.api.types to safely handle pandas extension dtypes (StringDtype,
    # BooleanDtype, etc.) that numpy cannot interpret via np.issubdtype.
    is_numeric        = pd.api.types.is_numeric_dtype(series)
    cardinality_ratio = n_unique / n_total
    if is_numeric and (n_unique > 20 or cardinality_ratio > 0.05):
        return _result(
            "NON_SENSITIVE", "continuous_numeric", "HIGH",
            f"'{col}' is a continuous numeric column ({n_unique} unique values). "
            "Splitting predictions across arbitrary numeric thresholds does not constitute "
            "meaningful fairness analysis.",
            False,
        )

    # ── 6. Unknown → block by default (prefer False Negative) ─────
    return _result(
        "UNKNOWN", "unknown", "LOW",
        f"Cannot confidently determine whether '{col}' represents a protected attribute. "
        "Excluded by default to avoid false bias signals.",
        False,
    )


# ── Dataset-Level Analysis ────────────────────────────────────────────

def detect_sensitive_features(
    df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    """
    Classify every non-target column and return a dataset-level fairness
    eligibility verdict.

    Returns
    -------
    {
      fairness_applicable          : bool
      eligible_columns             : list[str]   — shown in the dropdown
      sensitive_columns            : list[str]
      potentially_sensitive_columns: list[str]
      non_sensitive_columns        : list[str]
      unknown_columns              : list[str]
      total_columns_analysed       : int
      message                      : str         — user-facing explanation
      column_details               : list[dict]  — per-column results
    }
    """
    feature_cols = [c for c in df.columns if c != target_col]
    column_details: List[Dict[str, Any]] = []

    sensitive:             List[str] = []
    potentially_sensitive: List[str] = []
    non_sensitive:         List[str] = []
    unknown:               List[str] = []

    for col in feature_cols:
        result = classify_column(col, df[col])
        column_details.append(result)

        cat = result["category"]
        if cat == "SENSITIVE":
            sensitive.append(col)
        elif cat == "POTENTIALLY_SENSITIVE":
            potentially_sensitive.append(col)
        elif cat == "NON_SENSITIVE":
            non_sensitive.append(col)
        else:
            unknown.append(col)

    # ── Fairness eligibility decision ──────────────────────────────
    eligible_columns = sensitive + potentially_sensitive

    if sensitive:
        fairness_applicable = True
        message = (
            f"✅ {len(sensitive)} protected attribute(s) detected: "
            f"{', '.join(sensitive)}. Fairness analysis is applicable."
        )
    elif potentially_sensitive:
        fairness_applicable = True
        message = (
            f"⚠️ No direct protected attributes found. "
            f"{len(potentially_sensitive)} potentially sensitive feature(s) detected: "
            f"{', '.join(potentially_sensitive)}. "
            "These are proxy attributes — use with caution."
        )
    else:
        fairness_applicable = False
        message = (
            "⚠️ No valid sensitive (protected) features detected. "
            "Bias & fairness analysis is not applicable to this dataset. "
            "The columns present appear to represent business, medical measurement, "
            "or technical data — not human identity attributes."
        )

    return {
        "fairness_applicable"             : fairness_applicable,
        "eligible_columns"                : eligible_columns,
        "sensitive_columns"               : sensitive,
        "potentially_sensitive_columns"   : potentially_sensitive,
        "non_sensitive_columns"           : non_sensitive,
        "unknown_columns"                 : unknown,
        "total_columns_analysed"          : len(feature_cols),
        "message"                         : message,
        "column_details"                  : column_details,
    }
