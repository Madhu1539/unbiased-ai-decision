"""
preprocessing.py  —  ML pipeline preprocessing utilities.
---------------------------------------------------------
Performance-optimized for large datasets (50,000+ rows).

Key fixes:
  - Removed deprecated fillna(inplace=True) → use assignment instead
  - Eliminated redundant df.copy() in hot paths
  - Vectorised missing-value handling across all columns at once
  - IQR outlier mask built via vectorised operations, no column loop allocation
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


# ─────────────────────── Missing Values ────────────────────────────────
def handle_missing_values(
    df: pd.DataFrame, strategy: str = "mean"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fill or drop missing values.
    strategy: 'mean' | 'median' | 'mode' | 'drop'
    Returns: (cleaned_df, log_messages)

    NOTE: always works on a copy — caller's DataFrame is never mutated.
    """
    logs   = []
    df     = df.copy()
    total  = int(df.isnull().sum().sum())
    logs.append(f"Total missing cells before: {total}")

    if strategy == "drop":
        original_len = len(df)
        df = df.dropna().reset_index(drop=True)
        logs.append(f"Dropped {original_len - len(df)} rows containing NaN values.")
        return df, logs

    # ── Vectorised fill for numeric columns ────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if num_cols:
        missing_num = df[num_cols].isnull().sum()
        missing_num = missing_num[missing_num > 0]
        if len(missing_num):
            if strategy == "mean":
                fill_values = df[missing_num.index].mean()
            elif strategy == "median":
                fill_values = df[missing_num.index].median()
            else:  # mode fallback
                fill_values = df[missing_num.index].mode().iloc[0]
            # Single vectorised assignment — no per-column loop
            df[missing_num.index] = df[missing_num.index].fillna(fill_values)
            for col, cnt in missing_num.items():
                logs.append(
                    f"Column '{col}': filled {cnt} NaN(s) with "
                    f"{strategy} ({fill_values[col]:.4f})."
                )

    # ── Categorical fill (mode per column, unavoidable loop) ───────────
    for col in cat_cols:
        missing = int(df[col].isnull().sum())
        if missing == 0:
            continue
        fill = df[col].mode().iloc[0] if len(df[col].mode()) else "Unknown"
        df[col] = df[col].fillna(fill)          # ← no deprecated inplace=True
        logs.append(f"Column '{col}': filled {missing} NaN(s) with mode ('{fill}').")

    logs.append(f"Total missing cells after: {int(df.isnull().sum().sum())}")
    return df, logs


# ─────────────────────── Encoding ──────────────────────────────────────
def encode_categorical(
    df: pd.DataFrame,
    target_col: str,
    strategy: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], List[str]]:
    """
    Encode categorical (object-type) columns, EXCLUDING the target.
    Returns: (encoded_df, label_encoders_dict, logs)
    """
    df = df.copy()
    label_encoders: Dict[str, LabelEncoder] = {}
    logs: List[str] = []
    cat_cols = [
        c for c in df.select_dtypes(include="object").columns if c != target_col
    ]

    if not cat_cols:
        logs.append("No categorical columns found to encode.")
        return df, label_encoders, logs

    if strategy == "label":
        for col in cat_cols:
            le        = LabelEncoder()
            df[col]   = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            logs.append(f"Label-encoded '{col}' → {list(le.classes_[:5])} …")
    else:  # one-hot
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        logs.append(f"One-hot encoded columns: {cat_cols}")

    return df, label_encoders, logs


# ─────────────────────── Outlier Detection ─────────────────────────────
def detect_and_remove_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove rows containing outliers from numeric columns (excluding target).
    method: 'iqr' | 'zscore'

    Optimised: builds a single boolean mask from all columns at once
    instead of iterating and & -ing per column.
    """
    df          = df.copy()
    logs: List[str] = []
    original_len = len(df)
    num_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)

    if not num_cols:
        logs.append("No numeric feature columns to check for outliers.")
        return df, logs

    num_data = df[num_cols]

    if method == "iqr":
        Q1  = num_data.quantile(0.25)
        Q3  = num_data.quantile(0.75)
        IQR = Q3 - Q1
        # Vectorised: all columns at once → boolean DataFrame, then row-wise all()
        mask = ((num_data >= (Q1 - 1.5 * IQR)) & (num_data <= (Q3 + 1.5 * IQR))).all(axis=1)
    else:  # zscore — vectorised via (x - mean) / std on the whole matrix
        z    = (num_data - num_data.mean()) / num_data.std(ddof=0)
        mask = (z.abs() < 3).all(axis=1)

    # Log per-column outlier counts
    if method == "iqr":
        Q1  = num_data.quantile(0.25)
        Q3  = num_data.quantile(0.75)
        IQR = Q3 - Q1
        for col in num_cols:
            col_outliers = int(
                ((num_data[col] < Q1[col] - 1.5 * IQR[col]) |
                 (num_data[col] > Q3[col] + 1.5 * IQR[col])).sum()
            )
            if col_outliers:
                logs.append(f"Column '{col}': {col_outliers} outlier(s) detected.")
    else:
        z = (num_data - num_data.mean()) / num_data.std(ddof=0)
        for col in num_cols:
            col_outliers = int((z[col].abs() >= 3).sum())
            if col_outliers:
                logs.append(f"Column '{col}': {col_outliers} outlier(s) detected.")

    df      = df[mask].reset_index(drop=True)
    removed = original_len - len(df)
    logs.append(
        f"Removed {removed} outlier row(s). Dataset size: {original_len} → {len(df)}."
    )
    return df, logs


# ─────────────────────── Scaling ───────────────────────────────────────
def scale_features(
    df: pd.DataFrame,
    target_col: str,
    method: str = "standard",
) -> Tuple[pd.DataFrame, object, List[str]]:
    """
    Scale numeric feature columns (NOT the target).
    method: 'standard' | 'minmax'
    Returns: (scaled_df, fitted_scaler, logs)
    """
    df = df.copy()
    logs: List[str] = []
    num_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col
    ]

    if not num_cols:
        logs.append("No numeric feature columns to scale.")
        return df, None, logs

    scaler        = StandardScaler() if method == "standard" else MinMaxScaler()
    df[num_cols]  = scaler.fit_transform(df[num_cols])
    logs.append(
        f"Applied {'StandardScaler' if method == 'standard' else 'MinMaxScaler'} "
        f"to {len(num_cols)} numeric column(s)."
    )
    return df, scaler, logs
