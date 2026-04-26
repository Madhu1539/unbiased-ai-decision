"""
preprocessing.py  —  /api/preprocess

Production-grade sklearn Pipeline preprocessing with strict leakage prevention.

ALL transformations live inside a single Pipeline:
    Pipeline([
        ('preprocessor', ColumnTransformer(..., remainder='passthrough')),
        ('variance_filter', VarianceThreshold(...)),
        ('correlation_filter', CorrelationDropper()),
    ])

Custom transformers (all implement BaseEstimator + TransformerMixin):
    OutlierCapper         — IQR bounds fit on X_train
    RareCategoryGrouper   — frequency map fit on X_train
    FrequencyEncoder      — frequency encoding fit on X_train
    SafeLabelEncoder      — ordinal label encoding with unseen→−1
    SimpleTargetEncoder   — smoothed target encoding, fit on (X_train, y_train)
    CorrelationDropper    — correlation matrix fit on X_train (numpy indices)

Endpoints:
    GET  /api/preprocess/analyze  → column analysis + explosion estimates
    POST /api/preprocess/apply    → build, fit, save, return summary
    GET  /api/preprocess/preview  → X_train head + feature names
    GET  /api/preprocess/status   → pipeline applied or not
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder,
    PowerTransformer, RobustScaler, StandardScaler,
)
from starlette.concurrency import run_in_threadpool

from backend.services.session_store import DATA_DIR, store
from backend.utils.helpers import safe_json

router = APIRouter(prefix="/api/preprocess", tags=["Preprocessing"])
logger = logging.getLogger(__name__)

PIPELINE_PATH  = os.path.join(DATA_DIR, "preprocessing_pipeline.pkl")
METADATA_PATH  = os.path.join(DATA_DIR, "preprocessing_metadata.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Transformers  (ALL return numpy arrays, ALL compatible with joblib)
# ═══════════════════════════════════════════════════════════════════════════════

class OutlierCapper(BaseEstimator, TransformerMixin):
    """IQR-based capping. Bounds computed on X_train only."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        arr = np.array(X, dtype=float)
        q1  = np.nanpercentile(arr, 25, axis=0)
        q3  = np.nanpercentile(arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X, y=None):
        arr = np.array(X, dtype=float).copy()
        for i in range(arr.shape[1]):
            arr[:, i] = np.clip(arr[:, i], self.lower_[i], self.upper_[i])
        return arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Maps infrequent categories → 'Other'. Frequency computed on X_train."""

    def __init__(self, threshold: float = 0.05, fill_value: str = "Other"):
        self.threshold  = threshold
        self.fill_value = fill_value

    def fit(self, X, y=None):
        arr = np.array(X, dtype=object)
        self.frequent_: List[set] = []
        n = len(arr)
        for i in range(arr.shape[1]):
            freq = pd.Series(arr[:, i].astype(str)).value_counts(normalize=True)
            self.frequent_.append(set(freq[freq >= self.threshold].index))
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X, y=None):
        arr = np.array(X, dtype=object).copy()
        for i in range(arr.shape[1]):
            mask = ~pd.Series(arr[:, i].astype(str)).isin(self.frequent_[i])
            arr[mask.values, i] = self.fill_value
        return arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.array([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replaces each category with its frequency in X_train. Returns float array."""

    def fit(self, X, y=None):
        arr = np.array(X, dtype=object)
        self.freq_maps_: List[Dict] = []
        n = len(arr)
        for i in range(arr.shape[1]):
            freq = pd.Series(arr[:, i].astype(str)).value_counts()
            self.freq_maps_.append((freq / n).to_dict())
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X, y=None):
        arr = np.array(X, dtype=object)
        out = np.zeros((arr.shape[0], arr.shape[1]), dtype=float)
        for i, fmap in enumerate(self.freq_maps_):
            out[:, i] = pd.Series(arr[:, i].astype(str)).map(fmap).fillna(0.0).values
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([f"{n}_freq" for n in input_features], dtype=object)
        return np.array([f"x{i}_freq" for i in range(self.n_features_in_)], dtype=object)


class SafeLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Label encoding with unseen-category safety (maps to -1).
    WARNING: introduces ordinal relationships — only valid for ordinal data
    or tree-based models.
    """

    def fit(self, X, y=None):
        arr = np.array(X, dtype=object)
        self.mappings_: List[Dict] = []
        for i in range(arr.shape[1]):
            cats = sorted(set(arr[:, i].astype(str)))
            self.mappings_.append({c: j for j, c in enumerate(cats)})
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X, y=None):
        arr = np.array(X, dtype=object)
        out = np.full((arr.shape[0], arr.shape[1]), -1, dtype=float)
        for i, mapping in enumerate(self.mappings_):
            out[:, i] = pd.Series(arr[:, i].astype(str)).map(mapping).fillna(-1).values
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([f"{n}_label" for n in input_features], dtype=object)
        return np.array([f"x{i}_label" for i in range(self.n_features_in_)], dtype=object)


class SimpleTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target encoding (fit on X_train, y_train).
    ⚠ Leakage risk without cross-fitting — use with caution.
    """

    def __init__(self, smooth: int = 10):
        self.smooth = smooth

    def fit(self, X, y):
        arr    = np.array(X, dtype=object)
        y_arr  = np.array(y, dtype=float)
        self.global_mean_ = float(np.nanmean(y_arr))
        self.target_maps_: List[Dict] = []
        for i in range(arr.shape[1]):
            df  = pd.DataFrame({"cat": arr[:, i].astype(str), "y": y_arr})
            grp = df.groupby("cat")["y"].agg(["sum", "count"])
            smoothed = (grp["sum"] + self.smooth * self.global_mean_) / (grp["count"] + self.smooth)
            self.target_maps_.append(smoothed.to_dict())
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X, y=None):
        arr = np.array(X, dtype=object)
        out = np.full((arr.shape[0], arr.shape[1]), self.global_mean_, dtype=float)
        for i, tmap in enumerate(self.target_maps_):
            out[:, i] = pd.Series(arr[:, i].astype(str)).map(tmap).fillna(self.global_mean_).values
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([f"{n}_target" for n in input_features], dtype=object)
        return np.array([f"x{i}_target" for i in range(self.n_features_in_)], dtype=object)


class CorrelationDropper(BaseEstimator, TransformerMixin):
    """
    Drops one column from each highly correlated pair.
    Fully inside the pipeline. Works with numpy arrays internally.
    fit() computes correlation on X_train only.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        arr = np.array(X, dtype=float)
        # Avoid crash on constant columns (std=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(arr, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        n    = arr.shape[1]
        # Upper triangle pairs
        self.drop_indices_: List[int] = []
        seen: set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if j not in seen and abs(corr[i, j]) > self.threshold:
                    self.drop_indices_.append(j)
                    seen.add(j)
        self.keep_indices_  = [i for i in range(n) if i not in self.drop_indices_]
        self.n_features_in_ = n
        return self

    def transform(self, X, y=None):
        arr = np.array(X, dtype=float)
        return arr[:, self.keep_indices_]

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray([input_features[i] for i in self.keep_indices_], dtype=object)
        return np.array([f"x{i}" for i in self.keep_indices_], dtype=object)


# ═══════════════════════════════════════════════════════════════════════════════
# Request Schema
# ═══════════════════════════════════════════════════════════════════════════════

class PreprocessRequest(BaseModel):
    num_imputer:              str        = "median"        # mean | median | knn | none
    cat_imputer:              str        = "most_frequent" # most_frequent | constant | none
    outlier_handling:         str        = "iqr"           # iqr | none
    # NOTE: handle_skewness and scaler are intentionally REMOVED.
    # They belong in the Training pipeline (post-split, fitted on X_train only).
    rare_threshold:           float      = 0.05
    low_card_encoding:        str        = "ohe"           # ohe | frequency | label
    high_card_encoding:       str        = "frequency"     # frequency | label
    high_card_threshold:      int        = 15
    drop_correlated:          bool       = True
    correlation_threshold:    float      = 0.95
    force_reapply:            bool       = False
    drop_cols_before_pipeline: List[str] = []


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _require_processed_df():
    """
    Return (X, y, target) from the full dataset before any split.
    Falls back to raw_df if processed_df is not yet set.
    Never uses X_train/X_test.
    """
    df = store.get("processed_df")
    if df is None:
        df = store.get("raw_df")      # fallback before cleaning is complete
    if df is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset found. Upload a CSV first.",
        )
    # Persist as processed_df if it wasn't already
    if store.get("processed_df") is None:
        store.set("processed_df", df)

    target = store.get("target_column")
    if not target or target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Target column not set. Complete Data Upload first.",
        )
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    return X, y, target


def _get_scaler(name: str):
    if name == "none":    return None
    if name == "minmax":  return MinMaxScaler()
    if name == "robust":  return RobustScaler()
    return StandardScaler()   # "standard" or "auto"


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)


def _cat_encoder(strategy: str, y_train=None):
    if strategy == "frequency": return FrequencyEncoder()
    if strategy == "label":     return SafeLabelEncoder()
    if strategy == "target":
        if y_train is None:
            raise HTTPException(status_code=400, detail="Target encoding requires y_train — ensure split exists.")
        return SimpleTargetEncoder()
    return _make_ohe()   # "ohe" / default


def _build_pipeline(
    X_train: pd.DataFrame,
    config: PreprocessRequest,
    y_train=None,
) -> Tuple[Pipeline, List[str], List[str], List[str]]:
    """Build and return the full sklearn Pipeline + detected column lists."""
    num_cols       = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_all   = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols_low   = [c for c in cat_cols_all if X_train[c].nunique(dropna=True) <= config.high_card_threshold]
    cat_cols_high  = [c for c in cat_cols_all if X_train[c].nunique(dropna=True) >  config.high_card_threshold]

    # ── Numeric pipeline ─────────────────────────────────────────────
    # NO scaling or skewness here — those belong in the Training Pipeline.
    num_steps: List = []
    if config.num_imputer == "knn":
        num_steps.append(("imputer", KNNImputer(n_neighbors=5, add_indicator=False)))
    elif config.num_imputer == "mean":
        num_steps.append(("imputer", SimpleImputer(strategy="mean")))
    elif config.num_imputer != "none":
        num_steps.append(("imputer", SimpleImputer(strategy="median")))

    if config.outlier_handling == "iqr":
        num_steps.append(("outlier_capper", OutlierCapper(factor=1.5)))

    num_pipeline = Pipeline(num_steps) if num_steps else "passthrough"

    # ── Categorical low-cardinality ───────────────────────────────────
    cat_low_steps: List = []
    if config.cat_imputer == "none":
        pass  # No imputer — only valid when cat cols have no missing values
    elif config.cat_imputer == "constant":
        cat_low_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="Missing")))
    else:
        cat_low_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    cat_low_steps.append(("rare_grouper", RareCategoryGrouper(threshold=config.rare_threshold)))
    cat_low_steps.append(("encoder", _cat_encoder(config.low_card_encoding, y_train)))
    cat_low_pipeline = Pipeline(cat_low_steps)

    # ── Categorical high-cardinality ──────────────────────────────────
    cat_high_steps: List = []
    cat_high_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="Missing")))
    cat_high_steps.append(("rare_grouper", RareCategoryGrouper(threshold=config.rare_threshold)))
    cat_high_steps.append(("encoder", _cat_encoder(config.high_card_encoding, y_train)))
    cat_high_pipeline = Pipeline(cat_high_steps)

    # ── ColumnTransformer ─────────────────────────────────────────────
    transformers = []
    if num_cols:       transformers.append(("num",      num_pipeline,      num_cols))
    if cat_cols_low:   transformers.append(("cat_low",  cat_low_pipeline,  cat_cols_low))
    if cat_cols_high:  transformers.append(("cat_high", cat_high_pipeline, cat_cols_high))

    try:
        col_transformer = ColumnTransformer(
            transformers, remainder="passthrough", verbose_feature_names_out=False,
        )
    except TypeError:
        col_transformer = ColumnTransformer(transformers, remainder="passthrough")

    # ── Full pipeline (single object, saved to disk) ──────────────────
    corr_dropper = CorrelationDropper(threshold=config.correlation_threshold)
    pipe = Pipeline([
        ("preprocessor",     col_transformer),
        ("variance_filter",  VarianceThreshold(threshold=0.0)),
        ("correlation_filter", corr_dropper if config.drop_correlated else "passthrough"),
    ])

    return pipe, num_cols, cat_cols_low, cat_cols_high


def _extract_feature_names(pipeline: Pipeline, fallback_n: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Chain get_feature_names_out() through preprocessor → variance_filter → correlation_filter.
    Returns (final_names, variance_dropped, corr_dropped).
    """
    try:
        ct_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        return [f"feature_{i}" for i in range(fallback_n)], [], []

    # After VarianceThreshold
    vt = pipeline.named_steps["variance_filter"]
    vt_mask    = vt.get_support()
    all_names  = np.asarray(ct_names, dtype=object)
    var_dropped = [str(n) for n, keep in zip(all_names, vt_mask) if not keep]
    vt_names    = all_names[vt_mask]

    # After CorrelationDropper (if used)
    cd = pipeline.named_steps.get("correlation_filter")
    if cd is not None and cd != "passthrough" and hasattr(cd, "drop_indices_"):
        final_arr   = cd.get_feature_names_out(vt_names)
        drop_idx    = cd.drop_indices_
        corr_dropped= [str(vt_names[i]) for i in drop_idx]
    else:
        final_arr   = vt_names
        corr_dropped= []

    final_names = [_clean(str(n)) for n in final_arr]
    return final_names, var_dropped, corr_dropped


def _clean(name: str) -> str:
    """Remove pipeline prefixes added by ColumnTransformer."""
    for p in ("num__", "cat_low__", "cat_high__", "remainder__", "cat__"):
        if name.startswith(p):
            return name[len(p):]
    return name


def _build_feature_mapping(
    num_cols, cat_cols_low, cat_cols_high,
    final_names: List[str],
    config: PreprocessRequest,
) -> List[Dict]:
    """Build original→transformed feature mapping."""
    mapping = []
    # Numeric: 1-to-1
    for c in num_cols:
        match = [f for f in final_names if f == c or f.startswith(c + "_")]
        mapping.append({"original": c, "transformed": match or [c], "type": "numeric"})
    # Low cardinality cat
    for c in cat_cols_low:
        match = [f for f in final_names if f == c or f.startswith(c + "_") or f.endswith(f"_{c}")]
        mapping.append({"original": c, "transformed": match or [c], "type": f"cat_low ({config.low_card_encoding})"})
    # High cardinality cat
    for c in cat_cols_high:
        match = [f for f in final_names if f == c or f.startswith(c + "_") or f.endswith(f"_{c}")]
        mapping.append({"original": c, "transformed": match or [c], "type": f"cat_high ({config.high_card_encoding})"})
    return mapping


def _build_summary(num_cols, cat_cols_low, cat_cols_high, config: PreprocessRequest) -> List[Dict]:
    rows = []
    for c in num_cols:
        ops = [f"{config.num_imputer} imputation"]
        if config.outlier_handling == "iqr": ops.append("IQR outlier capping")
        rows.append({"feature": c, "type": "numeric", "ops": ops})
    for c in cat_cols_low:
        ops = [f"{config.cat_imputer} imputation",
               f"rare grouping (freq < {int(config.rare_threshold*100)}%)",
               f"{config.low_card_encoding.upper()} encoding"]
        rows.append({"feature": c, "type": "categorical_low", "ops": ops})
    for c in cat_cols_high:
        ops = ["constant imputation (Missing)",
               f"rare grouping (freq < {int(config.rare_threshold*100)}%)",
               f"{config.high_card_encoding.upper()} encoding"]
        rows.append({"feature": c, "type": "categorical_high", "ops": ops})
    return rows


def _save_metadata(
    config: PreprocessRequest,
    feature_names_before: List[str],
    feature_names_after:  List[str],
    var_dropped: List[str],
    corr_dropped: List[str],
    t_summary: List[Dict],
) -> None:
    meta = {
        "pipeline_version":   "2.0",
        "created_at":         datetime.now(timezone.utc).isoformat(),
        "random_state":       None,
        "library_versions": {
            "scikit-learn":   sklearn.__version__,
            "pandas":         pd.__version__,
            "numpy":          np.__version__,
            "scipy":          scipy.__version__,
        },
        "config":             config.model_dump() if hasattr(config, "model_dump") else config.dict(),
        "feature_names_before": feature_names_before,
        "feature_names_after":  feature_names_after,
        "n_features_before":    len(feature_names_before),
        "n_features_after":     len(feature_names_after),
        "dropped": {
            "variance":    var_dropped,
            "correlation": corr_dropped,
        },
        "transformation_summary": t_summary,
    }
    try:
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
    except Exception as e:
        logger.warning("[Preprocess] Could not save metadata: %s", e)


def _check_metadata_versions() -> List[str]:
    """Compare stored library versions with current. Return mismatch warnings."""
    if not os.path.exists(METADATA_PATH):
        return []
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        stored  = meta.get("library_versions", {})
        current = {
            "scikit-learn": sklearn.__version__,
            "pandas":       pd.__version__,
            "numpy":        np.__version__,
            "scipy":        scipy.__version__,
        }
        mismatches = []
        for lib, ver in current.items():
            if stored.get(lib) and stored[lib] != ver:
                mismatches.append(
                    f"{lib}: pipeline saved with {stored[lib]}, current is {ver}."
                )
        return mismatches
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/analyze", summary="Analyze full dataset columns — types, warnings, explosion estimate")
async def analyze():
    """Scans full processed_df (excluding target). No split required."""
    try:
        X_train, _, _ = _require_processed_df()
        n = len(X_train)

        num_cols   = X_train.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols   = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

        column_info: List[Dict] = []
        global_warnings: List[str] = []

        for col in X_train.columns:
            s             = X_train[col]
            missing       = int(s.isnull().sum())
            missing_pct   = round(100 * missing / n, 1)
            is_num        = col in num_cols
            n_unique      = int(s.nunique(dropna=True))

            entry: Dict[str, Any] = {
                "name":        col,
                "type":        "numeric" if is_num else "categorical",
                "n_unique":    n_unique,
                "missing":     missing,
                "missing_pct": missing_pct,
                "warn_missing":missing_pct > 40,
                "warn_card":   not is_num and n_unique > 20,
                "skewness":    None,
                "needs_transform": False,
            }

            if is_num:
                vals = s.dropna()
                skew = float(vals.skew()) if len(vals) > 3 else 0.0
                entry["skewness"]        = round(skew, 3)
                entry["needs_transform"] = abs(skew) > 1.0

            column_info.append(entry)

            if missing_pct > 40:
                global_warnings.append(f"'{col}' has {missing_pct}% missing — consider dropping.")
            if not is_num and n_unique > 50:
                global_warnings.append(f"'{col}' has very high cardinality ({n_unique} unique). Frequency or Target encoding recommended.")

        # Correlation — pairs above 0.85 flagged; severe = above 0.95
        corr_pairs: List[Dict] = []
        corr_pairs_085: List[Dict] = []   # used for recommendation threshold
        if len(num_cols) >= 2:
            with np.errstate(invalid="ignore"):
                corr_arr = X_train[num_cols].corr().abs()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    val = float(corr_arr.iloc[i, j])
                    if val > 0.85:
                        pair = {
                            "col_a": num_cols[i], "col_b": num_cols[j],
                            "corr": round(val, 4), "severe": val > 0.95,
                        }
                        corr_pairs_085.append(pair)
                        if val > 0.90:          # display table threshold stays at 0.90
                            corr_pairs.append(pair)
            if any(p["severe"] for p in corr_pairs):
                global_warnings.append(
                    f"{sum(1 for p in corr_pairs if p['severe'])} severely correlated pair(s) detected (>0.95). "
                    "CorrelationDropper will remove one per pair."
                )

        # Skewed (threshold 0.5 for advisory, 1.0 for PowerTransformer recommendation)
        skewed_cols      = [c["name"] for c in column_info if c.get("needs_transform")]
        mildly_skewed    = [c["name"] for c in column_info
                            if c.get("skewness") is not None and abs(c["skewness"]) > 0.5]

        # ── Task type (regression / classification) from session ─────────────
        task_type = store.get("task_type") or "unknown"   # set after training

        # ── Missing value facts (exact counts, not % thresholds) ─────────────
        num_missing_cols    = [c for c in column_info if c["type"] == "numeric"     and c["missing"] > 0]
        cat_missing_cols    = [c for c in column_info if c["type"] == "categorical" and c["missing"] > 0]
        has_num_missing     = len(num_missing_cols) > 0
        has_cat_missing     = len(cat_missing_cols) > 0
        max_num_missing_pct = max((c["missing_pct"] for c in num_missing_cols), default=0)

        # ── Outlier detection via IQR spread AND Z-score ─────────────────────
        has_heavy_outliers = False
        outlier_cols: List[str] = []
        for _col in num_cols:
            _vals = X_train[_col].dropna()
            if len(_vals) < 10 or float(_vals.std()) == 0:
                continue
            # IQR method
            _q1, _q3 = float(_vals.quantile(0.25)), float(_vals.quantile(0.75))
            _iqr = _q3 - _q1
            _iqr_outliers = ((_vals < _q1 - 1.5 * _iqr) | (_vals > _q3 + 1.5 * _iqr)).sum()
            # Z-score method (|z| > 3)
            _z_outliers = (np.abs((_vals - _vals.mean()) / _vals.std()) > 3).sum()
            if _iqr_outliers / len(_vals) > 0.05 or _z_outliers / len(_vals) > 0.03:
                outlier_cols.append(_col)
                has_heavy_outliers = True

        many_skewed   = len(skewed_cols) > max(1, len(num_cols) / 2)
        needs_outlier = has_heavy_outliers or many_skewed

        # ── Cardinality analysis ─────────────────────────────────────────────
        high_card_cols = [c for c in cat_cols if X_train[c].nunique() > 15]
        low_card_cols  = [c for c in cat_cols if X_train[c].nunique() <= 15]
        has_cat_cols   = len(cat_cols) > 0
        has_high_card  = len(high_card_cols) > 0

        # ── Per-knob data-driven decisions ───────────────────────────────────

        # 1. Numeric imputer — "none" when dataset is clean
        if not has_num_missing:
            rec_num_imp    = "none"
            reason_num_imp = "No missing values in numeric columns — imputation not needed."
        elif has_num_missing and n < 10_000:
            rec_num_imp    = "knn"
            reason_num_imp = (
                f"{len(num_missing_cols)} numeric column(s) have missing values "
                f"(max {max_num_missing_pct:.0f}%). "
                "Dataset is small enough for KNN — most accurate strategy."
            )
        else:
            rec_num_imp    = "median"
            reason_num_imp = (
                f"{len(num_missing_cols)} numeric column(s) have missing values. "
                f"Dataset has {n:,} rows — Median chosen (KNN too slow at this scale)."
            )

        # 2. Categorical imputer — "none" when dataset is clean
        if not has_cat_cols:
            rec_cat_imp    = "none"
            reason_cat_imp = "No categorical columns in dataset — categorical imputer not applicable."
        elif not has_cat_missing:
            rec_cat_imp    = "none"
            reason_cat_imp = "No missing values in categorical columns — imputation not needed."
        else:
            rec_cat_imp    = "constant"
            reason_cat_imp = (
                f"{len(cat_missing_cols)} categorical column(s) have missing values. "
                "Fill 'Missing' preserves the absence of information as a signal."
            )

        # 3. Outlier handling — "none" when data is clean
        if needs_outlier:
            parts: List[str] = []
            if outlier_cols:
                parts.append(f"{len(outlier_cols)} column(s) have >5% IQR outliers")
            if many_skewed:
                parts.append(f"{len(skewed_cols)} column(s) are heavily skewed (|skew|>1.0)")
            rec_outlier    = "iqr"
            reason_outlier = "IQR capping recommended: " + "; ".join(parts) + "."
        else:
            rec_outlier    = "none"
            reason_outlier = (
                "No significant outliers detected (IQR or Z-score) and no heavy skewness — "
                "outlier handling not required."
            )

        # 4. Low-cardinality encoding
        if not has_cat_cols:
            rec_low_enc    = "ohe"      # harmless; no cat_cols_low will be passed
            reason_low_enc = "No categorical columns — low-cardinality encoding not applicable."
        else:
            rec_low_enc    = "ohe"
            reason_low_enc = (
                f"{len(low_card_cols)} low-cardinality column(s) detected. "
                "OneHot encoding is the safest default for nominal features."
            )

        # 5. High-cardinality encoding — task-type aware
        if not has_cat_cols:
            rec_high_enc    = "frequency"   # harmless
            reason_high_enc = "No categorical columns — high-cardinality encoding not applicable."
        elif has_high_card:
            sample_cols     = ", ".join(high_card_cols[:3]) + ("…" if len(high_card_cols) > 3 else "")
            if task_type == "regression":
                rec_high_enc    = "frequency"
                reason_high_enc = (
                    f"{len(high_card_cols)} high-cardinality column(s) ({sample_cols}). "
                    "Frequency encoding chosen (regression task — target encoding has leakage risk)."
                )
            else:
                rec_high_enc    = "frequency"
                reason_high_enc = (
                    f"{len(high_card_cols)} high-cardinality column(s) ({sample_cols}). "
                    "Frequency encoding avoids OHE explosion."
                )
        else:
            rec_high_enc    = "ohe"
            reason_high_enc = "All categorical columns have low cardinality — OHE is fine."

        # 6. Correlation drop — recommend only when pairs > 0.85 exist
        n_pairs_085 = len(corr_pairs_085)
        if n_pairs_085 > 0:
            rec_drop_corr    = True
            reason_drop_corr = (
                f"{n_pairs_085} feature pair(s) have correlation > 0.85 — "
                "dropping one from each pair reduces redundancy and speeds training."
            )
        else:
            rec_drop_corr    = False
            reason_drop_corr = (
                "No features with correlation > 0.85 detected — "
                "CorrelationDropper will have no effect; disabled."
            )

        auto_hints = {
            # Imputers
            "recommended_num_imputer":   rec_num_imp,
            "reason_num_imputer":        reason_num_imp,
            "recommended_cat_imputer":   rec_cat_imp,
            "reason_cat_imputer":        reason_cat_imp,
            # Outliers
            "recommended_outlier":       rec_outlier,
            "reason_outlier":            reason_outlier,
            # Encoding
            "recommended_low_card_enc":  rec_low_enc,
            "reason_low_card_enc":       reason_low_enc,
            "recommended_high_card_enc": rec_high_enc,
            "reason_high_card_enc":      reason_high_enc,
            # Correlation
            "recommended_drop_corr":     rec_drop_corr,
            "reason_drop_corr":          reason_drop_corr,
            # Dataset metadata / info flags
            "task_type":                 task_type,
            "has_num_missing":           has_num_missing,
            "has_cat_missing":           has_cat_missing,
            "has_cat_cols":              has_cat_cols,
            "skewed_cols":               skewed_cols,
            "mildly_skewed_cols":        mildly_skewed,
            "outlier_cols":              outlier_cols,
            "handle_skewness_suggested": len(skewed_cols) > 0,
            "warn_knn_large":            n > 10_000,
            "recommended_scaler":        "robust" if needs_outlier else "standard",
        }


        # Feature explosion estimate per column (for UI estimator)
        # Assumes low-card=OHE (worst case), high-card=frequency (1 per col)
        explosion_per_col: Dict[str, int] = {}
        for col in num_cols:
            explosion_per_col[col] = 1
        for col in cat_cols:
            nu = X_train[col].nunique()
            if nu <= 15:
                explosion_per_col[col] = max(1, nu - 1)   # OHE drop=first
            else:
                explosion_per_col[col] = 1                 # freq/target/label

        version_warnings = _check_metadata_versions()

        return JSONResponse(content=safe_json({
            "n_rows":             n,
            "n_features":         len(X_train.columns),
            "num_cols":           num_cols,
            "cat_cols":           cat_cols,
            "skewed_cols":        skewed_cols,
            "column_info":        column_info,
            "corr_pairs":         corr_pairs,
            "warnings":           global_warnings,
            "version_warnings":   version_warnings,
            "auto_hints":         auto_hints,
            "explosion_per_col":  explosion_per_col,
        }))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Preprocess /analyze] %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@router.post("/apply", summary="Build pipeline on FULL dataset, transform, store back as processed_df")
async def apply_preprocessing(body: PreprocessRequest):
    """
    Builds a standard sklearn Pipeline and fits+transforms the FULL dataset
    (processed_df minus target) BEFORE any train/test split.

    Pipeline steps (no scaling/skewness — those belong in Training):
        ColumnTransformer(num: imputer+outlier, cat_low: imputer+rare+encoder,
                          cat_high: imputer+rare+encoder)
        → VarianceThreshold
        → CorrelationDropper (optional)

    Result is stored back as processed_df so the Split step sees clean numeric data.
    """
    try:
        X, y, target = _require_processed_df()

        # Drop user-selected high-missing columns
        cols_to_drop = [c for c in (body.drop_cols_before_pipeline or []) if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            logger.info("[Preprocess] Dropped cols: %s", cols_to_drop)

        # Re-run protection
        if store.get("preprocessing_pipeline") is not None and not body.force_reapply:
            raise HTTPException(
                status_code=409,
                detail="PIPELINE_EXISTS: Preprocessing already applied. Set force_reapply=true to overwrite.",
            )

        n_before = int(X.shape[1])
        features_before = list(X.columns)

        def _do_apply():
            pipeline, num_cols, cat_cols_low, cat_cols_high = _build_pipeline(X, body, y)

            # FIT + TRANSFORM on the full feature matrix
            X_arr = pipeline.fit_transform(X, y)

            final_names, var_dropped, corr_dropped = _extract_feature_names(
                pipeline, X_arr.shape[1]
            )
            if X_arr.shape[1] != len(final_names):
                final_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

            # Rebuild full DataFrame (features + target) as processed_df
            X_df = pd.DataFrame(X_arr, columns=final_names, index=X.index)
            processed_df_new = X_df.copy()
            processed_df_new[target] = y.values

            t_summary = _build_summary(num_cols, cat_cols_low, cat_cols_high, body)
            feat_map  = _build_feature_mapping(num_cols, cat_cols_low, cat_cols_high, final_names, body)

            pipeline_saved = False
            try:
                joblib.dump(pipeline, PIPELINE_PATH)
                pipeline_saved = True
            except Exception as e:
                logger.warning("[Preprocess] joblib.dump failed: %s", e)

            _save_metadata(body, features_before, final_names, var_dropped, corr_dropped, t_summary)
            return processed_df_new, pipeline, final_names, var_dropped, corr_dropped, t_summary, feat_map, num_cols, cat_cols_low, cat_cols_high, pipeline_saved

        processed_df_new, pipeline, final_names, var_dropped, corr_dropped, t_summary, feat_map, num_cols, cat_cols_low, cat_cols_high, pipeline_saved = await run_in_threadpool(_do_apply)

        # Persist: store transformed full df back as processed_df
        # Clear any stale split data so user must re-split after preprocessing
        store.update({
            "processed_df":           processed_df_new,
            "feature_columns":        final_names,
            "preprocessing_pipeline": pipeline,
            "preprocessing_log":      [f"{s['feature']}: {' → '.join(s['ops'])}" for s in t_summary],
            # Clear stale split — must re-run Split after preprocessing
            "X_train": None, "X_test": None, "y_train": None, "y_test": None,
        })

        logger.info(
            "[Preprocess] Done: %d → %d features | var_dropped=%d | corr_dropped=%d",
            n_before, len(final_names), len(var_dropped), len(corr_dropped),
        )

        return JSONResponse(content=safe_json({
            "message":              "Preprocessing applied to full dataset. Please re-run Split Data next.",
            "n_features_before":    n_before,
            "n_features_after":     len(final_names),
            "total_rows":           int(processed_df_new.shape[0]),
            "feature_names":        final_names,
            "feature_mapping":      feat_map,
            "dropped_variance":     var_dropped,
            "dropped_correlation":  corr_dropped,
            "transformation_summary": t_summary,
            "pipeline_saved":       pipeline_saved,
            "metadata_saved":       os.path.exists(METADATA_PATH),
        }))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Preprocess /apply] %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}")


@router.get("/preview", summary="Full dataset head + feature names + pipeline flag")
async def preview():
    """Shows the full processed_df (before split). Falls back to raw_df. No split required."""
    df = store.get("processed_df")
    if df is None:
        df = store.get("raw_df")
    target = store.get("target_column")
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded.")
    pp_done  = store.get("preprocessing_pipeline") is not None
    feature_cols = [c for c in df.columns if c != target] if target else list(df.columns)
    head = df[feature_cols].head(5).round(4).fillna("").astype(str).to_dict(orient="records")
    return JSONResponse(content=safe_json({
        "preprocessed": pp_done,
        "total_rows":   int(df.shape[0]),
        "n_features":   len(feature_cols),
        "features":     feature_cols,
        "head":         head,
    }))


@router.get("/status", summary="Whether preprocessing pipeline has been applied")
async def status():
    pipeline    = store.get("preprocessing_pipeline")
    df          = store.get("processed_df")
    if df is None:
        df = store.get("raw_df")
    target      = store.get("target_column")
    meta_exists = os.path.exists(METADATA_PATH)
    if pipeline is None:
        return JSONResponse(content={"done": False, "pipeline_saved": os.path.exists(PIPELINE_PATH)})
    feature_cols = [c for c in df.columns if c != target] if (df is not None and target) else \
                   (list(df.columns) if df is not None else [])
    return JSONResponse(content=safe_json({
        "done":             True,
        "n_features":       len(feature_cols),
        "features":         feature_cols,
        "total_rows":       int(df.shape[0]) if df is not None else 0,
        "pipeline_saved":   os.path.exists(PIPELINE_PATH),
        "metadata_saved":   meta_exists,
        "version_mismatch": _check_metadata_versions(),
    }))
