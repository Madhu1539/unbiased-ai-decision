"""
training.py  —  ML model training utilities
Production-grade pipeline architecture:
  - Pre-split: data is already cleaned, encoded, and feature-engineered
  - Post-split: skewness correction + scaling + model wrapped in sklearn Pipeline
  - Optional: imblearn Pipeline when resampling is used
  - Pipeline is fitted on X_train ONLY; X_test is only transformed (never fit)
  - Full pipeline stored in session as 'post_split_pipeline' for predictions

Performance optimisations:
  - n_jobs=-1 injected automatically for parallelisable models
  - SVM/KNN guarded with automatic dataset-size check (>20k → fallback)
"""
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ── Threshold above which SVM/KNN are automatically replaced ──────────
LARGE_DATASET_THRESHOLD = 20_000

# ── Models that support n_jobs ────────────────────────────────────────
_NJOBS_MODELS = {
    "Random Forest", "KNN", "Logistic Regression",
}

# ── Model Catalogue ────────────────────────────────────────────────────
CLASSIFICATION_MODELS = {
    "Logistic Regression"  : LogisticRegression,
    "Random Forest"        : RandomForestClassifier,
    "Decision Tree"        : DecisionTreeClassifier,
    "SVM"                  : SVC,
    "KNN"                  : KNeighborsClassifier,
    "Gradient Boosting"    : GradientBoostingClassifier,
}

REGRESSION_MODELS = {
    "Linear Regression"    : LinearRegression,
    "Ridge Regression"     : Ridge,
    "Lasso Regression"     : Lasso,
    "Random Forest"        : RandomForestRegressor,
    "Decision Tree"        : DecisionTreeRegressor,
    "KNN"                  : KNeighborsRegressor,
    "SVR"                  : SVR,
    "Gradient Boosting"    : GradientBoostingRegressor,
}

if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["XGBoost"] = XGBClassifier
    REGRESSION_MODELS["XGBoost"]     = XGBRegressor

MODEL_INFO = {
    "Logistic Regression": {
        "pros": ["Fast", "Interpretable", "Low memory", "Scales well to large data"],
        "cons": ["Linear boundary only", "Sensitive to outliers"],
    },
    "Random Forest": {
        "pros": ["Handles non-linearity", "Feature importance", "Robust", "Parallelisable (n_jobs=-1)"],
        "cons": ["Slower training than linear models", "Black-box"],
    },
    "Decision Tree": {
        "pros": ["Highly interpretable", "No scaling needed"],
        "cons": ["Prone to overfitting", "Unstable"],
    },
    "SVM": {
        "pros": ["Effective in high dimensions", "Robust margin"],
        "cons": ["Very slow on large data (O(n²))", "Sensitive to scaling", "NOT recommended for >20k rows"],
    },
    "KNN": {
        "pros": ["Simple", "No training phase"],
        "cons": ["Very slow inference on large data", "Memory-intensive", "NOT recommended for >20k rows"],
    },
    "Gradient Boosting": {
        "pros": ["High accuracy", "Handles mixed types"],
        "cons": ["Slower training", "Many hyperparameters"],
    },
    "XGBoost": {
        "pros": ["State-of-the-art accuracy", "Built-in regularisation", "Supports parallelism"],
        "cons": ["Complex tuning", "Not as interpretable"],
    },
    "Linear Regression": {
        "pros": ["Fast", "Interpretable", "Scales to large data"],
        "cons": ["Assumes linearity", "Outlier-sensitive"],
    },
    "Ridge Regression": {
        "pros": ["Handles multicollinearity", "Simple", "Fast"],
        "cons": ["Still linear"],
    },
    "Lasso Regression": {
        "pros": ["Feature selection via sparsity"],
        "cons": ["Drops correlated features arbitrarily"],
    },
    "SVR": {
        "pros": ["Robust to outliers in prediction"],
        "cons": ["Very slow on large data (O(n²))", "NOT recommended for >20k rows"],
    },
}

# ── Slow-model fall-back mappings for large datasets ──────────────────
_SLOW_CLS_FALLBACK  = "Random Forest"   # replaces SVM / KNN  for classification
_SLOW_REG_FALLBACK  = "Random Forest"   # replaces SVR / KNN  for regression
_SLOW_MODELS        = {"SVM", "KNN", "SVR"}


def get_available_models(task_type: str) -> Dict[str, Any]:
    catalogue = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    return {
        name: MODEL_INFO.get(name, {"pros": [], "cons": []})
        for name in catalogue
    }


def _build_model(
    model_name: str,
    catalogue: Dict,
    hyperparams: Dict[str, Any],
    class_weights: Any,
    n_samples: int,
) -> Tuple[Any, str]:
    """
    Instantiate the model class with validated hyperparams.
    Returns (model_instance, actual_model_name_used).

    Large-dataset guard: if n_samples > LARGE_DATASET_THRESHOLD and the
    requested model is in _SLOW_MODELS, substitute with a faster model
    and emit a warning string in the return name.
    """
    logger = logging.getLogger(__name__)

    actual_name = model_name

    # ── Large-dataset guard ───────────────────────────────────────────
    if n_samples > LARGE_DATASET_THRESHOLD and model_name in _SLOW_MODELS:
        is_cls      = "SVM" in model_name or model_name == "KNN"
        fallback    = _SLOW_CLS_FALLBACK if "SVM" in model_name or model_name == "KNN" else _SLOW_REG_FALLBACK
        # pick fallback from whichever catalogue we're in
        if fallback not in catalogue:
            fallback = next(iter(catalogue))
        logger.warning(
            "[Training] %s is not suitable for %d samples — substituting %s",
            model_name, n_samples, fallback,
        )
        actual_name = fallback
        model_name  = fallback
        hyperparams = {}   # reset to safe defaults

    ModelClass   = catalogue[model_name]
    valid_params = inspect.signature(ModelClass.__init__).parameters
    safe_params  = {k: v for k, v in hyperparams.items() if k in valid_params}

    # ── Auto-inject n_jobs=-1 for parallelisable models ───────────────
    if "n_jobs" in valid_params and "n_jobs" not in safe_params:
        safe_params["n_jobs"] = -1

    # ── Inject class_weight if applicable ─────────────────────────────
    if class_weights is not None and "class_weight" in valid_params:
        safe_params["class_weight"] = class_weights

    return ModelClass(**safe_params), actual_name


# ─── Pipeline factory ────────────────────────────────────────────────────────

def _build_post_split_pipeline(
    clf,
    scaler_type: str = "standard",
    apply_skewness: bool = True,
    balancing_technique: Optional[str] = None,
    cat_indices: Optional[List[int]] = None,
) -> Any:
    """
    Build a production sklearn or imblearn Pipeline:

    Without resampling (or class_weight / none):
        sklearn.Pipeline([
            ("skewness", PowerTransformer),   # optional
            ("scaler",   StandardScaler),      # optional
            ("model",    clf),
        ])

    With resampling (imblearn required):
        imblearn.Pipeline([
            ("resampler", SMOTE / ...),
            ("skewness",  PowerTransformer),
            ("scaler",    StandardScaler),
            ("model",     clf),
        ])

    CRITICAL: fit() on X_train only. predict() / transform() on X_test.
    """
    logger = logging.getLogger(__name__)

    # ── Determine scaler step ─────────────────────────────────────────
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "none" or scaler_type is None:
        scaler = None
    else:                               # default: "standard"
        scaler = StandardScaler()

    # ── Determine skewness step ───────────────────────────────────────
    # PowerTransformer(method='yeo-johnson') handles negative values safely.
    pt = PowerTransformer(method="yeo-johnson", standardize=False) if apply_skewness else None

    # ── Determine resampler step ──────────────────────────────────────
    uses_resampler = (
        balancing_technique
        and balancing_technique not in ("none", "class_weight", None)
    )

    if uses_resampler:
        try:
            from imblearn.pipeline import Pipeline as ImbPipeline
        except ImportError:
            logger.warning(
                "[Pipeline] imbalanced-learn not installed — "
                "falling back to sklearn Pipeline without resampling."
            )
            uses_resampler = False

    # ── Build step list ───────────────────────────────────────────────
    steps = []

    if uses_resampler:
        resampler = _get_imblearn_resampler(balancing_technique, cat_indices)
        if resampler is not None:
            steps.append(("resampler", resampler))

    if pt is not None:
        steps.append(("skewness", pt))
    if scaler is not None:
        steps.append(("scaler", scaler))

    steps.append(("model", clf))

    step_names = [s[0] for s in steps]
    logger.info("[Pipeline] Building pipeline: %s", " → ".join(step_names))

    if uses_resampler and any(s[0] == "resampler" for s in steps):
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline(steps)
    else:
        return SklearnPipeline(steps)


def _get_imblearn_resampler(technique: str, cat_indices: Optional[List[int]] = None):
    """Return an imblearn resampler instance for the given technique."""
    logger = logging.getLogger(__name__)
    try:
        from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN, SMOTETomek
    except ImportError:
        logger.warning("[Pipeline] imbalanced-learn not available — no resampler.")
        return None

    mapping = {
        "smote":       SMOTE(random_state=42),
        "adasyn":      ADASYN(random_state=42),
        "undersample": RandomUnderSampler(random_state=42),
        "smoteenn":    SMOTEENN(random_state=42),
        "smotetomek":  SMOTETomek(random_state=42),
    }
    if technique == "smotenc":
        if not cat_indices:
            logger.warning("[Pipeline] SMOTENC requested but no cat_indices — using SMOTE instead.")
            return SMOTE(random_state=42)
        return SMOTENC(categorical_features=cat_indices, random_state=42)

    return mapping.get(technique)


# ─── Main training entry point ────────────────────────────────────────────────

def train_model(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    model_name: str,
    task_type: str,
    hyperparams: Dict[str, Any] = {},
    balancing_technique: str = None,
    cat_indices: list = None,
    use_calibration: bool = False,
    scaler_type: str = "standard",
    apply_skewness: bool = True,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Train the requested model using a leakage-free Pipeline.

    INPUTS (already split by /api/split or train route):
      X_train, X_test  : DataFrames or arrays (numeric, post-preprocessing)
      y_train, y_test  : Series or arrays

    PIPELINE (built and fit on X_train only):
      [resampler →] skewness_correction → scaler → model

    Returns: (pipeline, X_train_arr, X_test_arr, y_train_arr, y_test_arr, y_pred, y_prob)

    The returned 'pipeline' is the full fitted sklearn / imblearn Pipeline.
    Store it as 'post_split_pipeline' in the session for reuse in predictions.

    scaler_type     : 'standard' | 'minmax' | 'none'
    apply_skewness  : bool — whether to apply PowerTransformer (Yeo-Johnson)
    balancing_technique: 'smote' | 'smotenc' | 'adasyn' | 'undersample' |
                         'smoteenn' | 'smotetomek' | 'class_weight' | None
    cat_indices     : list of int — categorical column indices for SMOTENC
    use_calibration : bool — wrap model with CalibratedClassifierCV after fitting
    """
    logger = logging.getLogger(__name__)

    # ── Convert to numpy arrays ────────────────────────────────────────
    if hasattr(X_train, 'values'):
        X_train_arr = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=np.float64)
    else:
        X_train_arr = np.array(X_train, dtype=np.float64)

    if hasattr(X_test, 'values'):
        X_test_arr = pd.DataFrame(X_test).apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=np.float64)
    else:
        X_test_arr = np.array(X_test, dtype=np.float64)

    y_train_arr = np.array(y_train)
    y_test_arr  = np.array(y_test)

    # ── Target label encoding ──────────────────────────────────────────
    target_le = None
    if not pd.api.types.is_numeric_dtype(pd.Series(y_train_arr)):
        target_le = LabelEncoder()
        y_train_arr = target_le.fit_transform(y_train_arr.astype(str))
        y_test_arr  = target_le.transform(y_test_arr.astype(str))
        logger.info(
            "[Training] Target label-encoded: classes=%s",
            list(target_le.classes_),
        )

    n_samples = len(X_train_arr)

    # ── Resolve class_weight for pipeline (no resampler) ──────────────
    class_weights = None
    if balancing_technique == "class_weight":
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train_arr)
        w = compute_class_weight("balanced", classes=classes, y=y_train_arr)
        class_weights = dict(zip(classes.tolist(), w.tolist()))
        logger.info("[Training] class_weight computed: %s", class_weights)

    # ── Build model instance ───────────────────────────────────────────
    catalogue = CLASSIFICATION_MODELS if task_type == "classification" else REGRESSION_MODELS
    if model_name not in catalogue:
        raise ValueError(f"Unknown model '{model_name}' for task '{task_type}'.")

    clf, actual_name = _build_model(
        model_name, catalogue, dict(hyperparams), class_weights, n_samples
    )

    # ── Build the Pipeline ────────────────────────────────────────────
    pipeline = _build_post_split_pipeline(
        clf=clf,
        scaler_type=scaler_type,
        apply_skewness=apply_skewness,
        balancing_technique=balancing_technique,
        cat_indices=cat_indices or [],
    )

    # ── Fit pipeline on X_train ONLY ──────────────────────────────────
    # For imblearn pipelines: resampler.fit_resample() runs only during fit()
    # For sklearn pipelines: scaler.fit() runs only during fit()
    # X_test is NEVER passed to fit() — only transform() is used via predict()
    pipeline.fit(X_train_arr, y_train_arr)
    logger.info(
        "[Training] Pipeline fitted on %d training samples. Steps: %s",
        n_samples,
        " → ".join(s[0] for s in pipeline.steps),
    )

    # ── Predict on X_test (transform only — no fit) ───────────────────
    y_pred = pipeline.predict(X_test_arr)

    # ── Probabilities (positive class) ────────────────────────────────
    y_prob: Any = None
    if task_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_test_arr)
            if proba.shape[1] == 2:
                # Binary classification — return probability of positive class
                y_prob = proba[:, 1].astype(np.float64)
            else:
                # Multi-class — return full probability matrix (shape: n_samples × n_classes)
                y_prob = proba.astype(np.float64)
            logger.info(
                "[Training] y_prob computed — shape=%s  n_classes=%d",
                y_prob.shape, proba.shape[1],
            )
        except Exception as prob_err:
            logger.warning("[Training] predict_proba failed: %s", prob_err)
            y_prob = None

    # ── Optional calibration ───────────────────────────────────────────
    calibrated = False
    if use_calibration and task_type == "classification":
        try:
            from sklearn.calibration import CalibratedClassifierCV
            # Calibrate on the raw clf (not pipeline) then rebuild pipeline
            calibrated_clf = CalibratedClassifierCV(
                pipeline.named_steps["model"], method="sigmoid", cv=3
            )
            calibrated_clf.fit(X_train_arr, y_train_arr)
            # Replace the model step in a fresh pipeline
            new_steps = [(n, s) for n, s in pipeline.steps if n != "model"]
            new_steps.append(("model", calibrated_clf))
            if hasattr(pipeline, 'steps') and any(s[0] == 'resampler' for s in pipeline.steps):
                from imblearn.pipeline import Pipeline as ImbPipeline
                pipeline = ImbPipeline(new_steps)
            else:
                pipeline = SklearnPipeline(new_steps)
            pipeline.fit(X_train_arr, y_train_arr)
            y_prob_cal = pipeline.predict_proba(X_test_arr)[:, 1].astype(np.float64)
            y_prob     = y_prob_cal
            calibrated = True
            logger.info(
                "[Training] Calibration applied. y_prob recomputed — min=%.4f max=%.4f",
                float(y_prob.min()), float(y_prob.max()),
            )
        except Exception as cal_err:
            logger.warning("[Training] Calibration failed (%s) — using uncalibrated pipeline.", cal_err)

    logger.info(
        "[Training] calibrated=%s  use_calibration_requested=%s",
        calibrated, use_calibration,
    )

    # ── Labels stay encoded (integers) ──────────────────────────────────
    # y_test/y_pred/y_train remain as encoded integers so sklearn metrics
    # (f1, precision, recall, roc_auc) work correctly.
    # target_le is returned so the route can persist it in session for
    # display-only decoding (frontend labels, confusion matrix axis labels).
    logger.info(
        "[Training] Returning encoded arrays. target_le=%s",
        list(target_le.classes_) if target_le is not None else None,
    )

    return pipeline, X_train_arr, X_test_arr, y_train_arr, y_test_arr, y_pred, y_prob, target_le


# ── Class Imbalance Analysis (Training data only) ────────────────────── #

def _recommend_technique(
    df: pd.DataFrame,
    feature_cols: list,
    train_size: int,
    minority_pct: float,
) -> Dict[str, Any]:
    """
    Pick the best balancing technique based on dataset size, feature types
    and imbalance severity.
    """
    size_cat     = "small" if train_size < 1_000 else "medium" if train_size < 10_000 else "large"
    # Safety: only use cols that actually exist in df (engineered cols may not be present)
    safe_cols    = [c for c in feature_cols if c in df.columns]
    num_cols     = len(df[safe_cols].select_dtypes(include=[np.number]).columns) if safe_cols else 0
    numeric_ratio = num_cols / len(safe_cols) if safe_cols else 1.0
    severity     = "severe" if minority_pct < 10 else "moderate" if minority_pct < 25 else "mild"

    if size_cat in ("small", "medium") and numeric_ratio >= 0.7:
        rec    = "smote"
        reason = (
            f"Your training set is {size_cat} ({train_size:,} samples) with "
            f"{int(numeric_ratio * 100)}% numeric features — ideal conditions for SMOTE. "
            f"SMOTE synthesises plausible minority-class samples via k-NN interpolation, "
            f"addressing the {severity} imbalance without discarding any existing data."
        )
    elif size_cat == "large":
        rec    = "undersample"
        reason = (
            f"With {train_size:,} training samples the dataset is large, so random "
            f"undersampling is efficient and the information lost from majority-class "
            f"removal is tolerable. Training speed is also significantly improved."
        )
    else:
        rec    = "class_weight"
        reason = (
            f"{int((1 - numeric_ratio) * 100)}% of features are non-numeric, making "
            f"SMOTE interpolation less meaningful. Class weighting penalises minority-class "
            f"errors more heavily without modifying the data at all, preserving all feature "
            f"relationships intact."
        )

    LABELS = {
        "smote"       : "SMOTE (Oversampling)",
        "undersample" : "Random Undersampling",
        "class_weight": "Class Weighting",
        "none"        : "No Balancing",
    }
    alternatives = [
        {"technique": t, "label": LABELS[t], "tradeoff": desc}
        for t, desc in [
            ("smote",
             "Generates synthetic minority samples — ideal for numeric feature sets, "
             "small-to-medium data size."),
            ("undersample",
             "Removes majority-class rows — fast for large datasets but loses training data."),
            ("class_weight",
             "Penalises minority errors in the loss function — no data change, "
             "works best when data integrity is critical."),
            ("none",
             "Skip resampling — use only when the dataset is acceptably balanced "
             "or the model handles imbalance natively."),
        ]
        if t != rec
    ]

    return {
        "recommended"            : rec,
        "recommended_label"      : LABELS[rec],
        "reason"                 : reason,
        "dataset_characteristics": {
            "size_category"        : size_cat,
            "train_samples"        : train_size,
            "numeric_feature_ratio": round(numeric_ratio, 2),
            "imbalance_severity"   : severity,
        },
        "alternatives": alternatives,
    }


def analyze_training_imbalance(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    test_size: float = 0.2,
    random_state: int = 42,
    eda_minority_pct: float = None,
) -> Dict[str, Any]:
    """
    Perform a dry-run split and analyse the CLASS DISTRIBUTION of y_train.
    Only y values are needed — avoids loading X into memory for the split.
    """
    y = df[target_col].to_numpy()

    # Stratified split preserves class ratios (no need to split X at all)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    try:
        train_idx, _ = next(sss.split(y, y))
        y_train = y[train_idx]
    except Exception:
        # Fallback for regression (non-stratifiable)
        n_train  = int(len(y) * (1 - test_size))
        y_train  = y[:n_train]

    unique, counts = np.unique(y_train, return_counts=True)
    total          = int(len(y_train))

    distribution = {
        str(cls): {"count": int(cnt), "pct": round(float(cnt) / total * 100, 2)}
        for cls, cnt in zip(unique, counts)
    }

    majority_idx  = int(np.argmax(counts))
    minority_idx  = int(np.argmin(counts))
    majority_class = str(unique[majority_idx])
    minority_class = str(unique[minority_idx])
    majority_count = int(counts[majority_idx])
    minority_count = int(counts[minority_idx])
    minority_pct   = round(float(minority_count) / total * 100, 2)
    imbalance_ratio = round(float(minority_count) / majority_count, 4)

    is_balanced = minority_pct >= 40.0

    consistency = None
    if eda_minority_pct is not None:
        diff = round(abs(minority_pct - eda_minority_pct), 2)
        consistency = {
            "eda_minority_pct"  : eda_minority_pct,
            "train_minority_pct": minority_pct,
            "difference"        : diff,
            "consistent"        : diff <= 5.0,
            "note": (
                "Training distribution closely mirrors the full dataset — the split is representative."
                if diff <= 5.0
                else f"Training minority % ({minority_pct}%) deviates by {diff}% from EDA "
                     f"({eda_minority_pct}%). Consider stratified splitting to preserve proportions."
            ),
        }

    recommendation = _recommend_technique(df, feature_cols, total, minority_pct) if not is_balanced else None

    return {
        "train_samples"  : total,
        "minority_class" : minority_class,
        "majority_class" : majority_class,
        "minority_count" : minority_count,
        "majority_count" : majority_count,
        "minority_pct"   : minority_pct,
        "imbalance_ratio": imbalance_ratio,
        "status"         : "Balanced" if is_balanced else "Imbalanced",
        "is_balanced"    : is_balanced,
        "distribution"   : distribution,
        "consistency_check": consistency,
        "recommendation" : recommendation,
    }


def apply_balancing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    technique: str,
    cat_indices: list = None,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Apply the selected balancing technique to TRAINING data ONLY.
    Test data is NEVER passed in or touched.

    Supported techniques
    --------------------
    none          → pass through (no change)
    class_weight  → returns class_weights dict (no data change)
    smote         → SMOTE oversampling (numeric only)
    smotenc       → SMOTENC for mixed data (requires cat_indices)
    adasyn        → ADASYN adaptive oversampling
    undersample   → Random undersampling to minority count
    smoteenn      → SMOTE + ENN hybrid
    smotetomek    → SMOTE + Tomek hybrid

    Returns
    -------
    (X_train_balanced, y_train_balanced, class_weights_or_None)
    """
    logger = logging.getLogger(__name__)

    # ── No change strategies ────────────────────────────────────────────
    if not technique or technique == "none":
        return X_train, y_train, None

    if technique == "class_weight":
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        w = compute_class_weight("balanced", classes=classes, y=y_train)
        return X_train, y_train, dict(zip(classes.tolist(), w.tolist()))

    if technique == "undersample":
        unique_classes, counts = np.unique(y_train, return_counts=True)
        minority_n = int(min(counts))
        rng = np.random.RandomState(42)
        X_parts, y_parts = [], []
        for cls in unique_classes:
            mask = y_train == cls
            Xi, yi = X_train[mask], y_train[mask]
            if len(Xi) > minority_n:
                idx = rng.choice(len(Xi), minority_n, replace=False)
                Xi, yi = Xi[idx], yi[idx]
            X_parts.append(Xi)
            y_parts.append(yi)
        return np.vstack(X_parts), np.concatenate(y_parts), None

    # ── imblearn strategies ─────────────────────────────────────────────
    try:
        import imblearn  # noqa
    except ImportError:
        raise ValueError(
            f"imbalanced-learn is required for '{technique}'. "
            "Install with: pip install imbalanced-learn"
        )

    if technique == "smote":
        from imblearn.over_sampling import SMOTE
        X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
        logger.info("[apply_balancing] SMOTE: %d → %d samples", len(y_train), len(y_res))
        return X_res, y_res, None

    if technique == "smotenc":
        if not cat_indices:
            raise ValueError(
                "SMOTENC requires cat_indices (list of categorical column indices). "
                "Ensure the imbalance config includes 'cat_indices' from preprocessing metadata."
            )
        from imblearn.over_sampling import SMOTENC
        X_res, y_res = SMOTENC(
            categorical_features=cat_indices, random_state=42
        ).fit_resample(X_train, y_train)
        logger.info(
            "[apply_balancing] SMOTENC: %d → %d samples, %d cat_indices",
            len(y_train), len(y_res), len(cat_indices),
        )
        return X_res, y_res, None

    if technique == "adasyn":
        from imblearn.over_sampling import ADASYN
        try:
            X_res, y_res = ADASYN(random_state=42).fit_resample(X_train, y_train)
            logger.info("[apply_balancing] ADASYN: %d → %d samples", len(y_train), len(y_res))
        except Exception as exc:
            # ADASYN can fail if a class is too small; fall back to SMOTE
            logger.warning("[apply_balancing] ADASYN failed (%s) — falling back to SMOTE", exc)
            from imblearn.over_sampling import SMOTE
            X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
        return X_res, y_res, None

    if technique == "smoteenn":
        from imblearn.combine import SMOTEENN
        X_res, y_res = SMOTEENN(random_state=42).fit_resample(X_train, y_train)
        logger.info("[apply_balancing] SMOTEENN: %d → %d samples", len(y_train), len(y_res))
        return X_res, y_res, None

    if technique == "smotetomek":
        from imblearn.combine import SMOTETomek
        X_res, y_res = SMOTETomek(random_state=42).fit_resample(X_train, y_train)
        logger.info("[apply_balancing] SMOTETomek: %d → %d samples", len(y_train), len(y_res))
        return X_res, y_res, None

    # Unknown technique — pass through with a warning
    logger.warning("[apply_balancing] Unknown technique '%s' — no balancing applied.", technique)
    return X_train, y_train, None

