"""
schemas.py
----------
Pydantic request / response schemas for all API routes.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
class TargetColumnRequest(BaseModel):
    target_column: str


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────
class PreprocessRequest(BaseModel):
    missing_strategy: str = "mean"          # mean | median | mode | drop
    encoding_strategy: str = "label"        # label | onehot
    handle_outliers: bool = False
    outlier_method: str = "iqr"             # iqr | zscore
    scaling: Optional[str] = None           # standard | minmax | None


# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
class FeatureSelectionRequest(BaseModel):
    selected_features: Optional[List[str]] = None   # None → use auto ranking
    top_k: int = 10


class ApplyFeaturesRequest(BaseModel):
    """Apply a list of feature specs (generated or manual) to the dataset."""
    feature_specs: List[Dict[str, Any]]   # each must have: name, feature_type, source_columns, enabled, operation (for manual)


class ManualFeatureRequest(BaseModel):
    """Create a single manual feature via a mathematical operation."""
    name      : str
    columns   : List[str]           # 1-2 source columns depending on operation
    operation : str                 # add | multiply | ratio | bin
    bin_count : int = 4             # used when operation='bin'
    bin_labels: Optional[List[str]] = None   # custom labels for bins


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
class TrainRequest(BaseModel):
    model_name: str                # e.g. "Random Forest"
    test_size: float = 0.2
    random_state: int = 42
    hyperparams: Dict[str, Any] = {}
    balancing_technique: Optional[str] = None  # 'smote' | 'undersample' | 'class_weight' | None
    use_calibration: bool = False              # wrap with CalibratedClassifierCV (classification only)
    # Post-split pipeline transformations (fit on X_train only)
    scaler: str = "standard"                  # standard | minmax | none
    apply_skewness: bool = True               # apply PowerTransformer (Yeo-Johnson) before scaling


class MultiTrainRequest(BaseModel):
    """Train one or more models sequentially with identical split + preprocessing."""
    model_names: List[str]                     # e.g. ["Random Forest", "Logistic Regression"]
    test_size: float = 0.2
    random_state: int = 42
    balancing_technique: Optional[str] = None  # resolved from session if omitted
    use_calibration: bool = False
    # Post-split pipeline transformations (fit on X_train only)
    scaler: str = "standard"                  # standard | minmax | none
    apply_skewness: bool = True               # apply PowerTransformer (Yeo-Johnson) before scaling


class ImbalanceAnalysisRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    eda_minority_pct: Optional[float] = None   # pass from EDA for consistency check


# ─────────────────────────────────────────────
# Bias Detection
# ─────────────────────────────────────────────
class BiasRequest(BaseModel):
    protected_attribute: str                    # column name
    privileged_value: Optional[Any] = None
    threshold: float = 0.5                      # classification threshold for y_pred recomputation
    debug: bool = False                         # if True: log group sizes, rates, TPR, per-group CM


# ─────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────
class PredictSingleRequest(BaseModel):
    input_data: Dict[str, Any]   # feature_name → value


# ─────────────────────────────────────────────
# Generic response
# ─────────────────────────────────────────────
class MessageResponse(BaseModel):
    message: str
    details: Optional[Any] = None
