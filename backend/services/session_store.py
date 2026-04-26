"""
session_store.py
----------------
Thread-safe in-memory store for dataset state, trained models,
and pipeline artefacts shared across all API routes.

PRODUCTION PIPELINE ORDER:
  1. Upload → 2. EDA → 3. Cleaning → 4. Preprocessing (FULL DATA, no scaling/skewness)
  5. Split  → 6. Feature Engineering (post-split) → 7. Class Imbalance (train only)
  8. Training (sklearn Pipeline: skewness → scaler → model, fit on X_train only)
  9. Evaluation → 10. Bias → 11. Predictions → 12. Reports

PERSISTENCE: Model and test data are automatically saved to disk
when training completes, and can be reloaded on evaluation.
"""
import os
import threading
from typing import Any, Dict, Optional

import joblib
import pandas as pd

# Storage paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.joblib")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.joblib")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


class SessionStore:
    """Singleton-style store that holds the current ML session state."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "raw_df": None,           # Original uploaded DataFrame
            "processed_df": None,     # After preprocessing
            "target_column": None,    # Name of target / label column
            "feature_columns": None,  # Selected feature columns
            "task_type": None,        # "classification" | "regression"
            "model": None,            # Trained sklearn estimator
            "model_name": None,       # Human-readable model name
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "y_pred": None,
            "y_prob": None,           # float64 P(positive class) — from predict_proba
            "applied_threshold": 0.5, # last threshold applied to y_prob
            # ── Versioning & Reproducibility ─────────────────────────────────
            "model_id"         : None,  # e.g. "random_forest_20260417_162000"
            "dataset_hash"     : None,  # MD5 hex of raw dataset CSV (first 16 chars)
            "training_timestamp": None, # ISO-8601 string
            "calibrated"       : False, # whether CalibratedClassifierCV was applied
            # ────────────────────────────────────────────────────────────────
            "preprocessing_log": [],  # List of step descriptions
            "cleaning_log": [],       # List of cleaning actions taken
            "split_config": None,     # Dict with test_size, random_state, stratify
            "fe_action_log": [],      # Feature engineering action history
            "fe_checkpoint_train": None,  # Undo checkpoint X_train
            "fe_checkpoint_test":  None,  # Undo checkpoint X_test
            "label_encoders": {},     # Col → fitted LabelEncoder
            "scaler": None,           # Fitted scaler (if any) — DEPRECATED: use post_split_pipeline
            "post_split_pipeline": None,  # Fitted sklearn/imblearn Pipeline (skewness→scaler→model)
            "engineered_specs": [],   # Persisted feature specs → re-applied after preprocessing
            "balancing_technique": None,  # Selected technique from Class Imbalance step
            "balancing_config":    None,  # Full strategy config dict (strategy, enabled, use_class_weight, …)
            "target_label_encoder": None, # sklearn LabelEncoder fitted on target — None if target was numeric
        }

    # ------------------------------------------------------------------ #
    #  Generic get / set helpers                                           #
    # ------------------------------------------------------------------ #
    def get(self, key: str) -> Any:
        with self._lock:
            return self._state.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value

    def update(self, data: Dict[str, Any]) -> None:
        with self._lock:
            self._state.update(data)

    def reset(self) -> None:
        """Clear all session state (called on new upload)."""
        with self._lock:
            for k in self._state:
                if isinstance(self._state[k], list):
                    self._state[k] = []
                elif isinstance(self._state[k], dict):
                    self._state[k] = {}
                else:
                    self._state[k] = None
        # Also clear persisted files
        self._clear_persistence()

    def _clear_persistence(self) -> None:
        """Remove persisted model and test data files."""
        try:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            if os.path.exists(TEST_DATA_PATH):
                os.remove(TEST_DATA_PATH)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Persistence methods - Model & Test Data                             #
    # ------------------------------------------------------------------ #
    def save_model(self) -> bool:
        """Persist trained model to disk using joblib."""
        model = self.get("model")
        if model is None:
            return False
        try:
            joblib.dump(model, MODEL_PATH)
            print(f"[SessionStore] Model saved to {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[SessionStore] Failed to save model: {e}")
            return False

    def load_model(self) -> bool:
        """Load trained model from disk if it exists."""
        if not os.path.exists(MODEL_PATH):
            return False
        try:
            model = joblib.load(MODEL_PATH)
            self.set("model", model)
            print(f"[SessionStore] Model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[SessionStore] Failed to load model: {e}")
            return False

    def save_test_data(self) -> bool:
        """Persist test data and session metadata to disk."""
        X_test = self.get("X_test")
        y_test = self.get("y_test")
        if X_test is None or y_test is None:
            return False
        try:
            test_data = {
                # Arrays
                "X_test"             : X_test,
                "y_test"             : y_test,
                "y_pred"             : self.get("y_pred"),
                "y_prob"             : self.get("y_prob"),
                # Threshold
                "applied_threshold"  : self.get("applied_threshold") or 0.5,
                # Versioning metadata
                "model_id"           : self.get("model_id"),
                "dataset_hash"       : self.get("dataset_hash"),
                "training_timestamp" : self.get("training_timestamp"),
                "calibrated"         : self.get("calibrated") or False,
                # Session metadata needed after restart
                "task_type"          : self.get("task_type"),
                "model_name"         : self.get("model_name"),
                "feature_columns"    : self.get("feature_columns"),
            }
            joblib.dump(test_data, TEST_DATA_PATH)
            print(f"[SessionStore] Test data saved to {TEST_DATA_PATH}")
            return True
        except Exception as e:
            print(f"[SessionStore] Failed to save test data: {e}")
            return False

    def load_test_data(self) -> bool:
        """Load test data and session metadata from disk."""
        if not os.path.exists(TEST_DATA_PATH):
            return False
        try:
            test_data = joblib.load(TEST_DATA_PATH)
            self.set("X_test",              test_data.get("X_test"))
            self.set("y_test",              test_data.get("y_test"))
            self.set("y_pred",              test_data.get("y_pred"))
            self.set("y_prob",              test_data.get("y_prob"))
            self.set("applied_threshold",   test_data.get("applied_threshold", 0.5))
            self.set("model_id",            test_data.get("model_id"))
            self.set("dataset_hash",        test_data.get("dataset_hash"))
            self.set("training_timestamp",  test_data.get("training_timestamp"))
            self.set("calibrated",          test_data.get("calibrated", False))
            # Restore session metadata (lost on backend restart)
            if test_data.get("task_type"):
                self.set("task_type",       test_data["task_type"])
            if test_data.get("model_name"):
                self.set("model_name",      test_data["model_name"])
            if test_data.get("feature_columns"):
                self.set("feature_columns", test_data["feature_columns"])
            print(f"[SessionStore] Test data loaded from {TEST_DATA_PATH}")
            return True
        except Exception as e:
            print(f"[SessionStore] Failed to load test data: {e}")
            return False

    def has_persisted_model(self) -> bool:
        """Check if a model file exists on disk."""
        return os.path.exists(MODEL_PATH) and os.path.exists(TEST_DATA_PATH)

    # ------------------------------------------------------------------ #
    #  Convenience properties                                              #
    # ------------------------------------------------------------------ #
    @property
    def raw_df(self) -> Optional[pd.DataFrame]:
        return self.get("raw_df")

    @raw_df.setter
    def raw_df(self, value: Optional[pd.DataFrame]) -> None:
        self.set("raw_df", value)

    @property
    def processed_df(self) -> Optional[pd.DataFrame]:
        return self.get("processed_df")

    @processed_df.setter
    def processed_df(self, value: Optional[pd.DataFrame]) -> None:
        self.set("processed_df", value)

    @property
    def target_column(self) -> Optional[str]:
        return self.get("target_column")

    @target_column.setter
    def target_column(self, value: Optional[str]) -> None:
        self.set("target_column", value)

    @property
    def task_type(self) -> Optional[str]:
        return self.get("task_type")

    @task_type.setter
    def task_type(self, value: Optional[str]) -> None:
        self.set("task_type", value)

    @property
    def engineered_specs(self) -> list:
        return self.get("engineered_specs") or []

    @engineered_specs.setter
    def engineered_specs(self, value: list) -> None:
        self.set("engineered_specs", value or [])


# Global singleton instance used by all routes
store = SessionStore()
