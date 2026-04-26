"""
imbalance.py  —  /api/imbalance

Production-grade Class Imbalance detection and strategy configuration.
Points 19-25: SMOTE vs SMOTENC detection, categorical index tracking,
large-dataset safety (>500k), high-dim safety, priority sampler selection,
hard failure conditions, encoding-aware resampler selection.

Actual resampling runs at TRAINING TIME inside imblearn.pipeline.Pipeline:
  Pipeline([('preprocessor', ...), ('resampler', handler), ('model', clf)])
  → fit() applies resampling on X_train/y_train only
  → predict() skips resampling automatically
  → X_test is NEVER modified

Endpoints:
  GET  /analyze  → analyse y_train + encoding detection + priority selection
  POST /preview  → simulate resampling in-memory (session NEVER modified)
  POST /confirm  → validate + save strategy config to session
  GET  /status   → return current strategy config
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.services.session_store import store
from backend.utils.helpers import safe_json

router = APIRouter(prefix="/api/imbalance", tags=["Class Imbalance"])
logger = logging.getLogger(__name__)

META_JSON_PATH = Path("preprocessing_metadata.json")

# ─── Strategy Registry ────────────────────────────────────────────────────────

STRATEGY_META: Dict[str, Dict[str, Any]] = {
    "none": {
        "label":       "No Balancing",
        "description": "Use training data as-is. Suitable when data is balanced.",
        "imblearn":    False,
        "category":    "none",
        "min_samples": 0,
    },
    "class_weight": {
        "label":       "Class Weighting",
        "description": "Adjusts the model loss function to penalise minority misclassification. Sets class_weight='balanced'. No data is created or removed.",
        "imblearn":    False,
        "category":    "no_resampling",
        "min_samples": 0,
    },
    "smote": {
        "label":       "SMOTE",
        "description": "Synthetic Minority Oversampling Technique — generates synthetic minority samples via KNN interpolation. Requires numeric features. Auto-promotes to SMOTENC when Ordinal/Label encoding is detected.",
        "imblearn":    True,
        "category":    "oversampling",
        "min_samples": 6,
    },
    "smotenc": {
        "label":       "SMOTENC",
        "description": "SMOTE for mixed numeric + categorical data. Uses categorical feature indices. ONLY valid when Ordinal or Label encoding is used (NOT OneHot).",
        "imblearn":    True,
        "category":    "oversampling",
        "min_samples": 6,
        "requires_ordinal": True,
    },
    "adasyn": {
        "label":       "ADASYN",
        "description": "Adaptive Synthetic Sampling — generates more samples in harder-to-learn regions. Numeric features only.",
        "imblearn":    True,
        "category":    "oversampling",
        "min_samples": 6,
    },
    "undersample": {
        "label":       "Random Undersampling",
        "description": "Randomly removes majority class samples until classes are balanced. Reduces total training data.",
        "imblearn":    True,
        "category":    "undersampling",
        "min_samples": 2,
    },
    "smoteenn": {
        "label":       "SMOTE + ENN",
        "description": "Hybrid: SMOTE oversampling followed by Edited Nearest Neighbours cleaning to remove noisy samples.",
        "imblearn":    True,
        "category":    "hybrid",
        "min_samples": 6,
    },
    "smotetomek": {
        "label":       "SMOTE + Tomek",
        "description": "Hybrid: SMOTE oversampling followed by Tomek Links removal for cleaner class boundaries.",
        "imblearn":    True,
        "category":    "hybrid",
        "min_samples": 6,
    },
}

VALID_STRATEGIES = set(STRATEGY_META.keys())

# Encoding types that expand categoricals → SMOTE (NOT SMOTENC)
OHE_ENCODINGS  = {"ohe", "onehot", "one_hot", "frequency", "target"}
# Encoding types that preserve categorical columns → SMOTENC eligible
ORDINAL_ENCODINGS = {"label", "ordinal"}

LARGE_DATASET_THRESHOLD = 500_000


# ─── Request schemas ──────────────────────────────────────────────────────────

class ConfirmRequest(BaseModel):
    strategy:          str  = "none"
    enabled:           bool = False
    force_on_balanced: bool = False
    force_large:       bool = False   # override large-dataset safety block


class PreviewRequest(BaseModel):
    strategy: str = "smote"


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _require_split():
    X_train = store.get("X_train")
    y_train = store.get("y_train")
    if X_train is None or y_train is None:
        raise HTTPException(
            status_code=400,
            detail="No train/test split found. Complete 'Split Data' first.",
        )
    return X_train, y_train


def _get_task_type() -> str:
    task = store.get("task_type")
    if task:
        return task
    target = store.get("target_column")
    processed = store.get("processed_df")
    raw       = store.get("raw_df")
    df = processed if processed is not None else raw
    if df is not None and target and target in df.columns:
        from backend.utils.helpers import infer_task_type
        return infer_task_type(df[target])
    return "classification"


def _compute_distribution(y: pd.Series) -> Dict[str, Any]:
    y = y.astype(str)
    n = len(y)
    counts = y.value_counts().sort_values()
    dist: Dict[str, Any] = {}
    for cls, cnt in counts.items():
        dist[str(cls)] = {"count": int(cnt), "pct": round(100 * cnt / n, 2)}

    if len(counts) < 2:
        only_cls = str(counts.index[0])
        return {
            "distribution":   dist, "n": n,
            "minority_class": only_cls, "majority_class": only_cls,
            "minority_count": int(counts.iloc[0]), "majority_count": int(counts.iloc[0]),
            "minority_ratio": 1.0, "n_classes": 1,
        }

    minority_cls   = str(counts.index[0])
    majority_cls   = str(counts.index[-1])
    minority_count = int(counts.iloc[0])
    majority_count = int(counts.iloc[-1])
    minority_ratio = round(minority_count / max(majority_count, 1), 4)

    return {
        "distribution":   dist, "n": n,
        "minority_class": minority_cls, "majority_class": majority_cls,
        "minority_count": minority_count, "majority_count": majority_count,
        "minority_ratio": minority_ratio, "n_classes": len(counts),
    }


def _severity(ratio: float) -> str:
    if ratio >= 0.5: return "balanced"
    if ratio >= 0.3: return "slight"
    if ratio >= 0.1: return "moderate"
    return "extreme"


# ─── Point 19 & 20: Encoding Detection + Categorical Index Tracking ───────────

def _detect_encoding_strategy() -> Dict[str, Any]:
    """
    Detect preprocessing encoding strategy.
    Priority:
      1. preprocessing_metadata.json (most reliable — stores exact config)
      2. Pipeline object inspection from session
      3. Fallback: unknown → default to SMOTE (safe for all encodings)

    Returns encoding type, smote_variant (smote | smotenc), cat_indices for SMOTENC.
    """
    _FALLBACK = {
        "encoding": "unknown", "low_card_enc": "unknown", "high_card_enc": "unknown",
        "is_ohe": True, "smote_variant": "smote", "cat_indices": [], "n_cat_features": 0,
        "pipeline_found": False, "source": "fallback",
        "note": "Preprocessing pipeline not found. Defaulting to SMOTE (safe for all encodings).",
    }

    # ── Approach 1: Read preprocessing_metadata.json ─────────────────
    if META_JSON_PATH.exists():
        try:
            with open(META_JSON_PATH) as f:
                meta = json.load(f)

            config   = meta.get("config", {})
            low_enc  = str(config.get("low_card_encoding", "ohe")).lower()
            high_enc = str(config.get("high_card_encoding", "frequency")).lower()

            # OHE or frequency/target → expanded columns → USE SMOTE (not SMOTENC)
            is_ohe = (low_enc in OHE_ENCODINGS) or (high_enc in OHE_ENCODINGS)

            # Point 19: SMOTENC only if ALL categorical uses ordinal/label (not expanded)
            use_smotenc_eligible = (low_enc in ORDINAL_ENCODINGS) and not is_ohe

            cat_indices: List[int] = []
            note = ""

            if use_smotenc_eligible:
                # Point 20: compute cat_indices from feature_names_after + cat_cols
                feature_names = meta.get("feature_names_after", [])
                cat_cols      = set(meta.get("cat_cols", []))
                if feature_names and cat_cols:
                    cat_indices = [
                        i for i, nm in enumerate(feature_names)
                        if nm in cat_cols
                    ]
                    if cat_indices:
                        note = (f"Ordinal/Label encoding detected ({low_enc}/{high_enc}) — "
                                f"SMOTENC will use {len(cat_indices)} categorical indices: {cat_indices[:5]}{'…' if len(cat_indices)>5 else ''}.")
                    else:
                        use_smotenc_eligible = False
                        note = (f"Non-OHE encoding ({low_enc}) but no categorical indices matched. "
                                "Falling back to standard SMOTE.")
                else:
                    use_smotenc_eligible = False
                    note = "Non-OHE encoding detected but feature_names_after not in metadata. Using SMOTE."
            else:
                note = (f"OHE/Frequency/Target encoding ({low_enc}/{high_enc}) — "
                        "using standard SMOTE (SMOTENC not applicable after OHE).")

            return {
                "encoding":       low_enc,
                "low_card_enc":   low_enc,
                "high_card_enc":  high_enc,
                "is_ohe":         is_ohe,
                "smote_variant":  "smotenc" if use_smotenc_eligible else "smote",
                "cat_indices":    cat_indices,
                "n_cat_features": len(cat_indices),
                "pipeline_found": True,
                "source":         "metadata_json",
                "note":           note,
            }
        except Exception as exc:
            logger.warning("[Imbalance] metadata.json parse failed: %s", exc)

    # ── Approach 2: Inspect pipeline object from session ─────────────
    pipeline = store.get("preprocessing_pipeline")
    if pipeline is not None and hasattr(pipeline, "named_steps"):
        try:
            preprocessor = pipeline.named_steps.get("preprocessor")
            detected_enc = "unknown"
            if preprocessor and hasattr(preprocessor, "transformers_"):
                for tname, transformer, _ in preprocessor.transformers_:
                    if "cat" in tname.lower():
                        steps_iter = transformer.steps if hasattr(transformer, "steps") else []
                        for _, step in steps_iter:
                            cls_name = type(step).__name__
                            if "OneHot" in cls_name:
                                detected_enc = "ohe"; break
                            elif "SafeLabel" in cls_name or "LabelEncoder" in cls_name:
                                detected_enc = "label"
                            elif "Ordinal" in cls_name:
                                detected_enc = "ordinal"
                            elif "Frequency" in cls_name:
                                detected_enc = "frequency"
                            elif "Target" in cls_name:
                                detected_enc = "target"

            is_ohe = detected_enc in OHE_ENCODINGS
            cat_indices = []
            if not is_ohe:
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    cat_indices = [i for i, nm in enumerate(feature_names) if "cat" in nm.lower()]
                except Exception:
                    cat_indices = []

            return {
                "encoding":       detected_enc,
                "low_card_enc":   detected_enc,
                "high_card_enc":  "unknown",
                "is_ohe":         is_ohe,
                "smote_variant":  "smote" if (is_ohe or not cat_indices) else "smotenc",
                "cat_indices":    cat_indices,
                "n_cat_features": len(cat_indices),
                "pipeline_found": True,
                "source":         "pipeline_inspection",
                "note":           f"Pipeline inspection: enc={detected_enc}, {len(cat_indices)} cat indices.",
            }
        except Exception as exc:
            logger.warning("[Imbalance] Pipeline inspection failed: %s", exc)

    return _FALLBACK


# ─── Point 23: Priority Sampler Selection Logic ───────────────────────────────

def _select_sampler_priority(
    requested: str,
    minority_ratio: float,
    n_train: int,
    n_features: int,
    minority_count: int,
    encoding_info: Dict[str, Any],
    force_large: bool = False,
    force_on_balanced: bool = False,
) -> Dict[str, Any]:
    """
    Priority order (Point 23):
      1. Balanced check            → No resampling (skipped when force_on_balanced=True)
      2. Large dataset (>500k)     → class_weight (unless force_large)
      3. SMOTE minority sample check → block if minority < 6
      4. Encoding-based selection  → SMOTE vs SMOTENC
      5. OHE + SMOTENC check       → hard ERROR (Point 24)
      6. High-dim (n_features >> n_samples) → class_weight
      7. Otherwise                 → use requested / upgraded strategy
    """
    warnings: List[str] = []
    overridden = False
    reason: Optional[str] = None
    cat_indices: List[int] = encoding_info.get("cat_indices", [])
    use_smotenc = False

    def _ret(strategy, override_reason=None):
        return {
            "strategy":       strategy,
            "overridden":     overridden,
            "override_reason": override_reason or reason,
            "warnings":       warnings,
            "cat_indices":    cat_indices if use_smotenc else [],
            "use_smotenc":    use_smotenc,
        }

    # ── Priority 1: Balanced ─────────────────────────────────────────
    # Skipped when the user explicitly force-enables resampling on balanced data.
    if minority_ratio >= 0.5 and requested not in ("none", "class_weight") and not force_on_balanced:
        overridden = True
        reason = "Dataset is balanced (ratio ≥ 0.5). Resampling not needed and may degrade performance."
        warnings.append(f"⚠ {reason}")
        return _ret("none", reason)

    if minority_ratio >= 0.5 and force_on_balanced and requested not in ("none", "class_weight"):
        warnings.append("ℹ Dataset is balanced but force-resampling is enabled by user. Applying requested strategy.")

    # ── Priority 2: Large dataset safety (Point 21) ──────────────────
    is_large = n_train > LARGE_DATASET_THRESHOLD
    smote_strategies = {"smote", "adasyn", "smoteenn", "smotetomek", "smotenc"}
    if is_large and requested in smote_strategies and not force_large:
        overridden = True
        reason = (f"Large dataset ({n_train:,} samples > {LARGE_DATASET_THRESHOLD:,}). "
                  "Resampling methods are disabled for memory safety. Using class_weight.")
        warnings.append(f"⚠ {reason}")
        warnings.append("ℹ Enable 'Force resampling on large dataset' to override (not recommended).")
        return _ret("class_weight", reason)

    if is_large and requested in smote_strategies and force_large:
        warnings.append(f"⚠ FORCED: Applying resampling on {n_train:,} rows (>500k). This may be very slow and memory-intensive.")

    # ── Priority 3: SMOTE minority sample check (Point 24) ───────────
    meta = STRATEGY_META.get(requested, {})
    if meta.get("min_samples", 0) > minority_count and requested in smote_strategies:
        overridden = True
        reason = (f"Insufficient minority samples ({minority_count} < "
                  f"{meta['min_samples']} needed for {STRATEGY_META[requested]['label']}). "
                  "Falling back to class_weight.")
        warnings.append(f"🚫 {reason}")
        return _ret("class_weight", reason)

    # ── Priority 4: Hard failure — SMOTENC on OHE data (Point 24) ─────
    if requested == "smotenc" and encoding_info.get("is_ohe", False):
        overridden = True
        reason = ("SMOTENC cannot be used with OneHot-encoded data. "
                  "OHE expands categoricals to binary dummies; SMOTENC expects original categorical indices. "
                  "Using class_weight instead.")
        warnings.append(f"🚫 {reason}")
        return _ret("class_weight", reason)

    # ── Priority 5: Encoding-based SMOTE → SMOTENC upgrade (Point 19) ─
    enc_info = encoding_info
    is_ohe        = enc_info.get("is_ohe", True)
    smote_variant = enc_info.get("smote_variant", "smote")

    if requested in ("smote", "adasyn") and smote_variant == "smotenc" and not is_ohe:
        enc_cat_indices = enc_info.get("cat_indices", [])
        if enc_cat_indices:
            # Auto-promote SMOTE → SMOTENC
            use_smotenc = True
            cat_indices = enc_cat_indices
            warnings.append(
                f"ℹ Ordinal/Label encoding detected → auto-upgraded to SMOTENC "
                f"with {len(cat_indices)} categorical feature indices."
            )
            requested = "smotenc"
        else:
            warnings.append("⚠ Non-OHE encoding detected but categorical indices not found — using standard SMOTE.")

    if requested == "smotenc" and not is_ohe:
        enc_cat_indices = enc_info.get("cat_indices", [])
        if enc_cat_indices:
            use_smotenc = True
            cat_indices = enc_cat_indices
        else:
            # Point 20 FAILSAFE: can't determine cat indices → fallback
            overridden = True
            reason = "Categorical feature tracking failed — cannot reliably determine indices for SMOTENC. Using class_weight instead."
            warnings.append(f"⚠ {reason}")
            return _ret("class_weight", reason)

    # ── OHE with any SMOTE variant → use standard SMOTE (Point 19) ────
    if is_ohe and requested == "smotenc":
        warnings.append("ℹ OHE detected — SMOTENC replaced with SMOTE automatically.")
        requested = "smote"
        use_smotenc = False
        cat_indices = []

    # ── Priority 6: High-dimensional safety (Point 22) ────────────────
    if n_features > n_train and requested in ("smote", "adasyn") and not use_smotenc:
        overridden = True
        reason = (f"n_features ({n_features}) >> n_train ({n_train}). "
                  "SMOTE creates noisy samples in sparse high-dimensional space. Using class_weight.")
        warnings.append(f"⚠ {reason}")
        return _ret("class_weight", reason)

    if n_features > 100 and requested in ("smote", "adasyn"):
        warnings.append(f"⚠ High-dimensional data ({n_features} features) — SMOTE may introduce noise. Consider class_weight.")

    return _ret(requested)


# ─── Resampler factory ────────────────────────────────────────────────────────

def _check_imblearn() -> bool:
    try:
        import imblearn  # noqa
        return True
    except ImportError:
        return False


def _get_resampler(strategy: str, random_state: int = 42, cat_indices: Optional[List[int]] = None):
    """
    Return the imblearn resampler for the given strategy.
    For SMOTENC, cat_indices MUST be provided and non-empty.
    """
    if strategy in ("none", "class_weight"):
        return None
    if not _check_imblearn():
        raise HTTPException(
            status_code=500,
            detail="imbalanced-learn not installed. Run: pip install imbalanced-learn",
        )

    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek

    if strategy == "smotenc":
        # Point 24: Hard block if no cat_indices
        if not cat_indices:
            raise HTTPException(
                status_code=422,
                detail="SMOTENC requires valid categorical feature indices. None found.",
            )
        from imblearn.over_sampling import SMOTENC
        return SMOTENC(categorical_features=cat_indices, random_state=random_state)

    mapping = {
        "smote":       SMOTE(random_state=random_state),
        "adasyn":      ADASYN(random_state=random_state),
        "undersample": RandomUnderSampler(random_state=random_state),
        "smoteenn":    SMOTEENN(random_state=random_state),
        "smotetomek":  SMOTETomek(random_state=random_state),
    }
    resampler = mapping.get(strategy)
    if resampler is None:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")
    return resampler


def _build_recommendation(
    sev: str, minority_count: int, n_features: int,
    minority_ratio: float, encoding_info: Dict[str, Any],
) -> Dict[str, Any]:
    smote_blocked = minority_count < 6
    high_dim      = n_features > 100
    is_ohe        = encoding_info.get("is_ohe", True)
    smote_variant = encoding_info.get("smote_variant", "smote")

    if sev == "balanced":
        return {
            "strategy": "none", "label": "No Balancing",
            "reason": "Dataset sufficiently balanced (ratio ≥ 0.5). Resampling is NOT recommended — it may degrade performance.",
            "enable_balancing": False, "imblearn_needed": False,
        }

    if sev == "slight":
        return {
            "strategy": "class_weight", "label": "Class Weighting",
            "reason": "Slight imbalance (ratio 0.3–0.5). Class weighting adjusts loss without modifying data — preferred over resampling.",
            "enable_balancing": True, "imblearn_needed": False,
        }

    # moderate or extreme
    if smote_blocked:
        return {
            "strategy": "class_weight", "label": "Class Weighting",
            "reason": f"Only {minority_count} minority samples — insufficient for SMOTE (needs ≥ 6). Using class_weight.",
            "enable_balancing": True, "imblearn_needed": False,
        }

    if high_dim:
        return {
            "strategy": "class_weight", "label": "Class Weighting",
            "reason": f"High-dimensional ({n_features} features) — SMOTE may create noisy synthetic samples. Class weighting is safer.",
            "enable_balancing": True, "imblearn_needed": False,
        }

    if sev == "extreme":
        return {
            "strategy": "smotetomek", "label": "SMOTE + Tomek",
            "reason": "Extreme imbalance — hybrid SMOTE+Tomek oversamples minority and cleans boundary noise.",
            "enable_balancing": True, "imblearn_needed": True,
        }

    # moderate — choose based on encoding
    if smote_variant == "smotenc" and not is_ohe and encoding_info.get("cat_indices"):
        return {
            "strategy": "smotenc", "label": "SMOTENC",
            "reason":   (f"Ordinal/Label encoding detected with {encoding_info['n_cat_features']} categorical features. "
                         "SMOTENC generates synthetic samples respecting categorical structure."),
            "enable_balancing": True, "imblearn_needed": True,
        }

    return {
        "strategy": "smote", "label": "SMOTE",
        "reason": "Moderate imbalance — SMOTE synthesises minority class samples via KNN interpolation.",
        "enable_balancing": True, "imblearn_needed": True,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/analyze", summary="Analyse y_train distribution + encoding detection + priority selection")
async def analyze():
    """
    Analyses y_train ONLY (never y_test).
    Includes:
      - Class distribution + severity (balanced/slight/moderate/extreme)
      - Encoding detection (OHE vs Ordinal → SMOTE vs SMOTENC)
      - Priority sampler selection with override explanations
      - Large dataset + high-dim safety checks
    """
    try:
        X_train = store.get("X_train")
        y_train = store.get("y_train")
        task_type = _get_task_type()

        # Graceful gate — split hasn't been done yet
        if X_train is None or y_train is None:
            return JSONResponse(content={
                "split_required": True,
                "is_applicable":  False,
                "task_type":      task_type,
                "message":        "Split Data step must be completed before Class Imbalance analysis.",
            })

        if task_type != "classification":
            return JSONResponse(content={
                "task_type": task_type, "status": "N/A",
                "message": "Class imbalance handling only applies to classification tasks.",
                "is_applicable": False,
            })

        n_features   = int(X_train.shape[1])
        n_train_rows = int(X_train.shape[0])
        imblearn_ok  = _check_imblearn()

        # Encoding detection (Point 19-20) — run in main thread (no I/O blocking)
        encoding_info = _detect_encoding_strategy()

        def _run():
            y = y_train.reset_index(drop=True) if hasattr(y_train, "reset_index") else pd.Series(y_train)
            stats = _compute_distribution(y)

            sev         = _severity(stats["minority_ratio"])
            rec         = _build_recommendation(sev, stats["minority_count"], n_features,
                                                stats["minority_ratio"], encoding_info)
            smote_block = stats["minority_count"] < 6
            is_large    = n_train_rows > LARGE_DATASET_THRESHOLD

            # Priority selection for auto mode
            priority_sel = _select_sampler_priority(
                requested      = rec["strategy"],
                minority_ratio = stats["minority_ratio"],
                n_train        = n_train_rows,
                n_features     = n_features,
                minority_count = stats["minority_count"],
                encoding_info  = encoding_info,
            )

            severity_meta = {
                "balanced": {"label": "Balanced",           "color": "emerald", "range": "≥ 0.5"},
                "slight":   {"label": "Slight Imbalance",   "color": "amber",   "range": "0.3–0.5"},
                "moderate": {"label": "Moderate Imbalance", "color": "orange",  "range": "0.1–0.3"},
                "extreme":  {"label": "Extreme Imbalance",  "color": "red",     "range": "< 0.1"},
            }

            # Available strategies with disabled flags
            available = []
            for sid, smeta in STRATEGY_META.items():
                is_smotenc    = sid == "smotenc"
                needs_ordinal = smeta.get("requires_ordinal", False)
                has_cat_idx   = bool(encoding_info.get("cat_indices"))

                # Block reasons
                no_imblearn    = smeta["imblearn"] and not imblearn_ok
                too_few        = smeta["min_samples"] > stats["minority_count"]
                smotenc_on_ohe = is_smotenc and encoding_info.get("is_ohe", True)
                smotenc_no_idx = is_smotenc and not has_cat_idx
                large_resampl  = is_large and smeta["imblearn"] and sid not in ("undersample",)
                hidden         = is_smotenc and encoding_info.get("is_ohe", True)

                disabled = no_imblearn or too_few or smotenc_on_ohe or smotenc_no_idx
                disable_reason = (
                    "imbalanced-learn not installed" if no_imblearn
                    else f"Requires ≥ {smeta['min_samples']} minority samples" if too_few
                    else "Cannot use SMOTENC with OneHot encoding — use SMOTE instead" if smotenc_on_ohe
                    else "Categorical indices not found — run preprocessing first" if smotenc_no_idx
                    else None
                )

                available.append({
                    "id":             sid,
                    "label":          smeta["label"],
                    "description":    smeta["description"],
                    "category":       smeta["category"],
                    "imblearn":       smeta["imblearn"],
                    "disabled":       disabled,
                    "disable_reason": disable_reason,
                    "large_warning":  large_resampl,
                    "hidden":         hidden,   # UI can hide truly-inapplicable options
                })

            return {
                "task_type":       "classification",
                "is_applicable":   True,
                "n_train":         stats["n"],
                "n_features":      n_features,
                "minority_class":  stats["minority_class"],
                "majority_class":  stats["majority_class"],
                "minority_count":  stats["minority_count"],
                "majority_count":  stats["majority_count"],
                "minority_ratio":  stats["minority_ratio"],
                "n_classes":       stats["n_classes"],
                "distribution":    stats["distribution"],
                "severity":        sev,
                "severity_label":  severity_meta[sev]["label"],
                "severity_color":  severity_meta[sev]["color"],
                "severity_range":  severity_meta[sev]["range"],
                "recommendation":  rec,
                "priority_selection": priority_sel,
                "smote_blocked":   smote_block,
                "smote_block_reason": (f"Only {stats['minority_count']} minority samples — SMOTE needs ≥ 6" if smote_block else None),
                "imblearn_available": imblearn_ok,
                "large_dataset":   is_large,
                "large_dataset_warning": (
                    f"Large dataset ({n_train_rows:,} rows > {LARGE_DATASET_THRESHOLD:,}). "
                    "SMOTE/ADASYN are disabled by default." if is_large else None
                ),
                "high_dim_warning": n_features > 100,
                "high_dim_vs_samples": n_features > n_train_rows,
                "encoding_detection": {
                    "encoding":       encoding_info["encoding"],
                    "low_card_enc":   encoding_info.get("low_card_enc", "unknown"),
                    "high_card_enc":  encoding_info.get("high_card_enc", "unknown"),
                    "is_ohe":         encoding_info["is_ohe"],
                    "smote_variant":  encoding_info["smote_variant"],
                    "cat_indices":    encoding_info["cat_indices"],
                    "n_cat_features": encoding_info["n_cat_features"],
                    "source":         encoding_info.get("source", "unknown"),
                    "note":           encoding_info["note"],
                },
                "available_strategies": available,
                "current_strategy":     store.get("balancing_technique") or "none",
            }

        result = await run_in_threadpool(_run)
        return JSONResponse(content=safe_json(result))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Imbalance /analyze] %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@router.post("/preview", summary="Simulate resampling in-memory — session NEVER modified")
async def preview(body: PreviewRequest):
    """
    Applies priority selection logic, then temporarily resamples in-memory.
    Returns before/after distribution.
    X_train and y_train in session are NEVER modified.
    """
    try:
        X_train, y_train = _require_split()
        strategy = body.strategy.lower().strip()
        if strategy not in VALID_STRATEGIES:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")

        if strategy in ("none", "class_weight"):
            y = y_train.reset_index(drop=True) if hasattr(y_train, "reset_index") else pd.Series(y_train)
            stats = _compute_distribution(y)
            return JSONResponse(content=safe_json({
                "strategy": strategy, "resampled": False,
                "before": stats, "after": stats,
                "note": "No resampling applied — class_weight adjusts model loss, not data.",
            }))

        if not _check_imblearn():
            raise HTTPException(status_code=500, detail="imbalanced-learn not installed: pip install imbalanced-learn")

        encoding_info = _detect_encoding_strategy()

        def _run():
            y = y_train.reset_index(drop=True) if hasattr(y_train, "reset_index") else pd.Series(y_train)
            X = X_train.reset_index(drop=True) if hasattr(X_train, "reset_index") else X_train

            before = _compute_distribution(y)

            # Run priority selection (may override user's requested strategy)
            priority = _select_sampler_priority(
                requested=strategy, minority_ratio=before["minority_ratio"],
                n_train=before["n"], n_features=int(X.shape[1]),
                minority_count=before["minority_count"], encoding_info=encoding_info,
            )
            final_strategy = priority["strategy"]
            cat_indices    = priority["cat_indices"] if priority["use_smotenc"] else None

            if final_strategy in ("none", "class_weight"):
                return {
                    "strategy": final_strategy, "resampled": False,
                    "before": before, "after": before,
                    "priority_warnings": priority["warnings"],
                    "note": priority.get("override_reason") or f"{STRATEGY_META[final_strategy]['label']} — no data modification.",
                }

            # Use only numeric columns (post-preprocessing should be all numeric)
            X_num = X.select_dtypes(include=[np.number])
            if X_num.shape[1] == 0:
                return {
                    "error": "No numeric columns found. Apply preprocessing before previewing resampling.",
                    "before": before, "after": before, "resampled": False,
                }

            X_arr = X_num.fillna(0.0).values
            resampler = _get_resampler(final_strategy, random_state=42, cat_indices=cat_indices)
            X_res, y_res = resampler.fit_resample(X_arr, y.values)

            after = _compute_distribution(pd.Series(y_res.astype(str)))
            return {
                "strategy":        final_strategy,
                "original_request": strategy,
                "auto_upgraded":   priority.get("overridden", False) or (final_strategy != strategy),
                "resampled":       True,
                "before":          before,
                "after":           after,
                "delta_samples":   after["n"] - before["n"],
                "use_smotenc":     priority["use_smotenc"],
                "cat_indices":     priority["cat_indices"],
                "priority_warnings": priority["warnings"],
            }

        result = await run_in_threadpool(_run)
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        return JSONResponse(content=safe_json(result))

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[Imbalance /preview] %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Preview failed: {exc}")


@router.post("/confirm", summary="Validate + save balancing strategy to session")
async def confirm(body: ConfirmRequest):
    """
    Runs the full priority validation and saves the confirmed strategy.
    May override user's choice with a safer alternative (with explanation).
    Training route reads balancing_config and builds the correct pipeline.
    """
    strategy = body.strategy.lower().strip()
    if strategy not in VALID_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Invalid strategy '{strategy}'.")

    X_train = store.get("X_train")
    y_train = store.get("y_train")

    # Gather stats for priority validation
    n_train    = int(X_train.shape[0]) if X_train is not None else 0
    n_features = int(X_train.shape[1]) if X_train is not None else 0
    minority_ratio = 1.0
    minority_count = n_train

    if y_train is not None:
        y = pd.Series(y_train).astype(str)
        counts = y.value_counts().sort_values()
        if len(counts) >= 2:
            minority_count = int(counts.iloc[0])
            majority_count = int(counts.iloc[-1])
            minority_ratio = round(minority_count / max(majority_count, 1), 4)

    encoding_info = _detect_encoding_strategy()

    # Run priority logic — pass force_on_balanced so Priority 1 (balanced check)
    # is skipped when the user explicitly chooses to rebalance balanced data.
    priority = _select_sampler_priority(
        requested=strategy if body.enabled else "none",
        minority_ratio=minority_ratio,
        n_train=n_train,
        n_features=n_features,
        minority_count=minority_count,
        encoding_info=encoding_info,
        force_large=body.force_large,
        force_on_balanced=body.force_on_balanced,
    )

    final_strategy = priority["strategy"]
    meta = STRATEGY_META[final_strategy]

    config = {
        "strategy":             final_strategy,
        "original_request":     strategy,
        "enabled":              body.enabled and final_strategy != "none",
        "use_class_weight":     final_strategy == "class_weight",
        "use_resampler":        meta["imblearn"] and body.enabled,
        "imblearn_needed":      meta["imblearn"],
        "label":                meta["label"],
        "use_smotenc":          priority["use_smotenc"],
        "cat_indices":          priority["cat_indices"],
        "overridden":           priority["overridden"],
        "override_reason":      priority.get("override_reason"),
        "priority_warnings":    priority["warnings"],
        "encoding":             encoding_info["encoding"],
        "is_ohe":               encoding_info["is_ohe"],
        "force_large":          body.force_large,
    }

    store.set("balancing_technique", final_strategy if final_strategy != "none" else None)
    store.set("balancing_config",    config)

    return JSONResponse(content=safe_json({
        "message":          f"Strategy confirmed: {meta['label']}.",
        "strategy":         final_strategy,
        "original_request": strategy,
        "overridden":       priority["overridden"],
        "override_reason":  priority.get("override_reason"),
        "config":           config,
        "warnings":         priority["warnings"],
    }))


@router.get("/status", summary="Return current balancing strategy from session")
async def status():
    technique = store.get("balancing_technique")
    config    = store.get("balancing_config")
    enc       = _detect_encoding_strategy()
    return JSONResponse(content=safe_json({
        "technique":         technique or "none",
        "confirmed":         technique is not None,
        "config":            config,
        "label":             STRATEGY_META.get(technique or "none", {}).get("label", "No Balancing"),
        "imblearn_available": _check_imblearn(),
        "encoding_detection": {
            "encoding":      enc["encoding"],
            "smote_variant": enc["smote_variant"],
            "n_cat_features": enc["n_cat_features"],
            "is_ohe":         enc["is_ohe"],
        },
    }))
