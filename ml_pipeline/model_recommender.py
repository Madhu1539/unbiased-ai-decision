"""
model_recommender.py
====================
Rule-based model recommendation engine for the Model Selection page.
Pure function — no side-effects, no session state access.
"""
from typing import Any, Dict, List


# ── Per-model descriptions for UI cards ─────────────────────────────────────
MODEL_DESCRIPTIONS: Dict[str, Any] = {
    # ── Classification ────────────────────────────────────────────────
    "Logistic Regression": {
        "when_to_use": "Fast, interpretable baseline for binary or multi-class classification.",
        "strengths":   ["Highly interpretable", "Fast training & inference", "Low memory footprint", "Works well on linearly separable data"],
        "limitations": ["Cannot capture non-linear patterns", "Sensitive to outliers", "Requires feature scaling"],
        "badges":      ["Fast", "Interpretable", "Baseline"],
    },
    "Random Forest": {
        "when_to_use": "Mixed feature types, non-linear patterns, or moderate class imbalance.",
        "strengths":   ["Handles non-linearity", "Robust to outliers", "Built-in feature importance", "Handles imbalance via class_weight"],
        "limitations": ["Slower than linear models", "Memory-intensive for deep trees", "Less interpretable"],
        "badges":      ["Handles Imbalance", "Robust", "Advanced"],
    },
    "Decision Tree": {
        "when_to_use": "When you need a fully human-readable model or want to understand decision logic.",
        "strengths":   ["Fully interpretable tree rules", "No scaling needed", "Fast training"],
        "limitations": ["Prone to overfitting", "Unstable (high variance)", "Sensitive to data changes"],
        "badges":      ["Interpretable", "Fast"],
    },
    "SVM": {
        "when_to_use": "Small, high-dimensional datasets such as text or frequency features.",
        "strengths":   ["Effective in high dimensions", "Strong theoretical foundation", "Works well with clear margin"],
        "limitations": ["Very slow on large datasets (O(n^2))", "Sensitive to feature scaling", "Not recommended for >20k rows"],
        "badges":      ["High Dimensional"],
    },
    "KNN": {
        "when_to_use": "Simple non-parametric baseline on small datasets where proximity matters.",
        "strengths":   ["No training phase", "Intuitive and simple", "Non-parametric"],
        "limitations": ["Very slow inference on large data", "Memory-intensive", "Sensitive to irrelevant features"],
        "badges":      ["Simple"],
    },
    "Gradient Boosting": {
        "when_to_use": "When accuracy is the priority on medium to large structured datasets.",
        "strengths":   ["High accuracy", "Handles mixed feature types", "Built-in regularisation"],
        "limitations": ["Slower training than Random Forest", "Many hyperparameters", "Risk of overfitting without tuning"],
        "badges":      ["Handles Imbalance", "High Accuracy", "Advanced"],
    },
    "XGBoost": {
        "when_to_use": "Production-grade boosted trees for competitive accuracy on tabular data.",
        "strengths":   ["State-of-the-art accuracy", "Built-in L1/L2 regularisation", "Fast parallelism"],
        "limitations": ["Complex to tune", "Less interpretable", "Requires careful hyperparameter search"],
        "badges":      ["Handles Imbalance", "High Accuracy", "Advanced"],
    },
    # ── Regression ────────────────────────────────────────────────────
    "Linear Regression": {
        "when_to_use": "Simplest baseline for continuous target prediction with linear relationships.",
        "strengths":   ["Highly interpretable coefficients", "Fast training", "Scales to large data"],
        "limitations": ["Assumes linearity", "Sensitive to outliers", "Sensitive to multicollinearity"],
        "badges":      ["Fast", "Interpretable", "Baseline"],
    },
    "Ridge Regression": {
        "when_to_use": "When features are correlated or the dataset is high-dimensional.",
        "strengths":   ["L2 regularisation prevents overfitting", "Handles multicollinearity", "Stable"],
        "limitations": ["Still a linear model", "Does not perform feature selection"],
        "badges":      ["Fast", "Interpretable"],
    },
    "Lasso Regression": {
        "when_to_use": "When many features are irrelevant and you want automatic feature selection.",
        "strengths":   ["L1 regularisation performs feature selection", "Produces sparse solutions"],
        "limitations": ["Arbitrarily drops correlated features", "Can be numerically unstable"],
        "badges":      ["Feature Selection"],
    },
    "SVR": {
        "when_to_use": "Small datasets needing robust regression with margin tolerance.",
        "strengths":   ["Robust to outliers in prediction", "Flexible kernel"],
        "limitations": ["Very slow on large datasets (O(n^2))", "Sensitive to feature scaling"],
        "badges":      ["Robust"],
    },
    "KNN": {
        "when_to_use": "Simple non-parametric regressor for small datasets.",
        "strengths":   ["No training phase", "Non-parametric"],
        "limitations": ["Very slow on large data", "Memory-intensive"],
        "badges":      ["Simple"],
    },
    "Gradient Boosting": {
        "when_to_use": "High accuracy regression when features are mixed or outliers are present.",
        "strengths":   ["High accuracy", "Robust to outliers in features", "Handles non-linearity"],
        "limitations": ["Slower training", "Many hyperparameters"],
        "badges":      ["High Accuracy", "Robust", "Advanced"],
    },
}


def recommend_models(
    task_type: str,
    n_rows: int,
    n_features: int,
    n_numeric: int,
    n_categorical: int,
    imbalance_ratio: float = 1.0,
    has_high_outliers: bool = False,
    has_skewed_target: bool = False,
    has_high_cardinality: bool = False,
    model_info: Dict[str, Any] = None,
    all_model_names: List[str] = None,
) -> Dict[str, Any]:
    """
    Rule-based model recommendation engine.

    Parameters
    ----------
    task_type            : 'classification' | 'regression'
    n_rows               : total dataset rows
    n_features           : total feature count
    n_numeric            : numeric feature count
    n_categorical        : categorical feature count
    imbalance_ratio      : minority / majority class ratio (0–1). 1.0 = balanced.
    has_high_outliers    : True if any feature has >5% IQR outliers
    has_skewed_target    : True if regression target has |skew| > 2
    has_high_cardinality : True if any categorical column has unique_ratio > 0.1
    model_info           : dict from get_available_models() — {name: {pros, cons}}
    all_model_names      : ordered list of model names from catalogue

    Returns
    -------
    {
        "recommended_models": [{"name", "reason", "priority"}],
        "all_models":         {name: {pros, cons, when_to_use, strengths, limitations, badges}},
        "notes":              [str],
        "context":            {...dataset characteristics},
    }
    """
    if model_info is None:
        model_info = {}
    if all_model_names is None:
        all_model_names = list(model_info.keys())

    recommended: List[dict] = []
    notes: List[str]        = []

    is_large         = n_rows >= 1_000
    is_small         = n_rows < 500
    is_imbalanced    = imbalance_ratio < 0.6
    is_severe_imbal  = imbalance_ratio < 0.3
    is_high_dim      = n_features > 50
    mixed_types      = n_numeric > 0 and n_categorical > 0

    # ── CLASSIFICATION ──────────────────────────────────────────────────
    if task_type == "classification":
        recommended.append({
            "name":     "Logistic Regression",
            "reason":   "Reliable starting baseline — fast, interpretable, and low risk of overfitting.",
            "priority": 3,
        })

        if is_large or is_imbalanced or mixed_types or has_high_cardinality:
            parts = []
            if is_large:              parts.append("handles large datasets efficiently (n_jobs=-1)")
            if is_imbalanced:         parts.append("robust to class imbalance via class_weight")
            if mixed_types:           parts.append("works with mixed feature types natively")
            if has_high_cardinality:  parts.append("tree splits handle high-cardinality categoricals")
            recommended.append({
                "name":     "Random Forest",
                "reason":   "Recommended — " + ", ".join(parts) + ".",
                "priority": 1,
            })

        if is_imbalanced or is_large:
            gb_parts = []
            if is_severe_imbal: gb_parts.append("corrects minority-class errors sequentially via boosting")
            elif is_imbalanced: gb_parts.append("handles moderate imbalance via loss function weighting")
            if is_large:        gb_parts.append("scales well to large datasets")
            recommended.append({
                "name":     "Gradient Boosting",
                "reason":   ("Recommended for imbalanced data — " + ", ".join(gb_parts) + "."
                             if gb_parts else "High accuracy on structured tabular data."),
                "priority": 2,
            })

        if is_small:
            recommended.append({
                "name":     "KNN",
                "reason":   f"Small dataset ({n_rows:,} rows) — KNN avoids overfitting from complex models on limited data.",
                "priority": 2,
            })

        if is_high_dim:
            recommended.append({
                "name":     "SVM",
                "reason":   f"High-dimensional data ({n_features} features) — SVM has a strong theoretical margin in high dimensions.",
                "priority": 2,
            })
            notes.append("SVM training is O(n^2) — NOT recommended if rows > 20,000.")

        if is_severe_imbal:
            notes.append(
                f"Severe class imbalance (ratio = {imbalance_ratio:.2f}). "
                "Apply SMOTE or class_weight='balanced' in the Class Imbalance step before training."
            )

    # ── REGRESSION ──────────────────────────────────────────────────────
    else:
        recommended.append({
            "name":     "Linear Regression",
            "reason":   "Always start with a simple baseline to establish a performance floor.",
            "priority": 3,
        })

        if is_large or has_high_outliers or mixed_types:
            r_parts = []
            if is_large:          r_parts.append("scales to large datasets with n_jobs=-1")
            if has_high_outliers: r_parts.append("tree-based splits are inherently robust to feature outliers")
            if mixed_types:       r_parts.append("handles mixed feature types without manual encoding")
            recommended.append({
                "name":     "Random Forest",
                "reason":   "Recommended — " + ", ".join(r_parts) + ".",
                "priority": 1,
            })

        if is_high_dim or n_categorical > 10:
            recommended.append({
                "name":     "Ridge Regression",
                "reason":   f"{n_features} features — Ridge L2 regularisation prevents overfitting in high-dimensional spaces.",
                "priority": 2,
            })
            recommended.append({
                "name":     "Lasso Regression",
                "reason":   "High feature count — Lasso L1 regularisation automatically selects the most relevant features.",
                "priority": 2,
            })

        if is_small:
            recommended.append({
                "name":     "KNN",
                "reason":   f"Small dataset ({n_rows:,} rows) — simple non-parametric regressor with no assumptions.",
                "priority": 2,
            })

        if has_high_outliers:
            recommended.append({
                "name":     "Gradient Boosting",
                "reason":   "Outliers detected in features — gradient boosted trees are inherently resistant to extreme values.",
                "priority": 2,
            })

        if has_skewed_target:
            notes.append(
                "Skewed target variable detected. Consider applying log1p transformation "
                "to the target variable before training to improve linear model performance."
            )

    # ── Deduplicate: keep highest priority (lowest number) per model ──────
    seen: Dict[str, dict] = {}
    for r in recommended:
        name = r["name"]
        if name not in seen or r["priority"] < seen[name]["priority"]:
            seen[name] = r
    recommended = sorted(seen.values(), key=lambda x: x["priority"])

    # ── Build enriched model catalogue ────────────────────────────────────
    rec_names = {r["name"] for r in recommended}
    all_models: Dict[str, Any] = {}
    for name in all_model_names:
        base_info  = model_info.get(name, {"pros": [], "cons": []})
        desc_info  = MODEL_DESCRIPTIONS.get(name, {})
        badges     = list(desc_info.get("badges", []))
        if name in rec_names:
            badges = ["Recommended"] + badges
        all_models[name] = {
            "pros":        base_info.get("pros", []),
            "cons":        base_info.get("cons", []),
            "when_to_use": desc_info.get("when_to_use", ""),
            "strengths":   desc_info.get("strengths", []),
            "limitations": desc_info.get("limitations", []),
            "badges":      badges,
        }

    return {
        "recommended_models": recommended,
        "all_models":         all_models,
        "notes":              notes,
        "context": {
            "task_type":          task_type,
            "n_rows":             n_rows,
            "n_features":         n_features,
            "n_numeric":          n_numeric,
            "n_categorical":      n_categorical,
            "imbalance_ratio":    imbalance_ratio,
            "is_large_dataset":   is_large,
            "is_small_dataset":   is_small,
            "is_high_dim":        is_high_dim,
            "has_high_outliers":  has_high_outliers,
            "has_skewed_target":  has_skewed_target,
        },
    }
