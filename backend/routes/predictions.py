"""
predictions.py route  —  /api/predict

Data flow:
  - X (numeric, post-preprocessing) → pipeline.predict() → encoded int prediction
  - target_label_encoder.inverse_transform() → original label string (display only)
  - All numeric computation (metrics) uses encoded integers stored in y_test/y_pred
"""
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from backend.models.schemas import PredictSingleRequest
from backend.services.session_store import store
from backend.utils.helpers import safe_json

router = APIRouter(prefix="/api/predict", tags=["Predictions"])


def _get_model_step(pipeline):
    """Safely get the final estimator step from sklearn or imblearn pipeline."""
    # imblearn Pipeline also has .steps and named_steps
    if hasattr(pipeline, "named_steps"):
        step = pipeline.named_steps.get("model")
        if step is not None:
            return step
    # Fallback: last step (works for both pipeline types)
    if hasattr(pipeline, "steps"):
        return pipeline.steps[-1][1]
    return pipeline


def _decode_labels(arr, target_le):
    """Decode encoded integer labels back to original strings, or convert to str."""
    if target_le is not None:
        try:
            return [str(c) for c in target_le.inverse_transform(arr.astype(int))]
        except Exception:
            pass
    return [str(v) for v in arr]


@router.get("/batch", summary="Predicted vs Actual for test set")
async def batch_predictions():
    """
    Returns decoded (original label) actual vs predicted for display.
    y_test and y_pred stored as encoded integers; decoded here for the frontend.
    """
    model = store.get("model")
    y_test = store.get("y_test")
    y_pred = store.get("y_pred")
    target_le = store.get("target_label_encoder")

    if model is None:
        raise HTTPException(status_code=404, detail="No trained model.")
    if y_test is None or y_pred is None:
        raise HTTPException(status_code=404, detail="No test predictions found. Run training first.")

    n = min(200, len(y_test))
    y_test_arr = np.array(y_test[:n])
    y_pred_arr = np.array(y_pred[:n])

    actual    = _decode_labels(y_test_arr, target_le)
    predicted = _decode_labels(y_pred_arr, target_le)

    return JSONResponse(content=safe_json({
        "actual":    actual,
        "predicted": predicted,
        "count":     int(n),
        "model":     store.get("model_name"),
        "label_encoded": target_le is not None,
    }))


@router.post("/single", summary="Predict for a custom input row")
async def single_prediction(body: PredictSingleRequest):
    """
    Use the stored post_split_pipeline for inference.
    The pipeline applies skewness + scaler automatically (fitted on X_train only).
    Prediction is decoded back to original label if target was label-encoded.
    """
    pipeline     = store.get("post_split_pipeline") or store.get("model")
    feature_cols = store.get("feature_columns")
    target_le    = store.get("target_label_encoder")

    if pipeline is None or not feature_cols:
        raise HTTPException(status_code=400, detail="Model not available. Run training first.")

    # Build input array in correct feature order
    row = []
    for col in feature_cols:
        val = body.input_data.get(col)
        if val is None:
            raise HTTPException(status_code=422, detail=f"Missing value for feature '{col}'.")
        try:
            row.append(float(val))
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail=f"Non-numeric value for '{col}': {val}")

    X_input = np.array(row, dtype=np.float64).reshape(1, -1)

    # pipeline.predict() applies all transforms → model.predict()
    pred_encoded = pipeline.predict(X_input)[0]

    # Decode prediction to original label string
    if target_le is not None:
        try:
            pred_label = str(target_le.inverse_transform([int(pred_encoded)])[0])
        except Exception:
            pred_label = str(pred_encoded)
    else:
        pred_label = str(pred_encoded)

    # Probabilities — keyed by original class labels
    probability = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X_input)[0]
            model_step = _get_model_step(pipeline)
            encoded_classes = getattr(model_step, "classes_", list(range(len(proba))))
            # Decode class keys to original labels if possible
            if target_le is not None:
                try:
                    decoded_classes = [str(c) for c in target_le.inverse_transform(
                        np.array(encoded_classes, dtype=int)
                    )]
                except Exception:
                    decoded_classes = [str(c) for c in encoded_classes]
            else:
                decoded_classes = [str(c) for c in encoded_classes]
            probability = {cls: float(p) for cls, p in zip(decoded_classes, proba)}
        except Exception:
            probability = None

    return JSONResponse(content=safe_json({
        "prediction":      pred_label,
        "probability":     probability,
        "features_used":   feature_cols,
        "pipeline_applied": True,
        "label_encoded":   target_le is not None,
    }))
