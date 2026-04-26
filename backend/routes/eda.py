"""
eda.py route  —  /api/eda

Performance fix: all blocking pandas/numpy EDA calls wrapped in
run_in_threadpool so they never stall the FastAPI event loop.
"""
import logging
import traceback

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.services.session_store import store
from backend.utils.helpers import safe_json
from ml_pipeline.eda import (
    class_distribution,
    correlation_matrix,
    feature_vs_target,
    full_ml_readiness_analysis,
    histogram_data,
    summary_statistics,
    value_counts_all,
)

router = APIRouter(prefix="/api/eda", tags=["EDA"])
logger = logging.getLogger(__name__)


def _get_df():
    # Must use 'is not None' — never use 'or' on a DataFrame (ambiguous truth value)
    df = store.processed_df if store.processed_df is not None else store.raw_df
    if df is None:
        raise HTTPException(
            status_code=404,
            detail="No dataset loaded. Please upload a CSV file first.",
        )
    return df


@router.get("/summary", summary="Summary statistics")
async def get_summary():
    try:
        df     = _get_df()
        result = await run_in_threadpool(summary_statistics, df)
        return JSONResponse(content=safe_json(result))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[EDA /summary] error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to compute summary statistics: {str(exc)}")


@router.get("/histogram", summary="Histogram for a column")
async def get_histogram(column: str = Query(...), bins: int = Query(20)):
    try:
        df = _get_df()
        if column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{column}' not found.")
        result = await run_in_threadpool(histogram_data, df, column, bins)
        return JSONResponse(content=safe_json(result))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[EDA /histogram] column=%s error:\n%s", column, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to compute histogram for '{column}': {str(exc)}")


@router.get("/correlation", summary="Correlation matrix")
async def get_correlation():
    try:
        df     = _get_df()
        result = await run_in_threadpool(correlation_matrix, df)
        return JSONResponse(content=safe_json(result))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[EDA /correlation] error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to compute correlation matrix: {str(exc)}")


@router.get("/valuecounts", summary="Value counts for categorical columns")
async def get_value_counts():
    try:
        df     = _get_df()
        result = await run_in_threadpool(value_counts_all, df)
        return JSONResponse(content=safe_json(result))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[EDA /valuecounts] error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to compute value counts: {str(exc)}")


@router.get("/feature-vs-target", summary="Feature vs target relationship")
async def get_feature_vs_target(feature: str = Query(...)):
    try:
        df     = _get_df()
        target = store.target_column
        if target is None:
            raise HTTPException(status_code=400, detail="Target column not set.")
        if feature not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{feature}' not found.")
        result = await run_in_threadpool(feature_vs_target, df, feature, target)
        return JSONResponse(content=safe_json(result))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[EDA /feature-vs-target] feature=%s error:\n%s", feature, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to compute feature-vs-target for '{feature}': {str(exc)}")


@router.get("/class-distribution", summary="Target variable class distribution analysis")
async def get_class_distribution():
    """
    Analyse the class distribution of the target variable.
    Always uses raw_df to reflect the original distribution.
    """
    try:
        df = store.raw_df
        if df is None:
            raise HTTPException(status_code=404, detail="No dataset loaded. Please upload a CSV file first.")
        target = store.target_column
        if target is None:
            raise HTTPException(status_code=400, detail="Target column not set.")
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in the dataset.")
        result = await run_in_threadpool(class_distribution, df, target)
        return JSONResponse(content=safe_json(result))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[EDA /class-distribution] error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to compute class distribution: {str(exc)}")


# ── NEW: GET /api/eda/v2/analysis ─────────────────────────────────────

@router.get("/v2/analysis", summary="ML Readiness Analysis (v2)")
async def get_ml_readiness():
    """
    Full ML readiness and decision-support analysis.
    Returns: overview, data_quality, target_analysis, feature_relationships,
             correlation_analysis, feature_diagnostics, suggested_actions,
             data_quality_score.

    All existing /eda/* endpoints remain unchanged and active.
    """
    try:
        df     = _get_df()
        target = store.target_column   # may be None — handled gracefully
        result = await run_in_threadpool(full_ml_readiness_analysis, df, target)
        return JSONResponse(content=safe_json(result))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[EDA /v2/analysis] error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"ML readiness analysis failed: {str(exc)}")
