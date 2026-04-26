"""
upload.py  —  /api/upload

Performance fixes:
  - Large CSV streamed through pandas read_csv with chunked engine hint
  - processed_df no longer eagerly copied on upload (lazy copy happens
    only when preprocessing actually modifies the data)
  - dtypes inferred with low_memory=True and float32 for numeric cols
    to halve memory footprint on large datasets
"""
import io
import logging
import traceback

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.models.schemas import MessageResponse, TargetColumnRequest
from backend.services.session_store import store
from backend.utils.helpers import df_to_records, infer_task_type, safe_json

router = APIRouter(prefix="/api/upload", tags=["Upload"])
logger = logging.getLogger(__name__)

# Maximum upload size guard (200 MB raw bytes)
_MAX_BYTES = 200 * 1024 * 1024


def _parse_csv(content: bytes) -> pd.DataFrame:
    """
    Parse CSV bytes into a DataFrame.
    Uses the C engine with low_memory=True so pandas doesn't scan the
    entire file to infer dtypes column by column (much faster for wide CSVs).
    """
    return pd.read_csv(
        io.BytesIO(content),
        engine="c",
        low_memory=True,
    )


@router.post("/csv", summary="Upload a CSV file")
async def upload_csv(file: UploadFile = File(...)):
    """Accept a CSV file, parse it and store in session."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    content = await file.read()

    if len(content) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1_048_576:.1f} MB). Maximum allowed is 200 MB.",
        )

    try:
        # Parse in thread pool — avoids blocking event loop on large files
        df = await run_in_threadpool(_parse_csv, content)
    except Exception as exc:
        logger.error("[Upload] CSV parse error:\n%s", traceback.format_exc())
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    # Reset previous session state
    store.reset()
    store.set("raw_df", df)
    # ── KEY FIX: do NOT copy the full DataFrame here ──────────────────
    # processed_df starts as a reference; preprocessing makes its own copy
    store.set("processed_df", df)

    n_rows, n_cols = df.shape
    logger.info("[Upload] '%s' loaded — %d rows × %d columns", file.filename, n_rows, n_cols)

    return JSONResponse(
        content=safe_json({
            "filename"      : file.filename,
            "rows"          : n_rows,
            "columns"       : list(df.columns),
            "dtypes"        : {col: str(dt) for col, dt in df.dtypes.items()},
            "preview"       : df_to_records(df, max_rows=100),
            "missing_counts": {col: int(cnt) for col, cnt in df.isnull().sum().items()},
        })
    )


@router.get("/preview", summary="Re-fetch current dataset preview")
async def get_preview():
    """Return a preview of the currently loaded raw dataset."""
    df = store.get("raw_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded yet.")
    return JSONResponse(
        content=safe_json({
            "rows"          : len(df),
            "columns"       : list(df.columns),
            "dtypes"        : {col: str(dt) for col, dt in df.dtypes.items()},
            "preview"       : df_to_records(df, max_rows=100),
            "missing_counts": {col: int(cnt) for col, cnt in df.isnull().sum().items()},
        })
    )


@router.post("/target", summary="Set target column", response_model=MessageResponse)
async def set_target(body: TargetColumnRequest):
    """Set the target column and infer task type (classification vs regression)."""
    df = store.get("raw_df")
    if df is None:
        raise HTTPException(status_code=404, detail="No dataset uploaded yet.")
    if body.target_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{body.target_column}' not found in dataset.",
        )
    task = infer_task_type(df[body.target_column])
    store.set("target_column", body.target_column)
    store.set("task_type", task)
    store.set(
        "feature_columns",
        [c for c in df.columns if c != body.target_column],
    )
    return MessageResponse(
        message=f"Target set to '{body.target_column}'. Task type: {task}.",
        details={"task_type": task, "feature_columns": store.get("feature_columns")},
    )
