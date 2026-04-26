"""
reports.py route  —  /api/reports
Export processed dataset (CSV) and evaluation report (PDF).
"""
import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from backend.services.session_store import store
from ml_pipeline.evaluation import classification_metrics, regression_metrics

router = APIRouter(prefix="/api/reports", tags=["Reports"])


@router.get("/download/csv", summary="Download processed dataset as CSV")
async def download_csv():
    df = store.processed_df
    if df is None:
        raise HTTPException(status_code=404, detail="No processed dataset available.")
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=processed_dataset.csv"},
    )


@router.get("/download/pdf", summary="Download evaluation report as PDF")
async def download_pdf():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        raise HTTPException(status_code=500, detail="reportlab not installed.")

    model = store.get("model")
    if model is None:
        raise HTTPException(status_code=400, detail="Train a model first.")

    y_test = store.get("y_test")
    y_pred = store.get("y_pred")
    X_test = store.get("X_test")
    task = store.task_type
    model_name = store.get("model_name")

    if task == "classification":
        metrics = classification_metrics(y_test, y_pred, model=model, X_test=X_test)
    else:
        metrics = regression_metrics(y_test, y_pred)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Unbiased AI Decision Dashboard", styles["Title"]))
    elements.append(Paragraph("Model Evaluation Report", styles["Heading2"]))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Meta
    elements.append(Paragraph(f"<b>Model:</b> {model_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Task Type:</b> {task}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Test Samples:</b> {len(y_test)}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Metrics table
    elements.append(Paragraph("Performance Metrics", styles["Heading3"]))
    table_data = [["Metric", "Value"]]
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            table_data.append([k.replace("_", " ").title(), f"{v:.4f}"])
    t = Table(table_data, colWidths=[250, 200])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("© Unbiased AI Decision Dashboard", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=evaluation_report.pdf"},
    )
