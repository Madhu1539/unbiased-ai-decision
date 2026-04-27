# ── Stage: Production image ───────────────────────────────────────────────
# python:3.11-slim gives a minimal Debian base with Python 3.11.
# It is much smaller than python:3.11 (full) but still supports
# all scientific libraries (numpy, scikit-learn, xgboost, etc.)
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────
# libgomp1  → required by XGBoost (OpenMP threading)
# build-essential → needed to compile any C extensions during pip install
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────
# Copy only requirements first so Docker layer cache is reused
# on subsequent builds when only source code changes.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# ── Copy project source ───────────────────────────────────────────────────
# Copies backend/, ml_pipeline/, database/, data/ into /app
COPY backend/   ./backend/
COPY ml_pipeline/ ./ml_pipeline/
COPY database/  ./database/
COPY data/      ./data/

# ── Cloud Run port ────────────────────────────────────────────────────────
# Cloud Run injects $PORT at runtime (default 8080).
# We expose it here for documentation; the CMD uses it dynamically.
EXPOSE 8080

# ── Start command ─────────────────────────────────────────────────────────
# Uses shell form so ${PORT:-8080} is evaluated at runtime.
# This means Cloud Run can inject any PORT value and uvicorn will use it.
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080}
