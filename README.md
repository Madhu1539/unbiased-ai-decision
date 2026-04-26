# 🤖 Unbiased AI Decision Dashboard

> **Google Hackathon Project — "Unbiased AI Decision" Challenge**

A **complete, production-grade end-to-end ML dashboard** for transparent, fair, and unbiased AI decisions. Upload any CSV dataset and walk through the full machine learning pipeline — from raw data through to bias-aware evaluation, AI-powered fairness auditing, and export.

---

## ✨ ML Pipeline (in order)

| Step | Module | Description |
|------|--------|-------------|
| 1 | 📂 **Data Upload** | CSV upload, dataset preview, target column selection |
| 2 | 📊 **EDA** | Summary statistics, histograms, correlation heatmap |
| 3 | 🧹 **Data Cleaning** | Missing value handling, duplicate removal, type fixes |
| 4 | ⚙️ **Preprocessing** | Encoding, outlier detection, scaling (StandardScaler / MinMax) |
| 5 | ✂️ **Split Data** | Train/test split with configurable ratio and random seed |
| 6 | 🧠 **Feature Engineering** | Random Forest importance ranking, manual feature selection |
| 7 | ⚖️ **Class Imbalance** | SMOTE, ADASYN, class weighting — leakage-safe imblearn pipeline |
| 8 | 🤖 **Model Selection** | 8+ algorithms with auto-recommendation and pros/cons |
| 9 | 🚀 **Training** | Animated training monitor with progress logs |
| 10 | 📈 **Evaluation** | Metrics, confusion matrix, ROC curve, live threshold control |
| 11 | ⚖️ **Bias & Fairness** | Disparate impact, demographic parity, AI audit via Gemini |
| 12 | 🔍 **Predictions** | Batch scatter chart + custom single-row prediction form |
| 13 | 📤 **Reports** | Download processed CSV + PDF evaluation report |

---

## 🤖 Gemini AI Fairness Auditor

The dashboard integrates **Google Gemini 2.0 Flash** to generate structured AI-powered fairness audit reports.

- Sends evaluation metrics (accuracy, F1, fairness scores) to Gemini
- Returns a structured JSON report with:
  - **Bias Summary** — plain-English overview of detected bias
  - **Key Issues** — specific problematic metrics
  - **Root Causes** — likely sources of bias in the data/model
  - **Fairness Risk Level** — High / Medium / Low
  - **Actionable Fixes** — concrete remediation steps
- Accessible via the **"Generate AI Insights"** button on the Bias & Fairness page

### Setup

```bash
# Copy the example env file
cp .env.example .env

# Add your Gemini API key (get one at https://aistudio.google.com/app/apikey)
# Edit .env:
GEMINI_API_KEY=AIza...your_key_here
```

---

## 📁 Project Structure

```
Unbiased AI Decision/
├── .env                  # Your real API keys (gitignored)
├── .env.example          # Safe template to copy from
├── frontend/             # React 18 + Vite + Tailwind CSS
│   └── src/
│       ├── components/   # Sidebar, Header, StatCard, ConfusionMatrix
│       ├── pages/        # 13 ML pipeline pages
│       └── services/     # Axios API client (api.js)
├── backend/              # FastAPI Python backend
│   ├── main.py           # App entry point + route registration
│   ├── routes/           # Route modules per pipeline stage
│   ├── services/
│   │   ├── session_store.py   # In-memory session state
│   │   └── gemini_service.py  # Gemini 2.0 Flash integration
│   └── utils/            # Helpers (safe_json, etc.)
├── ml_pipeline/          # scikit-learn ML modules
│   ├── preprocessing.py
│   ├── eda.py
│   ├── training.py
│   ├── evaluation.py     # Centralized threshold pipeline
│   ├── bias_explanation.py
│   ├── threshold_optimizer.py
│   └── sensitive_feature_detection.py
├── database/             # SQLite + SQLAlchemy
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- **Python** 3.9+
- **Node.js** 18+
- **npm** 9+

---

### 1️⃣ Backend Setup

```bash
# Navigate to project root
cd "Unbiased AI Decision"

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start the API server (from project root)
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

API live at **http://localhost:8000**  
Interactive docs at **http://localhost:8000/docs**

---

### 2️⃣ Frontend Setup

```bash
# In a new terminal
cd "Unbiased AI Decision/frontend"

npm install
npm run dev
```

Dashboard at **http://localhost:5173**

---

## 🔗 Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload/csv` | Upload CSV file |
| POST | `/api/upload/target` | Set target column |
| GET | `/api/eda/summary` | Summary statistics |
| POST | `/api/preprocess` | Run preprocessing |
| POST | `/api/split` | Train/test split |
| GET | `/api/features/importance` | Feature importance |
| POST | `/api/features/select` | Save feature selection |
| GET | `/api/imbalance/status` | Class imbalance analysis |
| POST | `/api/imbalance/apply` | Apply resampling strategy |
| GET | `/api/train/models` | Available models |
| POST | `/api/train` | Train model |
| GET | `/api/evaluate` | Evaluation metrics (threshold=0.5) |
| GET | `/api/evaluate/metrics` | Live metrics at any threshold |
| GET | `/api/evaluate/threshold` | Find optimal threshold |
| POST | `/api/evaluate/threshold/apply` | Apply & persist threshold |
| GET | `/api/evaluate/error-analysis` | FP/FN misclassification analysis |
| POST | `/api/bias/analyze` | Run bias & fairness analysis |
| POST | `/api/fairness-audit` | Gemini AI fairness audit |
| GET | `/api/predict/batch` | Batch predictions |
| POST | `/api/predict/single` | Single row prediction |
| GET | `/api/reports/download/csv` | Download cleaned CSV |
| GET | `/api/reports/download/pdf` | Download PDF report |

---

## 🤖 Supported ML Models

**Classification:** Logistic Regression, Random Forest, Decision Tree, SVM, KNN, Gradient Boosting, XGBoost, Naive Bayes

**Regression:** Linear, Ridge, Lasso, Random Forest, Decision Tree, KNN, SVR, Gradient Boosting, XGBoost

---

## ⚖️ Fairness Metrics

| Metric | Description |
|--------|-------------|
| **Disparate Impact Ratio** | 80% rule — values < 0.8 signal bias |
| **Demographic Parity Difference** | Positive prediction rate gap across groups |
| **Equalized Odds** | True positive rate parity across groups |
| **Group Accuracy** | Per-group accuracy comparison |
| **AI Fairness Audit** | Gemini-generated structured bias report |

---

## 📊 Evaluation Highlights

- **Live threshold control** — drag a slider to recompute all classification metrics in real time (backend is sole source of truth)
- **Optimal threshold finder** — sweeps 200 thresholds, maximises F1 or recall
- **Error analysis panel** — inspect FP/FN misclassifications with confidence scores and top contributing features
- **Regression charts** — Predicted vs Actual scatter + Residual plot
- **ROC curve** — with operating-point marker at current threshold
- **Confusion matrix** — updates live with threshold

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, Tailwind CSS 3, Recharts, Lucide Icons, Axios |
| Backend | FastAPI, Uvicorn, Pydantic v2 |
| ML | scikit-learn, pandas, NumPy, XGBoost, imbalanced-learn |
| AI Audit | Google Gemini 2.0 Flash (`google-genai` SDK) |
| Database | SQLite, SQLAlchemy |
| Reports | ReportLab (PDF) |

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes (for AI audit) | Google Gemini API key from [aistudio.google.com](https://aistudio.google.com/app/apikey) |

---

## 📝 Notes

- The backend uses an **in-memory session store** — state resets on server restart. Retrain after restarting.
- For production, replace the session store with a database-backed solution.
- XGBoost is optional; the system falls back gracefully if not installed.
- Gemini AI audit requires a valid API key and internet connectivity. The rest of the dashboard works fully without it.
- Class imbalance resampling (SMOTE/ADASYN) is applied **only on training data** inside the pipeline to prevent data leakage.

---

*Built with ❤️ for transparent and fair AI decisions — Google Hackathon 2026.*
