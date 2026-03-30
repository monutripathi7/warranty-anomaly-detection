# Modi Auto Group — Warranty Anomaly Detection
# Project Summary

## What It Does
AI system that flags suspicious warranty claims across Modi Auto Group's dealerships (Modi Hyundai, Viva Honda, Modi Motors Mumbai, Modi Motors Pune). Dealer staff submit a claim via a web dashboard and get an instant anomaly probability score.

## How It Works
1. Synthetic data (10M rows, 28 columns) mimics real warranty claim patterns from Modi Group's Excel data
2. Gradient boosting model (XGBoost or LightGBM) trained on 19 features with 5-fold cross-validation
3. FastAPI serves predictions via REST API + HTML dashboard with prefilled test records

## 5 Anomaly Patterns Detected
- Cost-based: Part cost > 35K or high cost on low-mileage vehicles
- Duplicate VIN: Same vehicle with 4+ claims in 30 days
- Tax mismatch: GST deviates from expected 18% rate
- Temporal: Regular claims on vehicles older than 5 years
- Approval: Accepted claims with < 50% approved amount

## Key Files
| File | Purpose |
|------|---------|
| `src/app.py` | FastAPI server + dashboard (production) |
| `src/data_engine.py` | Synthetic data generation (CPU) |
| `src/trainer.py` | LightGBM training (CPU) |
| `src/warranty_model_v1.json` | Trained XGBoost model artifact |
| `src/categorical_mappings.json` | Encoding mappings for inference |
| `notebooks/xgb_full_pipeline.py` | GPU data gen + XGBoost training (Colab) |

## Quick Start

### Option A: Local (CPU)
```bash
pip install polars numpy lightgbm scikit-learn joblib fastapi uvicorn pandas
python -c "from data_engine import generate_big_data; generate_big_data(total_records=100_000)"
python -c "from trainer import run_training; run_training()"
uvicorn app:app --reload
```

### Option B: Colab (GPU) + VPS
On Colab (T4 GPU):
```python
!pip install -q polars xgboost scikit-learn joblib cupy-cuda12x
from xgb_full_pipeline import run_pipeline
run_pipeline(total_records=2_000_000)  # 2M for free tier
# Download warranty_model_v1.json + categorical_mappings.json
```

On VPS:
```bash
pip install fastapi uvicorn xgboost pandas
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints
- `GET /` — Web dashboard with form, sample records, CSV upload, and SHAP charts
- `POST /predict` — Score a single claim (JSON in, anomaly score out)
- `POST /predict/batch` — Score multiple claims in one JSON request
- `POST /predict/batch/csv` — Upload a CSV file and score every row
- `POST /explain` — SHAP feature contributions for a single claim
- `GET /health` — Status check

## Model Details
- 19 features: 9 numeric + 4 engineered + 6 categorical
- Dynamic class weighting (~199:1 for 0.5% anomaly rate)
- 5-fold stratified CV with early stopping (patience=50)
- Regularization: 80% column/row subsampling, min 50 samples per leaf
- Metric: PR-AUC (correct for imbalanced anomaly detection)

## Testing
29 tests (23 property-based using hypothesis + 4 batch + 2 explainability) covering 20+ correctness properties:
```bash
pytest tests/ -v
```

## Tech Stack
Polars, NumPy, XGBoost/LightGBM, FastAPI, Pydantic, SHAP, CuPy (GPU), Apache Parquet, pytest + hypothesis
