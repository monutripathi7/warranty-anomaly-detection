# Warranty Anomaly Detection

AI-powered warranty claim anomaly detection system using XGBoost/LightGBM with FastAPI serving and a dealer-facing web dashboard. Built for automotive dealership groups to flag suspicious claims in real-time.

## Overview

This system detects anomalous warranty claims across automobile dealerships using gradient boosting models trained on synthetic data that mirrors real warranty claim schemas. Dealer staff submit a claim via a web dashboard and get an instant anomaly probability score.

### 5 Anomaly Patterns Detected

| Pattern | Condition | Rationale |
|---------|-----------|-----------|
| Cost-based | Part cost > ₹35K or high cost on low-mileage vehicles | Unusually expensive claims |
| Duplicate VIN | Same vehicle with 4+ claims in 30 days | Rapid repeated claims suggest fraud |
| Tax mismatch | GST deviates from expected 18% rate | Data manipulation indicator |
| Temporal | Regular claims on vehicles older than 5 years | Expired warranty abuse |
| Approval | Accepted claims with < 50% approved amount | Approval irregularities |

## Architecture

```
Data Generation ──> Model Training ──> FastAPI Serving
(10M rows)          (XGBoost/LightGBM)  (REST API + Dashboard)
     │                    │                    │
     ▼                    ▼                    ▼
 Parquet files      .json/.pkl model     localhost:8000
```

## Quick Start

### Option A: Local (CPU)

```bash
pip install polars numpy lightgbm scikit-learn joblib fastapi uvicorn pandas

# Generate synthetic data (100K rows for quick test)
python -c "from data_engine import generate_big_data; generate_big_data(total_records=100_000)"

# Train model
python -c "from trainer import run_training; run_training()"

# Start server
uvicorn app:app --reload
```

Open http://localhost:8000 to access the dashboard.

### Option B: Google Colab (GPU) + VPS

On Colab (set runtime to T4 GPU):
```python
!pip install -q polars xgboost scikit-learn joblib cupy-cuda12x

from google.colab import files
files.upload()  # upload xgb_full_pipeline.py from colab/ folder

from xgb_full_pipeline import run_pipeline
run_pipeline(total_records=2_000_000)  # 2M for free tier RAM

from google.colab import files
files.download('warranty_model_v1.json')
files.download('categorical_mappings.json')
```

On your VPS:
```bash
pip install fastapi uvicorn xgboost pandas
# Place warranty_model_v1.json + categorical_mappings.json alongside app.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web dashboard with claim form and 3 sample test records |
| POST | `/predict` | Submit claim JSON, get anomaly score |
| GET | `/health` | Health check with model status |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Mileage": 45000,
    "Part_Cost": 1200.0,
    "Labour": 450.0,
    "Sublet": 0.0,
    "Claim_Type": "Regular",
    "Part_Type": "NONCS1000PARTS",
    "Cause": "ZZ3",
    "Nature": "L24",
    "Status": "Accept",
    "Dealership": "Modi Hyundai",
    "Claim_Date": "2024-03-15",
    "RO_Date": "2024-03-14",
    "Pdctn_Date": "2021-06-01",
    "Approve_Amount_by_HMI": 1800.0
  }'
```

### Response

```json
{
  "anomaly_probability": 0.0342,
  "flag_for_audit": false
}
```

## Model Details

- 19 features: 9 numeric + 4 engineered + 6 categorical
- Dynamic class weighting (~199:1 for 0.5% anomaly rate)
- 5-fold stratified cross-validation with early stopping (patience=50)
- Regularization: 80% column/row subsampling, min 50 samples per leaf
- Metric: PR-AUC (correct for imbalanced anomaly detection)
- Supports both XGBoost (.json) and LightGBM (.pkl) models

## Project Structure

```
├── app.py                          # FastAPI server + dashboard
├── data_engine.py                  # Synthetic data generation (CPU)
├── trainer.py                      # LightGBM training (CPU)
├── colab/
│   └── xgb_full_pipeline.py       # GPU data gen + XGBoost training (Colab)
├── tests/
│   ├── test_data_engine.py         # 14 property tests
│   ├── test_trainer.py             # 4 property tests
│   └── test_api.py                 # 5 property tests
├── docs/
│   ├── DETAILED_DOCUMENTATION.md   # Full technical documentation
│   └── SUMMARY.md                  # Short overview
└── sample_data/                    # Real dealership data samples (Excel)
```

## Testing

23 property-based tests using [Hypothesis](https://hypothesis.readthedocs.io/) covering 20 correctness properties:

```bash
pip install pytest hypothesis httpx
pytest tests/ -v
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data generation | Polars + NumPy / CuPy (GPU) |
| Model | XGBoost (GPU) / LightGBM (CPU) |
| API | FastAPI + Pydantic |
| Dashboard | Inline HTML/JS |
| Data format | Apache Parquet |
| Testing | pytest + Hypothesis |

## Documentation

- [Detailed Documentation](docs/DETAILED_DOCUMENTATION.md) — full technical reference
- [Summary](docs/SUMMARY.md) — quick overview

## License

MIT
