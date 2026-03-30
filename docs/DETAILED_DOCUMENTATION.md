# Modi Auto Group — Warranty Anomaly Detection System
# Detailed Technical Documentation

## 1. Project Overview

This system detects anomalous warranty claims for Modi Auto Group dealerships using AI. Modi Auto Group operates multiple dealerships: Modi Hyundai, Viva Honda, Modi Motors Mumbai (Jaguar Land Rover), and Modi Motors Pune (Jaguar Land Rover).

The system consists of a three-phase pipeline:
1. Synthetic data generation with realistic warranty claim patterns
2. Gradient boosting model training (XGBoost on GPU or LightGBM on CPU)
3. FastAPI web service with a dealer-facing dashboard

## 2. Architecture

```
Phase 1: Data Generation          Phase 2: Training              Phase 3: Serving
┌─────────────────────┐    ┌──────────────────────────┐    ┌─────────────────────────┐
│ data_engine.py      │    │ trainer.py (LightGBM)    │    │ app.py (FastAPI)        │
│ OR                  │───>│ OR                       │───>│                         │
│ xgb_full_pipeline.py│    │ xgb_full_pipeline.py     │    │ GET  /        Dashboard │
│ (Colab GPU)         │    │ (XGBoost GPU on Colab)   │    │ POST /predict Inference │
└─────────────────────┘    └──────────────────────────┘    │ GET  /health  Status    │
        │                           │                      └─────────────────────────┘
        ▼                           ▼                               │
  claims_batch_*.parquet    warranty_model_v1.json/.pkl             ▼
                            categorical_mappings.json         Web Dashboard
```

## 3. File Structure

```
project/
├── src/
│   ├── app.py                      # FastAPI server (production)
│   ├── data_engine.py              # Local CPU data generation
│   ├── trainer.py                  # Local LightGBM training (CPU)
│   ├── warranty_model_v1.json      # Trained XGBoost model artifact
│   └── categorical_mappings.json   # Categorical encoding mappings for inference
├── notebooks/
│   └── xgb_full_pipeline.py       # Combined GPU data gen + XGBoost training
├── tests/
│   ├── test_data_engine.py         # 14 property tests (Properties 1-14)
│   ├── test_trainer.py             # 4 property tests (Properties 15-18)
│   └── test_api.py                 # 11 tests: 5 property (Properties 19-20) + 4 batch + 2 explain
├── docs/
│   ├── DETAILED_DOCUMENTATION.md   # This file
│   └── SUMMARY.md                  # Short overview
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Test dependencies
└── README.md                       # Project overview
```


## 4. Data Schema (28 Columns)

The synthetic data mirrors Modi Group's real WarrantyClaimList schema (26 columns from the Excel files) plus Dealership and Is_Anomaly.

| Column | Type | Description | Generation Method |
|--------|------|-------------|-------------------|
| S_NO | int | Sequential row number | Sequential per chunk |
| VIN | str | Vehicle ID | "MAL" + 12 random alphanumeric chars |
| Claim_No | str | Unique claim ID | "CLM" + zero-padded sequential |
| ACL_No | str | ACL reference | "ACL" + random digits |
| Claim_Date | datetime | Date claim filed | Random in 2022-01-01 to 2024-12-31 |
| Claim_Type | str | Claim category | Random from: Campaign, TMA, Regular, Free Service Labor Claim |
| RO_No | str | Repair order number | "RO" + random digits |
| RO_Date | datetime | Repair order date | Claim_Date minus 0-7 days |
| Status | str | Claim status | Random from: Open, Pending, Accept, Suspense(P) |
| Mileage | int | Vehicle mileage (km) | Gamma(2, 10000) clipped to [0, 200000] |
| Cause | str | Cause code | Random from: ZZ2, ZZ3, ZZ4, ZZ7 |
| Nature | str | Nature code | Random from 16 codes: L23, L24, L31, W11, W13, W17, B32, B33, D91, D92, A38, Q26, V84, V88, DA1, DJ6 |
| Causal_Part | str | Part causing issue | "CP" + random digits |
| Main_OP | str | Main operation code | "OP" + random digits |
| Part_Desc | str | Part description | Random from 20 realistic descriptions |
| Part_Cost | float | Part cost | 40% zeros, remainder lognormal(7, 1) |
| Labour | float | Labour cost | Normal(500, 200) clipped to [100, 5000] |
| Sublet | float | Sublet cost | 90% zeros, remainder uniform(100, 2000) |
| IGST | float | Inter-state GST | 70% of records: 18% of pre-tax |
| CGST | float | Central GST | 30% of records: 9% of pre-tax |
| SGST | float | State GST | 30% of records: 9% of pre-tax |
| Total_Amt | float | Total claim amount | Part_Cost + Labour + Sublet + IGST + CGST + SGST |
| Approve_Amount_by_HMI | float | Manufacturer approved amount | 90-100% of Total_Amt (normal records) |
| Invoice_No | str | Invoice reference | "INV" + random digits |
| Part_Type | str | Part category | Random from: NONCS1000PARTS, RS10000PARTS |
| Pdctn_Date | datetime | Vehicle production date | Claim_Date minus 30-1800 days |
| Dealership | str | Dealership name | Random from: Modi Hyundai, Viva Honda, Modi Motors Mumbai, Modi Motors Pune |
| Is_Anomaly | int (0/1) | Anomaly label | OR of 5 anomaly patterns (training only) |

## 5. Anomaly Detection Patterns

Five distinct anomaly patterns are used, applied via OR logic. Target anomaly rate: 0.3-1.0%.

### Pattern 1: Cost-Based Anomaly
- Condition: Part_Cost > 35,000 OR (Mileage < 1,000 AND Part_Cost > 8,000)
- Rationale: Unusually expensive parts or high-cost claims on nearly-new vehicles

### Pattern 2: Duplicate VIN Claims
- Condition: Same VIN has > 3 claims within a 30-day window
- Rationale: Rapid repeated claims on the same vehicle suggest fraud or system abuse
- Implementation: VIN clusters of 4-6 records with dates spread within 0-15 days

### Pattern 3: Tax Mismatch
- Condition: |actual_tax - 0.18 * pre_tax| / pre_tax > 0.01
- Rationale: GST should be exactly 18% (inter-state) or 9%+9% (intra-state). Deviations indicate data manipulation
- Safe division: Uses max(pre_tax, 1e-6) to avoid division by zero

### Pattern 4: Temporal Anomaly
- Condition: (Claim_Date - Pdctn_Date) > 5 years AND Claim_Type == "Regular"
- Rationale: Regular warranty claims on vehicles older than 5 years are suspicious (warranty typically expires)

### Pattern 5: Approval Amount Anomaly
- Condition: Approve_Amount_by_HMI < 50% of Total_Amt AND Status == "Accept"
- Rationale: Accepted claims with very low approval amounts suggest irregularities

### Anomaly Rate Control
- Natural anomalies are detected first from generated data
- If rate < 0.3%: additional anomalous records are injected (distributed across all 5 patterns)
- If rate > 1.0%: excess anomaly labels are randomly flipped to 0
- Injection distribution: ~25% cost, ~20% duplicate VIN, ~20% tax mismatch, ~20% temporal, ~15% approval

## 6. Feature Engineering (19 Features)

The model uses 19 features organized into three groups:

### 9 Raw Numeric Features
| Feature | Description |
|---------|-------------|
| Mileage | Vehicle mileage at claim time |
| Part_Cost | Part cost amount |
| Labour | Labour cost amount |
| Sublet | Sublet cost amount |
| Total_Amt | Total claim amount including tax |
| IGST | Inter-state GST amount |
| CGST | Central GST amount |
| SGST | State GST amount |
| Approve_Amount_by_HMI | Manufacturer approved amount |

### 4 Engineered Features
| Feature | Formula | Safe Division |
|---------|---------|---------------|
| Vehicle_Age_Days | (Claim_Date - Pdctn_Date).days | N/A |
| Claim_RO_Gap_Days | (Claim_Date - RO_Date).days | N/A |
| Tax_Rate | (IGST + CGST + SGST) / max(Part_Cost + Labour + Sublet, 1e-6) | max(denom, 1e-6) |
| Approval_Ratio | Approve_Amount_by_HMI / max(Total_Amt, 1e-6) | max(denom, 1e-6) |

### 6 Categorical Index Features
| Feature | Source Column | Encoding |
|---------|--------------|----------|
| Claim_Type_idx | Claim_Type | Deterministic sorted-value mapping |
| Part_Type_idx | Part_Type | Deterministic sorted-value mapping |
| Cause_idx | Cause | Deterministic sorted-value mapping |
| Nature_idx | Nature | Deterministic sorted-value mapping |
| Status_idx | Status | Deterministic sorted-value mapping |
| Dealership_idx | Dealership | Deterministic sorted-value mapping |

Categorical mappings are saved to `categorical_mappings.json` during training and loaded at inference time to ensure consistency.

## 7. Model Training

### Two Training Options

#### Option A: LightGBM on CPU (trainer.py)
- Local execution, no GPU required
- Uses LightGBM with GOSS (Gradient-based One-Side Sampling)
- Model saved as `warranty_model_v1.pkl` (joblib)

#### Option B: XGBoost on GPU (colab/xgb_full_pipeline.py)
- Google Colab with T4 GPU
- Uses XGBoost with `tree_method="hist"` and `device="cuda"`
- CuPy for GPU-accelerated numeric data generation
- Model saved as `warranty_model_v1.json` (portable JSON format)

### Training Configuration

| Parameter | LightGBM Value | XGBoost Value |
|-----------|----------------|---------------|
| Objective | binary | binary:logistic |
| Metric | average_precision (PR-AUC) | aucpr |
| scale_pos_weight | Dynamic: neg_count / pos_count (~199 for 0.5% anomaly rate) | Same |
| Learning rate | 0.05 | 0.05 |
| Tree depth/leaves | num_leaves=63 | max_depth=6 |
| Min samples per leaf | min_child_samples=50 | min_child_weight=50 |
| Column subsampling | feature_fraction=0.8 | colsample_bytree=0.8 |
| Row subsampling | bagging_fraction=0.8 | subsample=0.8 |
| Max rounds | 1000 | 1000 |
| Early stopping | Patience=50 on validation PR-AUC | Same |

### Training Pipeline
1. Load all Parquet files via Polars lazy scan
2. Encode 6 categorical columns to integer indices (deterministic sorted mapping)
3. Save categorical mappings to JSON
4. Engineer 4 derived features with safe division
5. 80/20 stratified train/test split
6. Compute dynamic scale_pos_weight from label distribution
7. 5-fold stratified cross-validation (log per-fold and mean PR-AUC ± std)
8. Final training with 90/10 train/validation split and early stopping
9. Log feature importance ranked by gain
10. Evaluate on held-out test set (print PR-AUC)
11. Save model artifact

## 8. API Server (app.py)

### Endpoints

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| GET | `/` | HTML dashboard with claim form, sample records, and CSV upload | HTML page |
| GET | `/health` | Health check | `{"status": "healthy", "model_loaded": true/false}` |
| POST | `/predict` | Score a single warranty claim | `{"anomaly_probability": 0.0342, "flag_for_audit": false}` |
| POST | `/predict/batch` | Score multiple claims (JSON body) | `{"total": N, "results": [...]}` |
| POST | `/predict/batch/csv` | Score claims from an uploaded CSV file | `{"total": N, "results": [...]}` |
| POST | `/explain` | SHAP feature contributions for one claim | `{"anomaly_probability": ..., "base_value": ..., "contributions": [...]}` |

### Model Loading
- Auto-detects model format at startup
- Checks for `warranty_model_v1.json` (XGBoost) first, then `warranty_model_v1.pkl` (LightGBM)
- Sets `model_loaded=False` if neither found; `/predict` returns HTTP 503
- Loads `categorical_mappings.json` for inference encoding consistency

### Prediction Request (POST /predict)
```json
{
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
}
```

### Prediction Response
```json
{
  "anomaly_probability": 0.0342,
  "flag_for_audit": false
}
```

### Input Validation (Pydantic)
- Mileage: non-negative integer
- Part_Cost, Labour, Sublet, Approve_Amount_by_HMI: non-negative floats
- Claim_Type: must be one of 4 allowed values
- Part_Type: must be one of 2 allowed values
- Cause: must be one of 4 allowed values
- Nature: must be one of 16 allowed values
- Status: must be one of 4 allowed values
- Dealership: must be one of 4 allowed values
- Dates: ISO format (YYYY-MM-DD), validated at parse time
- Invalid inputs return HTTP 422 with descriptive error messages

### Inference Feature Engineering
At inference time, the `/predict` endpoint replicates the training feature pipeline on a single record:
1. Encode categoricals using saved `categorical_mappings.json`
2. Compute tax: IGST = pre_tax * 0.18 (assumes inter-state for inference)
3. Compute Total_Amt = pre_tax + IGST
4. Compute Vehicle_Age_Days, Claim_RO_Gap_Days from parsed dates
5. Compute Tax_Rate and Approval_Ratio with safe division
6. Build 19-feature DataFrame matching FEATURE_COLS order exactly
7. Call model.predict() (XGBoost DMatrix or LightGBM direct)

### Batch Prediction

In practice, dealerships don't submit claims one at a time — they export a day's worth of claims as a CSV and want all of them scored at once. Two batch endpoints support this workflow:

#### POST /predict/batch (JSON)
Accepts a JSON body with a `claims` array containing up to 500 `ClaimRequest` objects. Each claim is scored independently; a bad row returns an inline error instead of failing the whole batch. The response includes the original input alongside each result so callers can map scores back to their records.

```json
{
  "claims": [
    {"Mileage": 45000, "Part_Cost": 1200, "...": "..."},
    {"Mileage": 5000, "Part_Cost": 42000, "...": "..."}
  ]
}
```

Response:
```json
{
  "total": 2,
  "results": [
    {"anomaly_probability": 0.0342, "flag_for_audit": false, "input": {"..."}},
    {"anomaly_probability": 0.9100, "flag_for_audit": true, "input": {"..."}}
  ]
}
```

#### POST /predict/batch/csv (File Upload)
Accepts a multipart CSV file upload. Column names are normalised so common variations (spaces instead of underscores, lowercase, etc.) are handled automatically. This endpoint powers the dashboard's drag-and-drop CSV upload and can also be called programmatically:

```bash
curl -X POST http://localhost:8000/predict/batch/csv -F "file=@claims.csv"
```

The response format is identical to the JSON batch endpoint. Max 500 rows per request.

#### Shared Scoring Logic
Both batch endpoints and the single `/predict` endpoint share the same feature engineering pipeline via an internal `_score_single_claim()` helper. This avoids code duplication and ensures consistent behaviour across all prediction paths.

### CORS
- All origins allowed (`allow_origins=["*"]`) for development
- Should be restricted to specific domains in production

## 9. Model Explainability (SHAP)

### Overview
The `/explain` endpoint uses SHAP (SHapley Additive exPlanations) to decompose a prediction into per-feature contributions. For tree-based models like XGBoost and LightGBM, SHAP's `TreeExplainer` computes exact Shapley values in polynomial time — no sampling or approximation needed.

This is important for two reasons:
1. Auditors need to understand *why* a claim was flagged, not just that it was flagged
2. Interpretable ML is a core requirement in regulated industries and academic evaluation

### How It Works
1. The claim goes through the same feature engineering pipeline as `/predict`
2. SHAP's `TreeExplainer` decomposes the model output into additive contributions from each of the 19 features
3. The response includes a `base_value` (average model output across training data) and a `contributions` array sorted by absolute impact

### Request / Response

```json
POST /explain
{
  "Mileage": 45000,
  "Part_Cost": 42000,
  "...": "..."
}
```

```json
{
  "anomaly_probability": 0.91,
  "flag_for_audit": true,
  "base_value": 0.005,
  "contributions": [
    {"feature": "Part_Cost", "label": "Part Cost", "shap_value": 0.4523, "feature_value": 42000.0},
    {"feature": "Approval_Ratio", "label": "Approval Ratio", "shap_value": 0.2101, "feature_value": 0.85},
    {"feature": "Vehicle_Age_Days", "label": "Vehicle Age (days)", "shap_value": -0.0312, "feature_value": 1020.0},
    "..."
  ]
}
```

Positive SHAP values push the anomaly score up; negative values pull it down. The `contributions` array is sorted by absolute impact so the biggest drivers appear first.

### Dashboard Integration
When a claim is submitted through the dashboard, it calls `/explain` instead of `/predict`. The result card shows the anomaly probability alongside a horizontal bar chart of the top 12 feature contributions — red bars for features increasing the score, blue bars for features decreasing it.

### Lazy Initialization
The SHAP `TreeExplainer` is initialized on the first `/explain` call and cached for subsequent requests. This avoids slowing down server startup when explainability isn't needed.

## 10. Web Dashboard

### Features
- Modi Auto Group branding (dark blue header)
- Responsive grid layout (2-3 columns)
- Form with all 14 input fields: number inputs, date pickers, dropdowns
- JavaScript fetch() to POST /explain on form submit (returns score + SHAP breakdown)
- Result display with anomaly probability percentage
- SHAP feature contribution bar chart (top 12 features, red = increases score, blue = decreases)
- Red warning indicator when flag_for_audit is true (probability > 80%)
- Green "No Audit Required" indicator when safe
- CSV batch upload with drag-and-drop support
- Batch results table with summary statistics (total, safe, flagged, errors)

### 3 Prefilled Sample Records
| Sample | Button Color | Key Values | Expected Outcome |
|--------|-------------|------------|------------------|
| Normal Claim | Green | Mileage=45000, Part_Cost=1200, Regular, Modi Hyundai | Low anomaly score |
| Cost Anomaly | Red | Part_Cost=42000, RS10000PARTS, Viva Honda | High anomaly score (Part_Cost > 35000) |
| Temporal Anomaly | Red | Mileage=120000, Regular, Pdctn_Date=2017-01-15 | High anomaly score (vehicle > 5 years old) |

## 11. Google Colab Pipeline

### Setup (T4 GPU runtime)
```python
!pip install -q polars xgboost scikit-learn joblib cupy-cuda12x
```

### Data Generation
- Uses CuPy for GPU-accelerated random number generation (gamma, lognormal, normal, uniform distributions)
- String operations and Parquet writes run on CPU (not GPU-parallelizable)
- Default: 10M rows in 10 chunks of 1M each
- For Colab free tier (12GB RAM): use 2M rows to avoid OOM crashes

### Training
- XGBoost with `tree_method="hist"` and `device="cuda"` (XGBoost 3.x syntax)
- Auto-detects GPU availability, falls back to CPU
- Model saved as portable JSON format (no pickle version issues)

### Artifacts to Download
1. `warranty_model_v1.json` — trained XGBoost model
2. `categorical_mappings.json` — categorical encoding mappings

## 12. Testing

### Framework
- pytest for unit/integration tests
- hypothesis for property-based tests (PBT)
- starlette.testclient for API tests

### 20 Correctness Properties

| # | Property | Test File | Validates |
|---|----------|-----------|-----------|
| 1 | Schema Completeness (28 columns) | test_data_engine.py | Req 1.1 |
| 2 | VIN Format (^MAL[A-Z0-9]+$) | test_data_engine.py | Req 1.2 |
| 3 | Categorical Field Membership | test_data_engine.py | Req 1.3-1.7, 2.7 |
| 4 | Part Cost Zero Distribution (>=40%) | test_data_engine.py | Req 2.1 |
| 5 | Mileage Range [0, 200000] | test_data_engine.py | Req 2.2 |
| 6 | Labour Range [100, 5000] | test_data_engine.py | Req 2.3 |
| 7 | Total Amount = sum of components | test_data_engine.py | Req 2.4 |
| 8 | Date Ordering (Pdctn < RO <= Claim) | test_data_engine.py | Req 2.5, 2.6 |
| 9 | Cost-Based Anomaly Labeling | test_data_engine.py | Req 3.1 |
| 10 | Duplicate VIN Anomaly Labeling | test_data_engine.py | Req 3.2 |
| 11 | Tax Mismatch Anomaly Labeling | test_data_engine.py | Req 3.3 |
| 12 | Temporal Anomaly Labeling | test_data_engine.py | Req 3.4 |
| 13 | Approval Amount Anomaly Labeling | test_data_engine.py | Req 3.5 |
| 14 | Overall Anomaly Rate (0.3-1.0%) | test_data_engine.py | Req 3.6 |
| 15 | Categorical Encoding Distinct Integers | test_trainer.py | Req 5.1 |
| 16 | Date-Difference Feature Computation | test_trainer.py | Req 5.2, 5.3 |
| 17 | Ratio Feature Computation | test_trainer.py | Req 5.4, 5.5 |
| 18 | Scale Pos Weight = neg/pos | test_trainer.py | Req 6.1 |
| 19 | Prediction Response Format & Threshold | test_api.py | Req 7.1-7.3 |
| 20 | Input Validation Rejects Invalid Inputs | test_api.py | Req 7.4, 8.3, 8.4 |
| 21-24 | Batch Prediction (JSON + CSV + edge cases) | test_api.py | Req 7.1-7.3 |
| 25-26 | SHAP Explainability (response format + validation) | test_api.py | Req 7.1-7.3 |

### Running Tests
```bash
pip install pytest hypothesis httpx
pytest tests/ -v   # 29 tests total
```

## 13. Deployment

### VPS Deployment (CPU-only)
The trained model and categorical mappings are included in `src/`, so deployment only requires the `src/` directory:
- `src/app.py`
- `src/warranty_model_v1.json` (XGBoost) OR `warranty_model_v1.pkl` (LightGBM)
- `src/categorical_mappings.json`
- `src/data_engine.py` (only if you need to regenerate data on the server)
- `src/trainer.py` (only if you need to retrain on the server)

```bash
pip install fastapi uvicorn xgboost pandas
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Dependencies

#### Production (VPS)
- fastapi
- uvicorn
- xgboost (if using XGBoost model) OR lightgbm (if using LightGBM model)
- pandas
- shap
- joblib (if using LightGBM model)

#### Training (Local)
- polars
- numpy
- lightgbm
- scikit-learn
- joblib

#### Training (Colab GPU)
- polars
- xgboost
- scikit-learn
- joblib
- cupy-cuda12x (for GPU data generation)

#### Testing
- pytest
- hypothesis
- httpx (for API tests)

## 14. Error Handling

| Scenario | Handling | HTTP Code |
|----------|----------|-----------|
| Model file not found at startup | model_loaded=False, log warning | 503 on /predict |
| Missing request field | Pydantic auto-validation | 422 |
| Invalid categorical value | Pydantic Literal type rejection | 422 |
| Negative numeric value | Pydantic Field(ge=0) rejection | 422 |
| Invalid date format | Caught in endpoint handler | 422 |
| Unknown categorical in mappings | Explicit check against mappings dict | 422 |
| Model prediction error | Catch exception, log traceback | 500 |
| SHAP explanation failure | Catch exception, log traceback | 500 |
| Batch too large (>500 claims) | Size check before processing | 422 |
| Non-CSV file uploaded | Extension check | 422 |
| Unparseable CSV file | Catch pandas read_csv error | 422 |
| Bad row in batch | Inline error per row, batch continues | 200 (with error in result) |
| No Parquet files for training | FileNotFoundError raised | N/A |
| No class variation in labels | ValueError raised | N/A |
| Division by zero in features | max(denominator, 1e-6) | N/A |

## 15. Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Data generation | Polars + NumPy (CPU) / CuPy (GPU) | Columnar ops, chunked Parquet writes |
| Model (local) | LightGBM | Fast GBDT, native categorical support |
| Model (Colab) | XGBoost | Native CUDA GPU support, portable JSON format |
| Serialization | joblib (LightGBM) / JSON (XGBoost) | Standard persistence formats |
| API | FastAPI + Pydantic | Auto validation, OpenAPI docs |
| Dashboard | Inline HTML/JS | Zero build step, served by FastAPI |
| Data format | Apache Parquet | Columnar compression, efficient for 10M+ rows |
| Testing | pytest + hypothesis | Property-based testing for correctness guarantees |
| Explainability | SHAP (TreeExplainer) | Exact Shapley values for tree models, no approximation |
