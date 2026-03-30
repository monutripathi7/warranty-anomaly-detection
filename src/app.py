"""
Warranty Anomaly Detection — FastAPI Server

Serves the trained XGBoost/LightGBM model behind a REST API.
Supports single-claim prediction, batch JSON prediction, and CSV file uploads.
Also hosts an inline HTML dashboard so dealership staff can test claims
without needing a separate frontend build.

Endpoints:
    GET  /                  → dealer dashboard (HTML)
    GET  /health            → liveness check
    POST /predict           → score one claim
    POST /predict/batch     → score a list of claims (JSON body)
    POST /predict/batch/csv → score claims from an uploaded CSV file
    POST /explain           → SHAP feature contributions for one claim

Author: Monish Modi
"""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Literal
from datetime import datetime as dt
import pandas as pd
import numpy as np
import joblib
import json
import io
import logging
import traceback
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema — mirrors the 14 user-facing fields from the claim form.
# Pydantic handles type coercion and constraint checks (ge=0, Literal sets)
# so we get free 422 responses for bad input before any business logic runs.
# ---------------------------------------------------------------------------

class ClaimRequest(BaseModel):
    Mileage: int = Field(ge=0)
    Part_Cost: float = Field(ge=0.0)
    Labour: float = Field(ge=0.0)
    Sublet: float = Field(ge=0.0)
    Claim_Type: Literal["Campaign", "TMA", "Regular", "Free Service Labor Claim"]
    Part_Type: Literal["NONCS1000PARTS", "RS10000PARTS"]
    Cause: Literal["ZZ2", "ZZ3", "ZZ4", "ZZ7"]
    Nature: Literal[
        "L23", "L24", "L31", "W11", "W13", "W17",
        "B32", "B33", "D91", "D92", "A38", "Q26",
        "V84", "V88", "DA1", "DJ6",
    ]
    Status: Literal["Open", "Pending", "Accept", "Suspense(P)"]
    Dealership: Literal["Modi Hyundai", "Viva Honda", "Modi Motors Mumbai", "Modi Motors Pune"]
    Claim_Date: str   # yyyy-mm-dd
    RO_Date: str
    Pdctn_Date: str
    Approve_Amount_by_HMI: float = Field(ge=0.0)


class BatchClaimRequest(BaseModel):
    """Wrapper for the batch JSON endpoint — just a list of individual claims."""
    claims: List[ClaimRequest]


# ---------------------------------------------------------------------------
# App init + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Modi Auto Group - Warranty Anomaly Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # fine for dev; lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading
#
# We check for the XGBoost JSON artifact first because that's what the Colab
# GPU pipeline produces. If it's missing we fall back to the LightGBM pickle
# from the local CPU trainer. If neither exists the server still starts but
# /predict endpoints will return 503.
# ---------------------------------------------------------------------------

model = None
model_loaded = False
model_type = None          # "xgboost" or "lightgbm"
categorical_mappings: dict = {}

if os.path.exists("warranty_model_v1.json"):
    try:
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model("warranty_model_v1.json")
        model_loaded = True
        model_type = "xgboost"
        logger.info("XGBoost model loaded from warranty_model_v1.json")
    except Exception as e:
        logger.warning(f"Failed to load XGBoost model: {e}")
elif os.path.exists("warranty_model_v1.pkl"):
    try:
        model = joblib.load("warranty_model_v1.pkl")
        model_loaded = True
        model_type = "lightgbm"
        logger.info("LightGBM model loaded from warranty_model_v1.pkl")
    except Exception as e:
        logger.warning(f"Failed to load LightGBM model: {e}")
else:
    logger.warning("No model file found. /predict will return 503.")

try:
    with open("categorical_mappings.json") as f:
        categorical_mappings = json.load(f)
    logger.info("Categorical mappings loaded successfully.")
except FileNotFoundError:
    logger.warning("categorical_mappings.json not found.")

# ---------------------------------------------------------------------------
# Feature column order — must match exactly what the model saw during training.
# If you retrain with different features, update this list too.
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # raw numerics straight from the claim
    "Mileage", "Part_Cost", "Labour", "Sublet",
    "Total_Amt", "IGST", "CGST", "SGST",
    "Approve_Amount_by_HMI",
    # derived at inference time (same formulas as trainer.py)
    "Vehicle_Age_Days", "Claim_RO_Gap_Days", "Tax_Rate", "Approval_Ratio",
    # integer-encoded categoricals
    "Claim_Type_idx", "Part_Type_idx", "Cause_idx", "Nature_idx",
    "Status_idx", "Dealership_idx",
]

CATEGORICAL_COLS = ["Claim_Type", "Part_Type", "Cause", "Nature", "Status", "Dealership"]

# hard cap so one rogue request can't OOM the server
MAX_BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# CSV column normalisation
#
# Dealerships might export CSVs with slightly different headers — spaces
# instead of underscores, lowercase, etc. This map lets us accept the most
# common variations without forcing users to rename columns manually.
# ---------------------------------------------------------------------------

_CSV_COL_MAP = {
    "mileage": "Mileage", "part_cost": "Part_Cost", "part cost": "Part_Cost",
    "labour": "Labour", "sublet": "Sublet",
    "claim_type": "Claim_Type", "claim type": "Claim_Type",
    "part_type": "Part_Type", "part type": "Part_Type",
    "cause": "Cause", "nature": "Nature", "status": "Status",
    "dealership": "Dealership",
    "claim_date": "Claim_Date", "claim date": "Claim_Date",
    "ro_date": "RO_Date", "ro date": "RO_Date",
    "pdctn_date": "Pdctn_Date", "production date": "Pdctn_Date",
    "pdctn date": "Pdctn_Date",
    "approve_amount_by_hmi": "Approve_Amount_by_HMI",
    "approved amount (hmi)": "Approve_Amount_by_HMI",
    "approve amount by hmi": "Approve_Amount_by_HMI",
}


def _normalize_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Try to rename CSV columns to match our expected field names.
    Columns that already have the right casing are left untouched."""
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in _CSV_COL_MAP:
            rename[col] = _CSV_COL_MAP[key]
    return df.rename(columns=rename)

# ---------------------------------------------------------------------------
# Shared feature engineering
#
# Both scoring and SHAP explanation need the same feature DataFrame built
# from a raw claim dict. This helper does the date parsing, categorical
# encoding, tax calculation, and derived features — returning either a
# ready-to-score DataFrame or an error dict.
# ---------------------------------------------------------------------------

def _build_feature_df(claim_dict: dict):
    """Turn a raw claim dict into a single-row DataFrame matching FEATURE_COLS.

    Returns (df, None) on success or (None, {"error": str}) on failure.
    """
    try:
        claim_date = dt.strptime(claim_dict["Claim_Date"], "%Y-%m-%d")
        ro_date    = dt.strptime(claim_dict["RO_Date"],    "%Y-%m-%d")
        pdctn_date = dt.strptime(claim_dict["Pdctn_Date"], "%Y-%m-%d")
    except (ValueError, KeyError) as e:
        return None, {"error": f"bad date: {e}"}

    try:
        cat_indices = {}
        for col in CATEGORICAL_COLS:
            value = claim_dict.get(col)
            mapping = categorical_mappings.get(col, {})
            if value not in mapping:
                return None, {"error": f"Unknown value '{value}' for {col}"}
            cat_indices[col + "_idx"] = mapping[value]

        pre_tax   = float(claim_dict["Part_Cost"]) + float(claim_dict["Labour"]) + float(claim_dict["Sublet"])
        igst      = pre_tax * 0.18
        total_amt = pre_tax + igst

        vehicle_age_days  = (claim_date - pdctn_date).days
        claim_ro_gap_days = (claim_date - ro_date).days
        tax_rate          = igst / max(pre_tax, 1e-6)
        approval_ratio    = float(claim_dict["Approve_Amount_by_HMI"]) / max(total_amt, 1e-6)

        row = {
            "Mileage": int(claim_dict["Mileage"]),
            "Part_Cost": float(claim_dict["Part_Cost"]),
            "Labour": float(claim_dict["Labour"]),
            "Sublet": float(claim_dict["Sublet"]),
            "Total_Amt": total_amt, "IGST": igst, "CGST": 0.0, "SGST": 0.0,
            "Approve_Amount_by_HMI": float(claim_dict["Approve_Amount_by_HMI"]),
            "Vehicle_Age_Days": float(vehicle_age_days),
            "Claim_RO_Gap_Days": float(claim_ro_gap_days),
            "Tax_Rate": tax_rate, "Approval_Ratio": approval_ratio,
        }
        row.update(cat_indices)

        df = pd.DataFrame([row], columns=FEATURE_COLS)
        return df, None
    except Exception as e:
        return None, {"error": str(e)}


def _score_single_claim(claim_dict: dict) -> dict:
    """Run feature engineering and model inference on one claim.

    Returns {"anomaly_probability": float, "flag_for_audit": bool}
    on success, or {"error": str} if something is wrong with the row.
    """
    df, err = _build_feature_df(claim_dict)
    if err:
        return err

    try:
        if model_type == "xgboost":
            import xgboost as xgb
            dmat = xgb.DMatrix(df)
            score = float(model.predict(dmat)[0])
        else:
            score = float(model.predict(df)[0])

        return {
            "anomaly_probability": round(score, 4),
            "flag_for_audit": bool(score > 0.8),
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# Inline dashboard HTML
#
# Keeping the dashboard as a single inline string avoids the need for a
# static-files directory or a JS build step. It's not the prettiest
# approach for a large frontend, but for a single-page demo it works well.
# ---------------------------------------------------------------------------

def _build_dashboard_html() -> str:
    """Return the full dealer dashboard as an HTML string."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Modi Auto Group - Warranty Anomaly Detection</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #333; }
  header { background: #1a2744; color: #fff; padding: 20px 32px; display: flex; align-items: center; justify-content: space-between; }
  header h1 { font-size: 1.5rem; font-weight: 600; }
  header span { font-size: 0.9rem; opacity: 0.8; }
  .container { max-width: 1100px; margin: 24px auto; padding: 0 16px; }
  .card { background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 24px; margin-bottom: 24px; }
  .card h2 { font-size: 1.15rem; margin-bottom: 16px; color: #1a2744; border-bottom: 2px solid #e8ecf1; padding-bottom: 8px; }
  .form-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }
  .form-group { display: flex; flex-direction: column; }
  .form-group label { font-size: 0.82rem; font-weight: 600; margin-bottom: 4px; color: #555; }
  .form-group input, .form-group select { padding: 8px 10px; border: 1px solid #ccd1d9; border-radius: 4px; font-size: 0.92rem; }
  .form-group input:focus, .form-group select:focus { outline: none; border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59,130,246,0.15); }
  .btn-row { display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }
  .btn { padding: 10px 22px; border: none; border-radius: 5px; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
  .btn:hover { opacity: 0.85; }
  .btn-primary { background: #1a2744; color: #fff; }
  .btn-green { background: #16a34a; color: #fff; }
  .btn-red { background: #dc2626; color: #fff; }
  #result-area { display: none; }
  .result-box { text-align: center; padding: 28px; border-radius: 8px; }
  .result-box .prob { font-size: 2.8rem; font-weight: 700; }
  .result-box .label { font-size: 1rem; margin-top: 4px; color: #555; }
  .result-box .flag { font-size: 1.2rem; margin-top: 14px; font-weight: 600; padding: 8px 18px; border-radius: 6px; display: inline-block; }
  .flag-safe { background: #dcfce7; color: #166534; }
  .flag-audit { background: #fee2e2; color: #991b1b; }
  .error-msg { color: #dc2626; text-align: center; padding: 12px; font-weight: 500; }
  .sample-section { display: flex; gap: 10px; flex-wrap: wrap; }
  /* SHAP contribution chart */
  .shap-chart { margin-top: 20px; }
  .shap-bar-row { display: flex; align-items: center; margin-bottom: 4px; font-size: 0.82rem; }
  .shap-bar-label { width: 160px; text-align: right; padding-right: 10px; color: #555; flex-shrink: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .shap-bar-track { flex: 1; height: 20px; position: relative; background: #f0f0f0; border-radius: 3px; }
  .shap-bar-fill { position: absolute; top: 0; height: 100%; border-radius: 3px; min-width: 2px; }
  .shap-bar-fill.positive { background: #ef4444; }
  .shap-bar-fill.negative { background: #3b82f6; }
  .shap-bar-val { width: 70px; text-align: left; padding-left: 8px; font-weight: 600; font-size: 0.78rem; }
  .shap-legend { display: flex; gap: 16px; margin-bottom: 10px; font-size: 0.8rem; color: #555; }
  .shap-legend span { display: inline-flex; align-items: center; gap: 4px; }
  .shap-legend .dot { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  .shap-legend .dot-pos { background: #ef4444; }
  .shap-legend .dot-neg { background: #3b82f6; }
  /* batch / CSV upload styles */
  .csv-upload-area { border: 2px dashed #ccd1d9; border-radius: 8px; padding: 28px; text-align: center; cursor: pointer; transition: border-color 0.2s, background 0.2s; }
  .csv-upload-area:hover, .csv-upload-area.dragover { border-color: #3b82f6; background: #eff6ff; }
  .csv-upload-area input[type=file] { display: none; }
  .csv-upload-area p { font-size: 0.92rem; color: #555; margin-top: 6px; }
  .batch-table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 0.85rem; }
  .batch-table th, .batch-table td { padding: 8px 10px; border: 1px solid #e8ecf1; text-align: left; }
  .batch-table th { background: #f5f7fa; font-weight: 600; color: #1a2744; }
  .batch-table .flag-audit-cell { background: #fee2e2; color: #991b1b; font-weight: 600; }
  .batch-table .flag-safe-cell { background: #dcfce7; color: #166534; font-weight: 600; }
  .batch-summary { display: flex; gap: 18px; margin-bottom: 14px; flex-wrap: wrap; }
  .batch-summary .stat { background: #f5f7fa; border-radius: 6px; padding: 12px 20px; text-align: center; }
  .batch-summary .stat .num { font-size: 1.5rem; font-weight: 700; }
  .batch-summary .stat .lbl { font-size: 0.8rem; color: #555; }
</style>
</head>
<body>

<header>
  <h1>Modi Auto Group &mdash; Warranty Anomaly Detection</h1>
  <span>Dealer Dashboard</span>
</header>

<div class="container">

  <!-- quick-fill buttons so reviewers can test without typing -->
  <div class="card">
    <h2>Sample Records</h2>
    <p style="font-size:0.85rem;color:#666;margin-bottom:12px;">Click a button to populate the form with a test record, then submit to see the prediction.</p>
    <div class="sample-section">
      <button class="btn btn-green" onclick="loadSample('normal')">&#9989; Normal Claim</button>
      <button class="btn btn-red" onclick="loadSample('cost')">&#9888; Cost Anomaly</button>
      <button class="btn btn-red" onclick="loadSample('temporal')">&#9888; Temporal Anomaly</button>
    </div>
  </div>

  <!-- CSV batch upload — drag-and-drop or click to browse -->
  <div class="card">
    <h2>Batch Prediction &mdash; CSV Upload</h2>
    <p style="font-size:0.85rem;color:#666;margin-bottom:12px;">Upload a CSV file with multiple warranty claims. Required columns: Mileage, Part_Cost, Labour, Sublet, Claim_Type, Part_Type, Cause, Nature, Status, Dealership, Claim_Date, RO_Date, Pdctn_Date, Approve_Amount_by_HMI.</p>
    <div class="csv-upload-area" id="csv-drop-zone"
         onclick="document.getElementById('csv-file-input').click()"
         ondragover="event.preventDefault(); this.classList.add('dragover')"
         ondragleave="this.classList.remove('dragover')"
         ondrop="handleCsvDrop(event)">
      <input type="file" id="csv-file-input" accept=".csv" onchange="handleCsvFile(this.files[0])">
      <span style="font-size:2rem;">&#128196;</span>
      <p>Drag &amp; drop a CSV file here, or click to browse</p>
    </div>
    <div id="csv-status" style="margin-top:10px;font-size:0.88rem;color:#555;"></div>
  </div>

  <!-- batch results table (hidden until a CSV is processed) -->
  <div class="card" id="batch-result-area" style="display:none;">
    <h2>Batch Results</h2>
    <div class="batch-summary" id="batch-summary"></div>
    <div style="overflow-x:auto;">
      <table class="batch-table" id="batch-table"></table>
    </div>
  </div>

  <!-- single-claim form -->
  <div class="card">
    <h2>Submit Warranty Claim</h2>
    <form id="claim-form" onsubmit="submitClaim(event)">
      <div class="form-grid">
        <div class="form-group">
          <label for="Mileage">Mileage (km)</label>
          <input type="number" id="Mileage" name="Mileage" min="0" required>
        </div>
        <div class="form-group">
          <label for="Part_Cost">Part Cost</label>
          <input type="number" id="Part_Cost" name="Part_Cost" min="0" step="0.01" required>
        </div>
        <div class="form-group">
          <label for="Labour">Labour</label>
          <input type="number" id="Labour" name="Labour" min="0" step="0.01" required>
        </div>
        <div class="form-group">
          <label for="Sublet">Sublet</label>
          <input type="number" id="Sublet" name="Sublet" min="0" step="0.01" required>
        </div>
        <div class="form-group">
          <label for="Approve_Amount_by_HMI">Approved Amount (HMI)</label>
          <input type="number" id="Approve_Amount_by_HMI" name="Approve_Amount_by_HMI" min="0" step="0.01" required>
        </div>
        <div class="form-group">
          <label for="Claim_Type">Claim Type</label>
          <select id="Claim_Type" name="Claim_Type" required>
            <option value="Campaign">Campaign</option>
            <option value="TMA">TMA</option>
            <option value="Regular" selected>Regular</option>
            <option value="Free Service Labor Claim">Free Service Labor Claim</option>
          </select>
        </div>
        <div class="form-group">
          <label for="Part_Type">Part Type</label>
          <select id="Part_Type" name="Part_Type" required>
            <option value="NONCS1000PARTS">NONCS1000PARTS</option>
            <option value="RS10000PARTS">RS10000PARTS</option>
          </select>
        </div>
        <div class="form-group">
          <label for="Cause">Cause</label>
          <select id="Cause" name="Cause" required>
            <option value="ZZ2">ZZ2</option>
            <option value="ZZ3">ZZ3</option>
            <option value="ZZ4">ZZ4</option>
            <option value="ZZ7">ZZ7</option>
          </select>
        </div>
        <div class="form-group">
          <label for="Nature">Nature</label>
          <select id="Nature" name="Nature" required>
            <option value="L23">L23</option>
            <option value="L24">L24</option>
            <option value="L31">L31</option>
            <option value="W11">W11</option>
            <option value="W13">W13</option>
            <option value="W17">W17</option>
            <option value="B32">B32</option>
            <option value="B33">B33</option>
            <option value="D91">D91</option>
            <option value="D92">D92</option>
            <option value="A38">A38</option>
            <option value="Q26">Q26</option>
            <option value="V84">V84</option>
            <option value="V88">V88</option>
            <option value="DA1">DA1</option>
            <option value="DJ6">DJ6</option>
          </select>
        </div>
        <div class="form-group">
          <label for="Status">Status</label>
          <select id="Status" name="Status" required>
            <option value="Open">Open</option>
            <option value="Pending">Pending</option>
            <option value="Accept">Accept</option>
            <option value="Suspense(P)">Suspense(P)</option>
          </select>
        </div>
        <div class="form-group">
          <label for="Dealership">Dealership</label>
          <select id="Dealership" name="Dealership" required>
            <option value="Modi Hyundai">Modi Hyundai</option>
            <option value="Viva Honda">Viva Honda</option>
            <option value="Modi Motors Mumbai">Modi Motors Mumbai</option>
            <option value="Modi Motors Pune">Modi Motors Pune</option>
          </select>
        </div>
        <div class="form-group">
          <label for="Claim_Date">Claim Date</label>
          <input type="date" id="Claim_Date" name="Claim_Date" required>
        </div>
        <div class="form-group">
          <label for="RO_Date">RO Date</label>
          <input type="date" id="RO_Date" name="RO_Date" required>
        </div>
        <div class="form-group">
          <label for="Pdctn_Date">Production Date</label>
          <input type="date" id="Pdctn_Date" name="Pdctn_Date" required>
        </div>
      </div>
      <div class="btn-row">
        <button type="submit" class="btn btn-primary">Analyze Claim</button>
      </div>
    </form>
  </div>

  <!-- single-claim result card -->
  <div class="card" id="result-area">
    <h2>Prediction Result</h2>
    <div class="result-box" id="result-box">
      <div class="prob" id="result-prob"></div>
      <div class="label">Anomaly Probability</div>
      <div class="flag" id="result-flag"></div>
    </div>
    <div class="error-msg" id="result-error" style="display:none;"></div>
    <!-- SHAP feature contribution chart -->
    <div class="shap-chart" id="shap-chart" style="display:none;">
      <h2 style="margin-top:18px;">Feature Contributions (SHAP)</h2>
      <p style="font-size:0.82rem;color:#666;margin-bottom:10px;">Shows which features pushed the anomaly score up (red) or down (blue). Sorted by impact.</p>
      <div class="shap-legend">
        <span><span class="dot dot-pos"></span> Increases anomaly score</span>
        <span><span class="dot dot-neg"></span> Decreases anomaly score</span>
      </div>
      <div id="shap-bars"></div>
    </div>
  </div>

</div>

<script>
/* ---- sample records for one-click testing ---- */
const SAMPLES = {
  normal: {
    Mileage: 45000, Part_Cost: 1200, Labour: 450, Sublet: 0,
    Claim_Type: "Regular", Part_Type: "NONCS1000PARTS", Cause: "ZZ3", Nature: "L24",
    Status: "Accept", Dealership: "Modi Hyundai",
    Claim_Date: "2024-03-15", RO_Date: "2024-03-14", Pdctn_Date: "2021-06-01",
    Approve_Amount_by_HMI: 1800
  },
  cost: {
    Mileage: 5000, Part_Cost: 42000, Labour: 2500, Sublet: 500,
    Claim_Type: "Regular", Part_Type: "RS10000PARTS", Cause: "ZZ4", Nature: "D91",
    Status: "Open", Dealership: "Viva Honda",
    Claim_Date: "2024-02-20", RO_Date: "2024-02-19", Pdctn_Date: "2022-08-15",
    Approve_Amount_by_HMI: 45000
  },
  temporal: {
    Mileage: 120000, Part_Cost: 3500, Labour: 1800, Sublet: 200,
    Claim_Type: "Regular", Part_Type: "NONCS1000PARTS", Cause: "ZZ7", Nature: "W17",
    Status: "Pending", Dealership: "Modi Motors Mumbai",
    Claim_Date: "2024-06-10", RO_Date: "2024-06-09", Pdctn_Date: "2017-01-15",
    Approve_Amount_by_HMI: 5000
  }
};

function loadSample(key) {
  const s = SAMPLES[key];
  for (const [field, value] of Object.entries(s)) {
    const el = document.getElementById(field);
    if (el) el.value = value;
  }
  document.getElementById("result-area").style.display = "none";
}

/* ---- single claim submission ---- */
async function submitClaim(e) {
  e.preventDefault();
  const data = {};
  const fields = ["Mileage","Part_Cost","Labour","Sublet","Approve_Amount_by_HMI",
                   "Claim_Type","Part_Type","Cause","Nature","Status","Dealership",
                   "Claim_Date","RO_Date","Pdctn_Date"];
  for (const f of fields) {
    const el = document.getElementById(f);
    if (["Mileage"].includes(f)) {
      data[f] = parseInt(el.value, 10);
    } else if (["Part_Cost","Labour","Sublet","Approve_Amount_by_HMI"].includes(f)) {
      data[f] = parseFloat(el.value);
    } else {
      data[f] = el.value;
    }
  }

  const resultArea  = document.getElementById("result-area");
  const resultBox   = document.getElementById("result-box");
  const resultProb  = document.getElementById("result-prob");
  const resultFlag  = document.getElementById("result-flag");
  const resultError = document.getElementById("result-error");

  resultArea.style.display = "block";
  resultBox.style.display  = "none";
  resultError.style.display = "none";
  resultProb.textContent = "...";
  document.getElementById("shap-chart").style.display = "none";

  try {
    // call /explain instead of /predict — it returns the same score
    // plus SHAP feature contributions for the bar chart
    const resp = await fetch("/explain", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Request failed");
    }
    const result = await resp.json();
    resultBox.style.display = "block";
    resultProb.textContent = (result.anomaly_probability * 100).toFixed(2) + "%";

    if (result.flag_for_audit) {
      resultBox.style.background = "#fef2f2";
      resultProb.style.color = "#dc2626";
      resultFlag.className = "flag flag-audit";
      resultFlag.innerHTML = "&#9888; FLAG FOR AUDIT";
    } else {
      resultBox.style.background = "#f0fdf4";
      resultProb.style.color = "#166534";
      resultFlag.className = "flag flag-safe";
      resultFlag.innerHTML = "&#10003; No Audit Required";
    }

    // render the SHAP contribution bar chart
    if (result.contributions && result.contributions.length > 0) {
      renderShapChart(result.contributions);
    }
  } catch (err) {
    resultError.style.display = "block";
    resultError.textContent = "Error: " + err.message;
  }
}

/* ---- SHAP contribution bar chart ---- */
function renderShapChart(contributions) {
  const container = document.getElementById('shap-chart');
  const barsDiv   = document.getElementById('shap-bars');
  container.style.display = 'block';

  // find the max absolute SHAP value for scaling the bars
  const maxAbs = Math.max(...contributions.map(c => Math.abs(c.shap_value)), 0.001);

  // show top 12 features to keep the chart readable
  const top = contributions.slice(0, 12);

  let html = '';
  for (const c of top) {
    const pct   = Math.abs(c.shap_value) / maxAbs * 100;
    const cls   = c.shap_value >= 0 ? 'positive' : 'negative';
    const sign  = c.shap_value >= 0 ? '+' : '';
    // positive bars grow right from center, negative bars grow left
    const style = c.shap_value >= 0
      ? 'left:50%; width:' + (pct/2) + '%;'
      : 'right:50%; width:' + (pct/2) + '%;';
    html += '<div class="shap-bar-row">' +
      '<div class="shap-bar-label" title="' + c.feature + '">' + c.label + '</div>' +
      '<div class="shap-bar-track"><div class="shap-bar-fill ' + cls + '" style="' + style + '"></div></div>' +
      '<div class="shap-bar-val" style="color:' + (c.shap_value >= 0 ? '#dc2626' : '#2563eb') + '">' +
        sign + c.shap_value.toFixed(4) + '</div></div>';
  }
  barsDiv.innerHTML = html;
}

/* ---- CSV batch upload handlers ---- */
function handleCsvDrop(e) {
  e.preventDefault();
  document.getElementById('csv-drop-zone').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleCsvFile(file);
}

async function handleCsvFile(file) {
  if (!file || !file.name.endsWith('.csv')) {
    document.getElementById('csv-status').textContent = 'Please select a valid .csv file.';
    return;
  }
  document.getElementById('csv-status').textContent = 'Uploading ' + file.name + '...';
  const formData = new FormData();
  formData.append('file', file);

  try {
    const resp = await fetch('/predict/batch/csv', { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Upload failed');
    }
    const data = await resp.json();
    document.getElementById('csv-status').textContent =
      file.name + ' — ' + data.total + ' claims processed.';
    renderBatchResults(data);
  } catch (err) {
    document.getElementById('csv-status').textContent = 'Error: ' + err.message;
    document.getElementById('batch-result-area').style.display = 'none';
  }
}

/* ---- render the batch results table + summary stats ---- */
function renderBatchResults(data) {
  const area = document.getElementById('batch-result-area');
  area.style.display = 'block';

  const flagged = data.results.filter(r => r.flag_for_audit).length;
  const errors  = data.results.filter(r => r.error).length;
  const safe    = data.total - flagged - errors;

  document.getElementById('batch-summary').innerHTML =
    '<div class="stat"><div class="num">' + data.total + '</div><div class="lbl">Total Claims</div></div>' +
    '<div class="stat"><div class="num" style="color:#166534">' + safe + '</div><div class="lbl">Safe</div></div>' +
    '<div class="stat"><div class="num" style="color:#dc2626">' + flagged + '</div><div class="lbl">Flagged for Audit</div></div>' +
    (errors > 0 ? '<div class="stat"><div class="num" style="color:#b45309">' + errors + '</div><div class="lbl">Errors</div></div>' : '');

  let html = '<thead><tr><th>#</th><th>Dealership</th><th>Claim Type</th>' +
             '<th>Mileage</th><th>Total Cost</th><th>Anomaly %</th><th>Verdict</th></tr></thead><tbody>';
  data.results.forEach(function(r, i) {
    if (r.error) {
      html += '<tr><td>' + (i+1) + '</td><td colspan="5">' + r.error +
              '</td><td class="flag-audit-cell">Error</td></tr>';
    } else {
      const pct     = (r.anomaly_probability * 100).toFixed(2) + '%';
      const cls     = r.flag_for_audit ? 'flag-audit-cell' : 'flag-safe-cell';
      const verdict = r.flag_for_audit ? 'AUDIT' : 'Safe';
      const inp     = r.input || {};
      const cost    = (parseFloat(inp.Part_Cost||0) + parseFloat(inp.Labour||0) +
                       parseFloat(inp.Sublet||0)).toFixed(2);
      html += '<tr><td>' + (i+1) + '</td><td>' + (inp.Dealership||'-') +
              '</td><td>' + (inp.Claim_Type||'-') + '</td><td>' + (inp.Mileage||'-') +
              '</td><td>' + cost + '</td><td>' + pct +
              '</td><td class="' + cls + '">' + verdict + '</td></tr>';
    }
  });
  html += '</tbody>';
  document.getElementById('batch-table').innerHTML = html;
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the dealer dashboard."""
    return HTMLResponse(content=_build_dashboard_html())


@app.get("/health")
def health():
    """Simple liveness probe — also tells the caller whether a model is loaded."""
    return {"status": "healthy", "model_loaded": model_loaded}


@app.post("/predict")
def predict(claim: ClaimRequest):
    """Score a single warranty claim and return an anomaly probability.

    Replicates the same feature engineering pipeline used during training
    so the model sees identical feature distributions at inference time.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # parse dates up front so we can give a clear error message
    try:
        claim_date = dt.strptime(claim.Claim_Date, "%Y-%m-%d")
        ro_date    = dt.strptime(claim.RO_Date,    "%Y-%m-%d")
        pdctn_date = dt.strptime(claim.Pdctn_Date, "%Y-%m-%d")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"bad date format, need yyyy-mm-dd: {e}")

    try:
        # look up the integer index for each categorical
        cat_indices = {}
        for col in CATEGORICAL_COLS:
            value = getattr(claim, col)
            mapping = categorical_mappings.get(col, {})
            if value not in mapping:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown categorical value '{value}' for {col}",
                )
            cat_indices[col + "_idx"] = mapping[value]

        # we don't get raw tax columns at inference, so assume inter-state
        # (18% IGST) which covers the majority of claims
        pre_tax   = claim.Part_Cost + claim.Labour + claim.Sublet
        igst      = pre_tax * 0.18
        cgst      = 0.0
        sgst      = 0.0
        total_amt = pre_tax + igst

        # same derived features the trainer computes
        vehicle_age_days  = (claim_date - pdctn_date).days
        claim_ro_gap_days = (claim_date - ro_date).days

        # guard against division by zero — if there's no billable work,
        # a near-zero ratio is the right semantic answer
        tax_rate       = (igst + cgst + sgst) / max(pre_tax, 1e-6)
        approval_ratio = claim.Approve_Amount_by_HMI / max(total_amt, 1e-6)

        # assemble the feature vector in the exact column order the model expects
        row = {
            "Mileage": claim.Mileage,
            "Part_Cost": claim.Part_Cost,
            "Labour": claim.Labour,
            "Sublet": claim.Sublet,
            "Total_Amt": total_amt,
            "IGST": igst, "CGST": cgst, "SGST": sgst,
            "Approve_Amount_by_HMI": claim.Approve_Amount_by_HMI,
            "Vehicle_Age_Days": float(vehicle_age_days),
            "Claim_RO_Gap_Days": float(claim_ro_gap_days),
            "Tax_Rate": tax_rate,
            "Approval_Ratio": approval_ratio,
        }
        row.update(cat_indices)

        df = pd.DataFrame([row], columns=FEATURE_COLS)

        # run prediction — works with either xgboost or lightgbm
        if model_type == "xgboost":
            import xgboost as xgb
            dmat = xgb.DMatrix(df)
            score = float(model.predict(dmat)[0])
        else:
            score = float(model.predict(df)[0])

        return {
            "anomaly_probability": round(score, 4),
            "flag_for_audit": bool(score > 0.8),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


# ---------------------------------------------------------------------------
# Batch prediction endpoints
#
# In practice, dealerships don't submit claims one at a time — they export
# a day's worth of claims as a CSV and want all of them scored at once.
# Two flavours:
#   /predict/batch     → JSON body with a list of claims (for programmatic use)
#   /predict/batch/csv → multipart file upload (for the dashboard UI)
# ---------------------------------------------------------------------------

@app.post("/predict/batch")
def predict_batch(batch: BatchClaimRequest):
    """Score a list of warranty claims in one request.

    Each claim is scored independently; a bad row returns an inline error
    instead of failing the whole batch. The response includes the original
    input alongside each result so callers can map scores back to their data.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(batch.claims) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Batch too large. Max {MAX_BATCH_SIZE} claims per request.",
        )

    results = []
    for claim in batch.claims:
        claim_dict = claim.model_dump()
        result = _score_single_claim(claim_dict)
        result["input"] = claim_dict
        results.append(result)

    return {"total": len(results), "results": results}


@app.post("/predict/batch/csv")
async def predict_batch_csv(file: UploadFile = File(...)):
    """Accept a CSV file, score every row, return JSON results.

    Column names are normalised so common variations (spaces, lowercase)
    are handled automatically. This endpoint powers the dashboard's
    drag-and-drop CSV upload and can also be called programmatically
    with curl or any HTTP client.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    if len(df) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"CSV has {len(df)} rows. Max {MAX_BATCH_SIZE}.",
        )

    df = _normalize_csv_columns(df)

    results = []
    for _, row in df.iterrows():
        claim_dict = row.to_dict()
        # pandas gives us numpy scalars; convert to native Python for JSON
        claim_dict = {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in claim_dict.items()
        }
        result = _score_single_claim(claim_dict)
        result["input"] = claim_dict
        results.append(result)

    return {"total": len(results), "results": results}


# ---------------------------------------------------------------------------
# SHAP explainability
#
# TreeExplainer gives exact Shapley values for tree-based models (XGBoost,
# LightGBM) in polynomial time. We initialise it lazily on the first
# /explain call so startup isn't slowed down if nobody uses it.
#
# The response includes per-feature SHAP values, the base value (average
# model output), and human-readable feature names so the dashboard can
# render a contribution bar chart without any extra mapping.
# ---------------------------------------------------------------------------

_shap_explainer = None


def _get_shap_explainer():
    """Lazy-init the SHAP TreeExplainer. Cached after first call."""
    global _shap_explainer
    if _shap_explainer is not None:
        return _shap_explainer

    import shap
    _shap_explainer = shap.TreeExplainer(model)
    logger.info("SHAP TreeExplainer initialised.")
    return _shap_explainer


# human-friendly labels for the 19 model features — used in the /explain
# response so the dashboard can show readable names instead of column codes
_FEATURE_LABELS = {
    "Mileage": "Mileage (km)",
    "Part_Cost": "Part Cost",
    "Labour": "Labour Cost",
    "Sublet": "Sublet Cost",
    "Total_Amt": "Total Amount",
    "IGST": "IGST",
    "CGST": "CGST",
    "SGST": "SGST",
    "Approve_Amount_by_HMI": "HMI Approved Amount",
    "Vehicle_Age_Days": "Vehicle Age (days)",
    "Claim_RO_Gap_Days": "Claim-RO Gap (days)",
    "Tax_Rate": "Tax Rate",
    "Approval_Ratio": "Approval Ratio",
    "Claim_Type_idx": "Claim Type",
    "Part_Type_idx": "Part Type",
    "Cause_idx": "Cause Code",
    "Nature_idx": "Nature Code",
    "Status_idx": "Claim Status",
    "Dealership_idx": "Dealership",
}


@app.post("/explain")
def explain(claim: ClaimRequest):
    """Return SHAP feature contributions for a single claim.

    This tells you *why* the model scored a claim the way it did — which
    features pushed the probability up and which pulled it down. Useful
    for auditors who need to understand a flagged claim, and demonstrates
    interpretable ML in an academic context.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    df, err = _build_feature_df(claim.model_dump())
    if err:
        raise HTTPException(status_code=422, detail=err["error"])

    try:
        explainer = _get_shap_explainer()

        if model_type == "xgboost":
            import xgboost as xgb
            dmat = xgb.DMatrix(df)
            shap_values = explainer.shap_values(dmat)
            base_value = float(explainer.expected_value)
            score = float(model.predict(dmat)[0])
        else:
            shap_values = explainer.shap_values(df)
            # LightGBM binary classifier returns [class_0, class_1] arrays
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            base_value = float(
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
            score = float(model.predict(df)[0])

        # shap_values is shape (1, 19) — flatten to a plain list
        sv = shap_values[0] if hasattr(shap_values, '__len__') and len(shap_values) > 0 else shap_values
        if hasattr(sv, 'values'):
            sv = sv.values  # shap Explanation object
        if hasattr(sv, '__len__') and len(sv) > 0 and hasattr(sv[0], '__len__'):
            sv = sv[0]
        contributions = []
        for i, col in enumerate(FEATURE_COLS):
            contributions.append({
                "feature": col,
                "label": _FEATURE_LABELS.get(col, col),
                "shap_value": round(float(sv[i]), 6),
                "feature_value": float(df.iloc[0][col]),
            })

        # sort by absolute impact so the biggest drivers come first
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        return {
            "anomaly_probability": round(score, 4),
            "flag_for_audit": bool(score > 0.8),
            "base_value": round(base_value, 6),
            "contributions": contributions,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explain error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="SHAP explanation failed")


# To run locally: uvicorn app:app --reload
