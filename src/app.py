from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime as dt
import pandas as pd
import numpy as np
import joblib
import json
import logging
import traceback
import os

# -- logging setup --
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -- request schema for the /predict endpoint --

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
    Claim_Date: str  # yyyy-mm-dd
    RO_Date: str
    Pdctn_Date: str
    Approve_Amount_by_HMI: float = Field(ge=0.0)

# -- app setup --

app = FastAPI(title="Modi Auto Group - Warranty Anomaly Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -- model loading: checks for xgboost json first, then lightgbm pkl --

model = None
model_loaded = False
model_type = None  # tracks which framework we loaded
categorical_mappings: dict = {}

# try xgboost first since that's what the colab pipeline produces
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

# -- these must match the exact order used during training --

FEATURE_COLS = [
    # raw numerics from the claim
    "Mileage", "Part_Cost", "Labour", "Sublet",
    "Total_Amt", "IGST", "CGST", "SGST",
    "Approve_Amount_by_HMI",
    # stuff we derive at inference time
    "Vehicle_Age_Days", "Claim_RO_Gap_Days", "Tax_Rate", "Approval_Ratio",
    # encoded categoricals
    "Claim_Type_idx", "Part_Type_idx", "Cause_idx", "Nature_idx",
    "Status_idx", "Dealership_idx",
]

CATEGORICAL_COLS = ["Claim_Type", "Part_Type", "Cause", "Nature", "Status", "Dealership"]

# -- inline dashboard html --


def _build_dashboard_html() -> str:
    """builds the full dealer dashboard page as a single html string.
    kept inline to avoid a separate static files setup."""
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
</style>
</head>
<body>

<header>
  <h1>Modi Auto Group &mdash; Warranty Anomaly Detection</h1>
  <span>Dealer Dashboard</span>
</header>

<div class="container">

  <!-- Sample Records -->
  <div class="card">
    <h2>Sample Records</h2>
    <p style="font-size:0.85rem;color:#666;margin-bottom:12px;">Click a button to populate the form with a test record, then submit to see the prediction.</p>
    <div class="sample-section">
      <button class="btn btn-green" onclick="loadSample('normal')">&#9989; Normal Claim</button>
      <button class="btn btn-red" onclick="loadSample('cost')">&#9888; Cost Anomaly</button>
      <button class="btn btn-red" onclick="loadSample('temporal')">&#9888; Temporal Anomaly</button>
    </div>
  </div>

  <!-- Claim Form -->
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

  <!-- Result Area -->
  <div class="card" id="result-area">
    <h2>Prediction Result</h2>
    <div class="result-box" id="result-box">
      <div class="prob" id="result-prob"></div>
      <div class="label">Anomaly Probability</div>
      <div class="flag" id="result-flag"></div>
    </div>
    <div class="error-msg" id="result-error" style="display:none;"></div>
  </div>

</div>

<script>
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

async function submitClaim(e) {
  e.preventDefault();
  const form = document.getElementById("claim-form");
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

  const resultArea = document.getElementById("result-area");
  const resultBox = document.getElementById("result-box");
  const resultProb = document.getElementById("result-prob");
  const resultFlag = document.getElementById("result-flag");
  const resultError = document.getElementById("result-error");

  resultArea.style.display = "block";
  resultBox.style.display = "none";
  resultError.style.display = "none";
  resultProb.textContent = "...";

  try {
    const resp = await fetch("/predict", {
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
  } catch (err) {
    resultError.style.display = "block";
    resultError.textContent = "Error: " + err.message;
  }
}
</script>
</body>
</html>"""


# -- routes --


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the warranty anomaly detection dashboard."""
    return HTMLResponse(content=_build_dashboard_html())


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model_loaded}


@app.post("/predict")
def predict(claim: ClaimRequest):
    """takes a single warranty claim and returns an anomaly score.
    replicates the same feature engineering pipeline used during training."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # --- Parse dates ---
    try:
        claim_date = dt.strptime(claim.Claim_Date, "%Y-%m-%d")
        ro_date = dt.strptime(claim.RO_Date, "%Y-%m-%d")
        pdctn_date = dt.strptime(claim.Pdctn_Date, "%Y-%m-%d")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"bad date format, need yyyy-mm-dd: {e}")

    try:
        # look up the integer index for each categorical using the saved mappings
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

        # at inference we don't get raw tax columns, so we compute them
        # assuming inter-state (18% IGST) which is the majority case
        pre_tax = claim.Part_Cost + claim.Labour + claim.Sublet
        igst = pre_tax * 0.18
        cgst = 0.0
        sgst = 0.0
        total_amt = pre_tax + igst

        # same derived features the trainer computes
        vehicle_age_days = (claim_date - pdctn_date).days
        claim_ro_gap_days = (claim_date - ro_date).days
        # guard against division by zero — if there's no billable work,
        # a near-zero ratio is the right semantic answer
        tax_rate = (igst + cgst + sgst) / max(pre_tax, 1e-6)
        approval_ratio = claim.Approve_Amount_by_HMI / max(total_amt, 1e-6)

        # assemble the feature vector in the exact column order the model expects
        row = {
            "Mileage": claim.Mileage,
            "Part_Cost": claim.Part_Cost,
            "Labour": claim.Labour,
            "Sublet": claim.Sublet,
            "Total_Amt": total_amt,
            "IGST": igst,
            "CGST": cgst,
            "SGST": sgst,
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


# To run: uvicorn app:app --reload
