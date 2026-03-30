"""
Tests for the FastAPI prediction endpoints.

Covers three areas:
  1. Response format — every valid claim should come back with a probability
     in [0, 1] and a flag_for_audit that matches the 0.8 threshold.
  2. Input validation — missing fields, negative numbers, and bogus categorical
     values should all get rejected with 422.
  3. Batch prediction — the /predict/batch and /predict/batch/csv endpoints
     should handle multiple claims in one go and return per-row results.

Uses Hypothesis for property-based testing on the single-claim endpoints
and plain pytest for the batch endpoints.
"""

import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Valid value sets — kept in sync with the Literal types in app.py
# ---------------------------------------------------------------------------

ALLOWED_CLAIM_TYPES = ["Campaign", "TMA", "Regular", "Free Service Labor Claim"]
ALLOWED_PART_TYPES  = ["NONCS1000PARTS", "RS10000PARTS"]
ALLOWED_CAUSES      = ["ZZ2", "ZZ3", "ZZ4", "ZZ7"]
ALLOWED_NATURES = [
    "L23", "L24", "L31", "W11", "W13", "W17",
    "B32", "B33", "D91", "D92", "A38", "Q26",
    "V84", "V88", "DA1", "DJ6",
]
ALLOWED_STATUSES     = ["Open", "Pending", "Accept", "Suspense(P)"]
ALLOWED_DEALERSHIPS  = ["Modi Hyundai", "Viva Honda", "Modi Motors Mumbai", "Modi Motors Pune"]

REQUIRED_FIELDS = [
    "Mileage", "Part_Cost", "Labour", "Sublet",
    "Claim_Type", "Part_Type", "Cause", "Nature",
    "Status", "Dealership", "Claim_Date", "RO_Date",
    "Pdctn_Date", "Approve_Amount_by_HMI",
]


# ---------------------------------------------------------------------------
# Hypothesis strategy — generates a random but valid claim payload
# ---------------------------------------------------------------------------

@st.composite
def valid_claim_request(draw):
    """Build a claim dict that should always pass Pydantic validation."""
    mileage     = draw(st.integers(min_value=0, max_value=200000))
    part_cost   = draw(st.floats(min_value=0.0, max_value=50000.0, allow_nan=False, allow_infinity=False))
    labour      = draw(st.floats(min_value=0.0, max_value=5000.0,  allow_nan=False, allow_infinity=False))
    sublet      = draw(st.floats(min_value=0.0, max_value=2000.0,  allow_nan=False, allow_infinity=False))
    approve_amt = draw(st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False))

    claim_date   = draw(st.dates(min_value=datetime.date(2022, 1, 1), max_value=datetime.date(2024, 12, 31)))
    ro_offset    = draw(st.integers(min_value=0, max_value=7))
    pdctn_offset = draw(st.integers(min_value=30, max_value=1800))

    return {
        "Mileage": mileage,
        "Part_Cost": round(part_cost, 2),
        "Labour": round(labour, 2),
        "Sublet": round(sublet, 2),
        "Claim_Type":  draw(st.sampled_from(ALLOWED_CLAIM_TYPES)),
        "Part_Type":   draw(st.sampled_from(ALLOWED_PART_TYPES)),
        "Cause":       draw(st.sampled_from(ALLOWED_CAUSES)),
        "Nature":      draw(st.sampled_from(ALLOWED_NATURES)),
        "Status":      draw(st.sampled_from(ALLOWED_STATUSES)),
        "Dealership":  draw(st.sampled_from(ALLOWED_DEALERSHIPS)),
        "Claim_Date":  claim_date.isoformat(),
        "RO_Date":     (claim_date - datetime.timedelta(days=ro_offset)).isoformat(),
        "Pdctn_Date":  (claim_date - datetime.timedelta(days=pdctn_offset)).isoformat(),
        "Approve_Amount_by_HMI": round(approve_amt, 2),
    }


# ---------------------------------------------------------------------------
# Test helpers — mock the model so we can control the returned probability
# ---------------------------------------------------------------------------

def _make_test_client(mock_probability: float):
    """Spin up a TestClient with a fake model that always returns `mock_probability`.

    We patch the module-level globals in app.py directly so the endpoint
    handler thinks a real model is loaded. Categorical mappings use the
    same sorted-value scheme the trainer produces.
    """
    cat_mappings = {
        "Claim_Type":  {v: i for i, v in enumerate(sorted(ALLOWED_CLAIM_TYPES))},
        "Part_Type":   {v: i for i, v in enumerate(sorted(ALLOWED_PART_TYPES))},
        "Cause":       {v: i for i, v in enumerate(sorted(ALLOWED_CAUSES))},
        "Nature":      {v: i for i, v in enumerate(sorted(ALLOWED_NATURES))},
        "Status":      {v: i for i, v in enumerate(sorted(ALLOWED_STATUSES))},
        "Dealership":  {v: i for i, v in enumerate(sorted(ALLOWED_DEALERSHIPS))},
    }

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([mock_probability])

    import app as app_module

    original_model    = app_module.model
    original_loaded   = app_module.model_loaded
    original_mappings = app_module.categorical_mappings

    app_module.model               = mock_model
    app_module.model_loaded        = True
    app_module.categorical_mappings = cat_mappings

    client = TestClient(app_module.app)
    return client, app_module, original_model, original_loaded, original_mappings


def _restore_app(app_module, original_model, original_loaded, original_mappings):
    """Put the app module back the way we found it."""
    app_module.model               = original_model
    app_module.model_loaded        = original_loaded
    app_module.categorical_mappings = original_mappings


def _get_base_valid_claim():
    """A known-good claim payload we can mutate for negative tests."""
    return {
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
        "Approve_Amount_by_HMI": 1800.0,
    }


# ---------------------------------------------------------------------------
# Property: response format and audit threshold
#
# For any valid claim, the API must return anomaly_probability in [0, 1]
# and flag_for_audit == (probability > 0.8).
# ---------------------------------------------------------------------------

@given(
    claim=valid_claim_request(),
    mock_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prediction_response_format_and_threshold(claim, mock_prob):
    client, app_module, *restore_args = _make_test_client(mock_prob)
    try:
        resp = client.post("/predict", json=claim)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"

        data = resp.json()
        prob = data["anomaly_probability"]
        assert 0.0 <= prob <= 1.0, f"probability {prob} out of range"
        assert data["flag_for_audit"] == (prob > 0.8), (
            f"flag mismatch: prob={prob}, flag={data['flag_for_audit']}"
        )
    finally:
        _restore_app(app_module, *restore_args)


# ---------------------------------------------------------------------------
# Property: input validation rejects bad inputs
#
# Missing fields, negative numerics, and invalid categoricals should all
# get a 422 before any model code runs.
# ---------------------------------------------------------------------------

# Pydantic validation fires before the handler, so we don't need a loaded model
import app as _app_module
_validation_client = TestClient(_app_module.app)


@given(field=st.sampled_from(REQUIRED_FIELDS))
@settings(max_examples=100)
def test_input_validation_missing_field(field):
    """Dropping any required field should give us a 422."""
    payload = _get_base_valid_claim()
    del payload[field]
    resp = _validation_client.post("/predict", json=payload)
    assert resp.status_code == 422, f"Expected 422 when '{field}' is missing, got {resp.status_code}"


@given(neg_mileage=st.integers(min_value=-100000, max_value=-1))
@settings(max_examples=100)
def test_input_validation_negative_mileage(neg_mileage):
    """Negative mileage should be rejected."""
    payload = _get_base_valid_claim()
    payload["Mileage"] = neg_mileage
    resp = _validation_client.post("/predict", json=payload)
    assert resp.status_code == 422


@given(
    cost_field=st.sampled_from(["Part_Cost", "Labour", "Sublet", "Approve_Amount_by_HMI"]),
    neg_cost=st.floats(min_value=-1e6, max_value=-0.01, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_input_validation_negative_cost(cost_field, neg_cost):
    """Negative cost values should be rejected."""
    payload = _get_base_valid_claim()
    payload[cost_field] = round(neg_cost, 2)
    resp = _validation_client.post("/predict", json=payload)
    assert resp.status_code == 422


@given(data=st.tuples(
    st.sampled_from(["Claim_Type", "Part_Type", "Cause", "Nature", "Status", "Dealership"]),
    st.text(min_size=1, max_size=20).filter(
        lambda s: s not in (
            ALLOWED_CLAIM_TYPES + ALLOWED_PART_TYPES + ALLOWED_CAUSES +
            ALLOWED_NATURES + ALLOWED_STATUSES + ALLOWED_DEALERSHIPS
        )
    ),
))
@settings(max_examples=100)
def test_input_validation_invalid_categorical(data):
    """Made-up categorical values should be rejected."""
    cat_field, invalid_value = data
    payload = _get_base_valid_claim()
    payload[cat_field] = invalid_value
    resp = _validation_client.post("/predict", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Batch prediction tests
# ---------------------------------------------------------------------------

def test_batch_predict_json():
    """Sending two valid claims to /predict/batch should return two scored results."""
    client, app_module, *restore_args = _make_test_client(0.35)
    try:
        claims = [_get_base_valid_claim(), _get_base_valid_claim()]
        resp = client.post("/predict/batch", json={"claims": claims})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert "anomaly_probability" in r
            assert "flag_for_audit" in r
            assert "input" in r   # original claim echoed back
    finally:
        _restore_app(app_module, *restore_args)


def test_batch_predict_empty_list():
    """An empty claims list is valid — should return total=0."""
    client, app_module, *restore_args = _make_test_client(0.5)
    try:
        resp = client.post("/predict/batch", json={"claims": []})
        assert resp.status_code == 200
        assert resp.json()["total"] == 0
    finally:
        _restore_app(app_module, *restore_args)


def test_batch_predict_csv_upload():
    """Uploading a one-row CSV should return one scored result."""
    import io as _io
    client, app_module, *restore_args = _make_test_client(0.92)
    try:
        csv_content = (
            "Mileage,Part_Cost,Labour,Sublet,Claim_Type,Part_Type,Cause,Nature,"
            "Status,Dealership,Claim_Date,RO_Date,Pdctn_Date,Approve_Amount_by_HMI\n"
            "45000,1200,450,0,Regular,NONCS1000PARTS,ZZ3,L24,Accept,Modi Hyundai,"
            "2024-03-15,2024-03-14,2021-06-01,1800\n"
        )
        resp = client.post(
            "/predict/batch/csv",
            files={"file": ("claims.csv", _io.BytesIO(csv_content.encode()), "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["results"][0]["anomaly_probability"] == 0.92
        assert data["results"][0]["flag_for_audit"] is True
    finally:
        _restore_app(app_module, *restore_args)


def test_batch_csv_rejects_non_csv():
    """Uploading a .txt file should be rejected with 422."""
    import io as _io
    client, app_module, *restore_args = _make_test_client(0.5)
    try:
        resp = client.post(
            "/predict/batch/csv",
            files={"file": ("data.txt", _io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 422
    finally:
        _restore_app(app_module, *restore_args)


# ---------------------------------------------------------------------------
# SHAP explainability tests
#
# We can't use a real TreeExplainer with a MagicMock model, so we patch
# the _get_shap_explainer function to return a fake explainer that gives
# us controllable SHAP values.
# ---------------------------------------------------------------------------

def test_explain_returns_contributions():
    """POST /explain should return anomaly_probability, base_value, and
    a contributions list with one entry per feature."""
    from unittest.mock import patch

    client, app_module, *restore_args = _make_test_client(0.65)
    try:
        # build a fake SHAP explainer that returns predictable values
        fake_shap_values = np.zeros((1, 19))
        fake_shap_values[0, 0] = 0.15   # Mileage pushes score up
        fake_shap_values[0, 1] = -0.08  # Part_Cost pulls it down

        fake_explainer = MagicMock()
        fake_explainer.shap_values.return_value = fake_shap_values
        fake_explainer.expected_value = 0.5

        with patch.object(app_module, "_get_shap_explainer", return_value=fake_explainer):
            resp = client.post("/explain", json=_get_base_valid_claim())

        assert resp.status_code == 200
        data = resp.json()

        # must have the prediction score
        assert "anomaly_probability" in data
        assert 0.0 <= data["anomaly_probability"] <= 1.0

        # must have the SHAP base value
        assert "base_value" in data

        # must have exactly 19 feature contributions (one per FEATURE_COL)
        assert "contributions" in data
        assert len(data["contributions"]) == 19

        # each contribution should have the right keys
        for c in data["contributions"]:
            assert "feature" in c
            assert "label" in c
            assert "shap_value" in c
            assert "feature_value" in c

        # contributions should be sorted by absolute SHAP value (biggest first)
        abs_vals = [abs(c["shap_value"]) for c in data["contributions"]]
        assert abs_vals == sorted(abs_vals, reverse=True)
    finally:
        _restore_app(app_module, *restore_args)


def test_explain_rejects_invalid_input():
    """POST /explain with missing fields should return 422, same as /predict."""
    payload = _get_base_valid_claim()
    del payload["Mileage"]
    resp = _validation_client.post("/explain", json=payload)
    assert resp.status_code == 422
