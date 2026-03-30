"""Property-based tests for app.py prediction response and input validation (Properties 19–20).

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 8.3, 8.4**
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import patch, MagicMock
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Allowed categorical value sets (from design doc / app.py Pydantic model)
# ---------------------------------------------------------------------------
ALLOWED_CLAIM_TYPES = ["Campaign", "TMA", "Regular", "Free Service Labor Claim"]
ALLOWED_PART_TYPES = ["NONCS1000PARTS", "RS10000PARTS"]
ALLOWED_CAUSES = ["ZZ2", "ZZ3", "ZZ4", "ZZ7"]
ALLOWED_NATURES = [
    "L23", "L24", "L31", "W11", "W13", "W17",
    "B32", "B33", "D91", "D92", "A38", "Q26",
    "V84", "V88", "DA1", "DJ6",
]
ALLOWED_STATUSES = ["Open", "Pending", "Accept", "Suspense(P)"]
ALLOWED_DEALERSHIPS = ["Modi Hyundai", "Viva Honda", "Modi Motors Mumbai", "Modi Motors Pune"]

REQUIRED_FIELDS = [
    "Mileage", "Part_Cost", "Labour", "Sublet",
    "Claim_Type", "Part_Type", "Cause", "Nature",
    "Status", "Dealership", "Claim_Date", "RO_Date",
    "Pdctn_Date", "Approve_Amount_by_HMI",
]


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def valid_claim_request(draw):
    """Generate a valid ClaimRequest payload dict."""
    mileage = draw(st.integers(min_value=0, max_value=200000))
    part_cost = draw(st.floats(min_value=0.0, max_value=50000.0, allow_nan=False, allow_infinity=False))
    labour = draw(st.floats(min_value=0.0, max_value=5000.0, allow_nan=False, allow_infinity=False))
    sublet = draw(st.floats(min_value=0.0, max_value=2000.0, allow_nan=False, allow_infinity=False))
    approve_amt = draw(st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False))

    claim_type = draw(st.sampled_from(ALLOWED_CLAIM_TYPES))
    part_type = draw(st.sampled_from(ALLOWED_PART_TYPES))
    cause = draw(st.sampled_from(ALLOWED_CAUSES))
    nature = draw(st.sampled_from(ALLOWED_NATURES))
    status = draw(st.sampled_from(ALLOWED_STATUSES))
    dealership = draw(st.sampled_from(ALLOWED_DEALERSHIPS))

    # Generate valid ISO date strings
    claim_date = draw(st.dates(
        min_value=__import__("datetime").date(2022, 1, 1),
        max_value=__import__("datetime").date(2024, 12, 31),
    ))
    ro_offset = draw(st.integers(min_value=0, max_value=7))
    pdctn_offset = draw(st.integers(min_value=30, max_value=1800))

    ro_date = claim_date - __import__("datetime").timedelta(days=ro_offset)
    pdctn_date = claim_date - __import__("datetime").timedelta(days=pdctn_offset)

    return {
        "Mileage": mileage,
        "Part_Cost": round(part_cost, 2),
        "Labour": round(labour, 2),
        "Sublet": round(sublet, 2),
        "Claim_Type": claim_type,
        "Part_Type": part_type,
        "Cause": cause,
        "Nature": nature,
        "Status": status,
        "Dealership": dealership,
        "Claim_Date": claim_date.isoformat(),
        "RO_Date": ro_date.isoformat(),
        "Pdctn_Date": pdctn_date.isoformat(),
        "Approve_Amount_by_HMI": round(approve_amt, 2),
    }


# ---------------------------------------------------------------------------
# Mock model setup — used to create a TestClient with a controllable model
# ---------------------------------------------------------------------------

def _make_test_client(mock_probability: float):
    """Create a TestClient with a mocked model returning the given probability.

    We patch joblib.load and the categorical_mappings so the app starts
    with model_loaded=True and deterministic categorical encoding.
    """
    # Build categorical mappings matching the sorted-value scheme from trainer.py
    cat_mappings = {
        "Claim_Type": {v: i for i, v in enumerate(sorted(ALLOWED_CLAIM_TYPES))},
        "Part_Type": {v: i for i, v in enumerate(sorted(ALLOWED_PART_TYPES))},
        "Cause": {v: i for i, v in enumerate(sorted(ALLOWED_CAUSES))},
        "Nature": {v: i for i, v in enumerate(sorted(ALLOWED_NATURES))},
        "Status": {v: i for i, v in enumerate(sorted(ALLOWED_STATUSES))},
        "Dealership": {v: i for i, v in enumerate(sorted(ALLOWED_DEALERSHIPS))},
    }

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([mock_probability])

    # We need to reload the app module with patched globals
    import importlib
    import app as app_module

    # Patch the module-level globals directly
    original_model = app_module.model
    original_loaded = app_module.model_loaded
    original_mappings = app_module.categorical_mappings

    app_module.model = mock_model
    app_module.model_loaded = True
    app_module.categorical_mappings = cat_mappings

    client = TestClient(app_module.app)

    return client, app_module, original_model, original_loaded, original_mappings


def _restore_app(app_module, original_model, original_loaded, original_mappings):
    """Restore original app module globals after test."""
    app_module.model = original_model
    app_module.model_loaded = original_loaded
    app_module.categorical_mappings = original_mappings


# ---------------------------------------------------------------------------
# Property 19: Prediction Response Format and Threshold
# Feature: warranty-anomaly-detection, Property 19: Prediction Response Format and Threshold
# ---------------------------------------------------------------------------
# **Validates: Requirements 7.1, 7.2, 7.3**


@given(
    claim=valid_claim_request(),
    mock_prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prediction_response_format_and_threshold(claim, mock_prob):
    """For any valid ClaimRequest, the response must contain anomaly_probability
    in [0.0, 1.0] and flag_for_audit must equal (anomaly_probability > 0.8)."""
    client, app_module, orig_model, orig_loaded, orig_mappings = _make_test_client(mock_prob)
    try:
        response = client.post("/predict", json=claim)

        assert response.status_code == 200, (
            f"Expected 200 for valid claim, got {response.status_code}: {response.text}"
        )

        data = response.json()

        # Response must contain anomaly_probability
        assert "anomaly_probability" in data, "Response missing 'anomaly_probability' key"

        # anomaly_probability must be in [0.0, 1.0]
        prob = data["anomaly_probability"]
        assert isinstance(prob, (int, float)), f"anomaly_probability is not numeric: {type(prob)}"
        assert 0.0 <= prob <= 1.0, f"anomaly_probability {prob} not in [0.0, 1.0]"

        # Response must contain flag_for_audit
        assert "flag_for_audit" in data, "Response missing 'flag_for_audit' key"

        # flag_for_audit must equal (anomaly_probability > 0.8)
        expected_flag = prob > 0.8
        assert data["flag_for_audit"] == expected_flag, (
            f"flag_for_audit={data['flag_for_audit']} but anomaly_probability={prob}, "
            f"expected flag_for_audit={expected_flag}"
        )
    finally:
        _restore_app(app_module, orig_model, orig_loaded, orig_mappings)


# ---------------------------------------------------------------------------
# Property 20: Input Validation Rejects Invalid Inputs
# Feature: warranty-anomaly-detection, Property 20: Input Validation Rejects Invalid Inputs
# ---------------------------------------------------------------------------
# **Validates: Requirements 7.4, 8.3, 8.4**


def _get_base_valid_claim():
    """Return a known-valid claim payload for mutation."""
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


# Strategy: pick a required field to remove
_missing_field_strategy = st.sampled_from(REQUIRED_FIELDS)

# Strategy: generate negative mileage
_negative_mileage_strategy = st.integers(min_value=-100000, max_value=-1)

# Strategy: generate negative cost values
_negative_cost_strategy = st.floats(
    min_value=-1e6, max_value=-0.01, allow_nan=False, allow_infinity=False
)

# Strategy: pick a cost field to make negative
_cost_field_strategy = st.sampled_from(["Part_Cost", "Labour", "Sublet", "Approve_Amount_by_HMI"])

# Strategy: generate invalid categorical values
_invalid_categorical_strategy = st.tuples(
    st.sampled_from(["Claim_Type", "Part_Type", "Cause", "Nature", "Status", "Dealership"]),
    st.text(min_size=1, max_size=20).filter(
        lambda s: s not in (
            ALLOWED_CLAIM_TYPES + ALLOWED_PART_TYPES + ALLOWED_CAUSES +
            ALLOWED_NATURES + ALLOWED_STATUSES + ALLOWED_DEALERSHIPS
        )
    ),
)


# We use a plain TestClient for validation tests — model doesn't need to be loaded
# because Pydantic validation happens before the endpoint handler runs.
import app as _app_module
_validation_client = TestClient(_app_module.app)


@given(field=_missing_field_strategy)
@settings(max_examples=100)
def test_input_validation_missing_field(field):
    """For any request missing a required field, API must return HTTP 422."""
    payload = _get_base_valid_claim()
    del payload[field]

    response = _validation_client.post("/predict", json=payload)
    assert response.status_code == 422, (
        f"Expected 422 when '{field}' is missing, got {response.status_code}: {response.text}"
    )


@given(neg_mileage=_negative_mileage_strategy)
@settings(max_examples=100)
def test_input_validation_negative_mileage(neg_mileage):
    """For any request with negative Mileage, API must return HTTP 422."""
    payload = _get_base_valid_claim()
    payload["Mileage"] = neg_mileage

    response = _validation_client.post("/predict", json=payload)
    assert response.status_code == 422, (
        f"Expected 422 for Mileage={neg_mileage}, got {response.status_code}: {response.text}"
    )


@given(cost_field=_cost_field_strategy, neg_cost=_negative_cost_strategy)
@settings(max_examples=100)
def test_input_validation_negative_cost(cost_field, neg_cost):
    """For any request with a negative cost field, API must return HTTP 422."""
    payload = _get_base_valid_claim()
    payload[cost_field] = round(neg_cost, 2)

    response = _validation_client.post("/predict", json=payload)
    assert response.status_code == 422, (
        f"Expected 422 for {cost_field}={neg_cost}, got {response.status_code}: {response.text}"
    )


@given(data=_invalid_categorical_strategy)
@settings(max_examples=100)
def test_input_validation_invalid_categorical(data):
    """For any request with an invalid categorical value, API must return HTTP 422."""
    cat_field, invalid_value = data

    payload = _get_base_valid_claim()
    payload[cat_field] = invalid_value

    response = _validation_client.post("/predict", json=payload)
    assert response.status_code == 422, (
        f"Expected 422 for {cat_field}='{invalid_value}', "
        f"got {response.status_code}: {response.text}"
    )
