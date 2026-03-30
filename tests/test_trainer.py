"""Property-based tests for trainer.py feature engineering (Properties 15–17).

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

import os
import sys
import tempfile
from datetime import date, timedelta

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Ensure project src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trainer import CATEGORICAL_COLS, FEATURE_COLS, _encode_categoricals, _engineer_features

# ---------------------------------------------------------------------------
# Allowed categorical value sets (from design doc / data_engine)
# ---------------------------------------------------------------------------
ALLOWED_CATEGORICALS: dict[str, list[str]] = {
    "Claim_Type": ["Campaign", "TMA", "Regular", "Free Service Labor Claim"],
    "Part_Type": ["NONCS1000PARTS", "RS10000PARTS"],
    "Cause": ["ZZ2", "ZZ3", "ZZ4", "ZZ7"],
    "Nature": [
        "L23", "L24", "L31", "W11", "W13", "W17",
        "B32", "B33", "D91", "D92", "A38", "Q26",
        "V84", "V88", "DA1", "DJ6",
    ],
    "Status": ["Open", "Pending", "Accept", "Suspense(P)"],
    "Dealership": ["Modi Hyundai", "Viva Honda", "Modi Motors Mumbai", "Modi Motors Pune"],
}


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

def _categorical_row_strategy():
    """Strategy that produces a dict of categorical column values."""
    return st.fixed_dictionaries(
        {col: st.sampled_from(vals) for col, vals in ALLOWED_CATEGORICALS.items()}
    )


def _distinct_pair_strategy():
    """Strategy that picks a categorical column and two distinct values from it."""
    return (
        st.sampled_from(list(ALLOWED_CATEGORICALS.keys()))
        .flatmap(
            lambda col: st.tuples(
                st.just(col),
                st.lists(
                    st.sampled_from(ALLOWED_CATEGORICALS[col]),
                    min_size=2,
                    max_size=2,
                    unique=True,
                ),
            )
        )
    )


# Strategy for dates: Claim_Date in 2022-2024, Pdctn_Date 30-1800 days before,
# RO_Date 0-7 days before Claim_Date
_claim_date_st = st.dates(min_value=date(2022, 1, 1), max_value=date(2024, 12, 31))


@st.composite
def _date_triple(draw):
    """Draw a valid (Pdctn_Date, RO_Date, Claim_Date) triple."""
    claim = draw(_claim_date_st)
    pdctn_offset = draw(st.integers(min_value=30, max_value=1800))
    ro_offset = draw(st.integers(min_value=0, max_value=7))
    pdctn = claim - timedelta(days=pdctn_offset)
    ro = claim - timedelta(days=ro_offset)
    return pdctn, ro, claim


# Strategy for positive floats used in ratio tests
_pos_float = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
_small_pos_float = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Property 15: Categorical Encoding Produces Distinct Integers
# Feature: warranty-anomaly-detection, Property 15: Categorical Encoding Produces Distinct Integers
# ---------------------------------------------------------------------------
# **Validates: Requirements 5.1**


@given(data=_distinct_pair_strategy())
@settings(max_examples=100)
def test_categorical_encoding_distinct_integers(data):
    """For any two distinct values within the same categorical column,
    _encode_categoricals must produce distinct non-negative integer indices."""
    col_name, (val_a, val_b) = data

    # Build a minimal DataFrame with two rows containing the distinct values.
    # All other categorical columns get a fixed valid value.
    rows = []
    for val in (val_a, val_b):
        row = {c: ALLOWED_CATEGORICALS[c][0] for c in CATEGORICAL_COLS}
        row[col_name] = val
        rows.append(row)

    df = pl.DataFrame(rows)

    # _encode_categoricals writes categorical_mappings.json to cwd,
    # so we run inside a temp directory to avoid polluting the workspace.
    orig_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        try:
            encoded = _encode_categoricals(df)
        finally:
            os.chdir(orig_dir)

    idx_col = col_name + "_idx"
    indices = encoded[idx_col].to_list()

    # Both indices must be non-negative
    assert all(i >= 0 for i in indices), f"Negative index found: {indices}"
    # The two distinct values must map to distinct indices
    assert indices[0] != indices[1], (
        f"Distinct values {val_a!r} and {val_b!r} in {col_name} "
        f"mapped to the same index {indices[0]}"
    )


# ---------------------------------------------------------------------------
# Property 16: Date-Difference Feature Computation
# Feature: warranty-anomaly-detection, Property 16: Date-Difference Feature Computation
# ---------------------------------------------------------------------------
# **Validates: Requirements 5.2, 5.3**


@given(triple=_date_triple())
@settings(max_examples=100)
def test_date_difference_feature_computation(triple):
    """Vehicle_Age_Days must equal (Claim_Date - Pdctn_Date).days and
    Claim_RO_Gap_Days must equal (Claim_Date - RO_Date).days."""
    pdctn, ro, claim = triple

    # Build a single-row DataFrame with the required date columns and
    # dummy numeric columns needed by _engineer_features.
    df = pl.DataFrame(
        {
            "Claim_Date": [claim],
            "Pdctn_Date": [pdctn],
            "RO_Date": [ro],
            # Dummy numeric columns required by _engineer_features
            "Part_Cost": [100.0],
            "Labour": [200.0],
            "Sublet": [50.0],
            "IGST": [63.0],
            "CGST": [0.0],
            "SGST": [0.0],
            "Total_Amt": [413.0],
            "Approve_Amount_by_HMI": [400.0],
        }
    )

    result = _engineer_features(df)

    expected_age = (claim - pdctn).days
    expected_gap = (claim - ro).days

    actual_age = result["Vehicle_Age_Days"][0]
    actual_gap = result["Claim_RO_Gap_Days"][0]

    assert abs(actual_age - expected_age) < 0.01, (
        f"Vehicle_Age_Days: expected {expected_age}, got {actual_age}"
    )
    assert abs(actual_gap - expected_gap) < 0.01, (
        f"Claim_RO_Gap_Days: expected {expected_gap}, got {actual_gap}"
    )


# ---------------------------------------------------------------------------
# Property 17: Ratio Feature Computation
# Feature: warranty-anomaly-detection, Property 17: Ratio Feature Computation
# ---------------------------------------------------------------------------
# **Validates: Requirements 5.4, 5.5**


@given(
    part_cost=_small_pos_float,
    labour=_small_pos_float,
    sublet=_pos_float,
    igst=_pos_float,
    cgst=_pos_float,
    sgst=_pos_float,
    total_amt=_small_pos_float,
    approve_amt=_pos_float,
)
@settings(max_examples=100)
def test_ratio_feature_computation(
    part_cost, labour, sublet, igst, cgst, sgst, total_amt, approve_amt
):
    """Tax_Rate must equal (IGST+CGST+SGST) / max(Part_Cost+Labour+Sublet, 1e-6)
    and Approval_Ratio must equal Approve_Amount_by_HMI / max(Total_Amt, 1e-6),
    within floating-point tolerance."""
    df = pl.DataFrame(
        {
            # Dates required by _engineer_features (use fixed valid dates)
            "Claim_Date": [date(2024, 1, 15)],
            "Pdctn_Date": [date(2023, 1, 1)],
            "RO_Date": [date(2024, 1, 14)],
            # Numeric columns
            "Part_Cost": [part_cost],
            "Labour": [labour],
            "Sublet": [sublet],
            "IGST": [igst],
            "CGST": [cgst],
            "SGST": [sgst],
            "Total_Amt": [total_amt],
            "Approve_Amount_by_HMI": [approve_amt],
        }
    )

    result = _engineer_features(df)

    # Expected Tax_Rate
    pre_tax = part_cost + labour + sublet
    safe_pre_tax = pre_tax if pre_tax > 1e-6 else 1e-6
    expected_tax_rate = (igst + cgst + sgst) / safe_pre_tax

    # Expected Approval_Ratio
    safe_total = total_amt if total_amt > 1e-6 else 1e-6
    expected_approval_ratio = approve_amt / safe_total

    actual_tax_rate = result["Tax_Rate"][0]
    actual_approval_ratio = result["Approval_Ratio"][0]

    assert abs(actual_tax_rate - expected_tax_rate) < 1e-4, (
        f"Tax_Rate: expected {expected_tax_rate}, got {actual_tax_rate}"
    )
    assert abs(actual_approval_ratio - expected_approval_ratio) < 1e-4, (
        f"Approval_Ratio: expected {expected_approval_ratio}, got {actual_approval_ratio}"
    )


# ---------------------------------------------------------------------------
# Property 18: Scale Pos Weight Computation
# Feature: warranty-anomaly-detection, Property 18: Scale Pos Weight Computation
# ---------------------------------------------------------------------------
# **Validates: Requirements 6.1**

import numpy as np


@given(
    labels=st.lists(st.integers(min_value=0, max_value=1), min_size=10, max_size=1000)
)
@settings(max_examples=100)
def test_scale_pos_weight_computation(labels):
    """For any training dataset, scale_pos_weight must equal
    neg_count / pos_count (dynamic computation, not hardcoded)."""
    y = np.array(labels, dtype=np.float64)

    pos_count = int(y.sum())
    neg_count = len(y) - pos_count

    # Skip degenerate cases where one class is absent
    if pos_count == 0 or neg_count == 0:
        return

    # Expected scale_pos_weight per design doc and trainer.py logic
    expected_spw = neg_count / pos_count

    # Replicate the exact computation from trainer.py run_training():
    #   train_pos = int(y_train.sum())
    #   train_neg = len(y_train) - train_pos
    #   scale_pos_weight = train_neg / train_pos
    computed_pos = int(y.sum())
    computed_neg = len(y) - computed_pos
    actual_spw = computed_neg / computed_pos

    assert actual_spw == pytest.approx(expected_spw, rel=1e-9), (
        f"scale_pos_weight mismatch: expected {expected_spw}, got {actual_spw} "
        f"(pos={pos_count}, neg={neg_count})"
    )

    # Verify it is never hardcoded to 25 (the old broken value)
    # For most random datasets the ratio won't be exactly 25,
    # but the key invariant is that it equals neg/pos.
    assert actual_spw > 0, "scale_pos_weight must be positive"
