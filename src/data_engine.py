import polars as pl
import numpy as np
from datetime import datetime, timedelta

def _generate_dates(chunk_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """make three correlated date arrays for each claim record.
    claim date is random in 2022-2024, repair order is 0-7 days before,
    and production date is 30-1800 days before the claim."""
    # Claim_Date: random date in 2022-01-01 to 2024-12-31
    start = datetime(2022, 1, 1)
    end = datetime(2024, 12, 31)
    total_days = (end - start).days  # 1095 days

    claim_offsets = np.random.randint(0, total_days + 1, size=chunk_size)
    claim_dates = np.array([start + timedelta(days=int(d)) for d in claim_offsets])

    # RO_Date: Claim_Date minus 0–7 days (so RO_Date <= Claim_Date)
    ro_deltas = np.random.randint(0, 8, size=chunk_size)
    ro_dates = np.array([cd - timedelta(days=int(rd)) for cd, rd in zip(claim_dates, ro_deltas)])

    # Pdctn_Date: Claim_Date minus 30–1800 days (so Pdctn_Date < Claim_Date by at least 30 days)
    pdctn_deltas = np.random.randint(30, 1801, size=chunk_size)
    pdctn_dates = np.array([cd - timedelta(days=int(pd)) for cd, pd in zip(claim_dates, pdctn_deltas)])

    return claim_dates, ro_dates, pdctn_dates


def _compute_taxes(df: pl.DataFrame) -> pl.DataFrame:
    """add GST columns based on whether the transaction is inter-state or intra-state.
    roughly 70% of claims are inter-state (18% IGST), rest are intra-state (9% CGST + 9% SGST)."""
    n = len(df)
    pre_tax = (df["Part_Cost"] + df["Labour"] + df["Sublet"]).to_numpy()

    # Random mask: True = inter-state (70%), False = intra-state (30%)
    is_interstate = np.random.random(n) < 0.70

    igst = np.where(is_interstate, pre_tax * 0.18, 0.0)
    cgst = np.where(is_interstate, 0.0, pre_tax * 0.09)
    sgst = np.where(is_interstate, 0.0, pre_tax * 0.09)

    return df.with_columns([
        pl.Series("IGST", igst, dtype=pl.Float64),
        pl.Series("CGST", cgst, dtype=pl.Float64),
        pl.Series("SGST", sgst, dtype=pl.Float64),
    ])


def _generate_chunk(chunk_size: int, chunk_index: int) -> pl.DataFrame:
    """build one chunk of synthetic warranty claims matching the real schema.
    generates all 26 columns from the actual WarrantyClaimList plus Dealership."""
    rng = np.random.default_rng()

    # --- Categorical allowed sets ---
    claim_types = ["Campaign", "TMA", "Regular", "Free Service Labor Claim"]
    part_types = ["NONCS1000PARTS", "RS10000PARTS"]
    causes = ["ZZ2", "ZZ3", "ZZ4", "ZZ7"]
    natures = ["L23", "L24", "L31", "W11", "W13", "W17", "B32", "B33",
               "D91", "D92", "A38", "Q26", "V84", "V88", "DA1", "DJ6"]
    statuses = ["Open", "Pending", "Accept", "Suspense(P)"]
    dealerships = ["Modi Hyundai", "Viva Honda", "Modi Motors Mumbai", "Modi Motors Pune"]
    part_descriptions = [
        "VCU UNIT", "RING SET-PISTON", "BRAKE PAD SET", "OIL FILTER",
        "AIR FILTER", "SPARK PLUG", "CLUTCH PLATE", "RADIATOR ASSY",
        "ALTERNATOR", "STARTER MOTOR", "WATER PUMP", "TIMING BELT",
        "FUEL INJECTOR", "OXYGEN SENSOR", "CATALYTIC CONVERTER",
        "SHOCK ABSORBER", "STEERING RACK", "WHEEL BEARING", "CV JOINT",
        "HEADLAMP ASSY",
    ]

    # --- Sequential / string ID columns ---
    base_sno = chunk_index * chunk_size
    s_no = np.arange(base_sno + 1, base_sno + chunk_size + 1)

    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    vins = [
        "MAL" + "".join(rng.choice(chars, size=12))
        for _ in range(chunk_size)
    ]
    claim_nos = [f"CLM{base_sno + j + 1:010d}" for j in range(chunk_size)]
    acl_nos = [f"ACL{rng.integers(100000, 999999)}" for _ in range(chunk_size)]
    ro_nos = [f"RO{rng.integers(100000, 999999)}" for _ in range(chunk_size)]
    invoice_nos = [f"INV{rng.integers(100000, 999999)}" for _ in range(chunk_size)]
    causal_parts = [f"CP{rng.integers(10000, 99999)}" for _ in range(chunk_size)]
    main_ops = [f"OP{rng.integers(1000, 9999)}" for _ in range(chunk_size)]

    # --- Part_Desc ---
    part_desc = rng.choice(part_descriptions, size=chunk_size)

    # --- Numeric columns ---
    # Mileage: gamma(2, 10000) clipped to [0, 200000]
    mileage = np.clip(rng.gamma(2, 10000, size=chunk_size), 0, 200000).astype(int)

    # Part_Cost: 40% zeros, remainder lognormal(7, 1)
    part_cost = np.zeros(chunk_size, dtype=np.float64)
    nonzero_mask = rng.random(chunk_size) >= 0.40
    part_cost[nonzero_mask] = rng.lognormal(7, 1, size=int(nonzero_mask.sum()))

    # Labour: normal(500, 200) clipped to [100, 5000]
    labour = np.clip(rng.normal(500, 200, size=chunk_size), 100, 5000)

    # Sublet: 90% zeros, remainder uniform(100, 2000)
    sublet = np.zeros(chunk_size, dtype=np.float64)
    sublet_mask = rng.random(chunk_size) >= 0.90
    sublet[sublet_mask] = rng.uniform(100, 2000, size=int(sublet_mask.sum()))

    # --- Categorical columns ---
    claim_type = rng.choice(claim_types, size=chunk_size)
    part_type = rng.choice(part_types, size=chunk_size)
    cause = rng.choice(causes, size=chunk_size)
    nature = rng.choice(natures, size=chunk_size)
    status = rng.choice(statuses, size=chunk_size)
    dealership = rng.choice(dealerships, size=chunk_size)

    # --- Dates ---
    claim_dates, ro_dates, pdctn_dates = _generate_dates(chunk_size)

    # --- Build initial DataFrame (pre-tax) ---
    df = pl.DataFrame({
        "S_NO": s_no,
        "VIN": vins,
        "Claim_No": claim_nos,
        "ACL_No": acl_nos,
        "Claim_Date": claim_dates,
        "Claim_Type": claim_type.tolist(),
        "RO_No": ro_nos,
        "RO_Date": ro_dates,
        "Status": status.tolist(),
        "Mileage": mileage,
        "Cause": cause.tolist(),
        "Nature": nature.tolist(),
        "Causal_Part": causal_parts,
        "Main_OP": main_ops,
        "Part_Desc": part_desc.tolist(),
        "Part_Cost": part_cost,
        "Labour": labour,
        "Sublet": sublet,
        "Invoice_No": invoice_nos,
        "Part_Type": part_type.tolist(),
        "Pdctn_Date": pdctn_dates,
        "Dealership": dealership.tolist(),
    })

    # --- Taxes (IGST, CGST, SGST) ---
    df = _compute_taxes(df)

    # --- Total_Amt = Part_Cost + Labour + Sublet + IGST + CGST + SGST ---
    df = df.with_columns(
        (pl.col("Part_Cost") + pl.col("Labour") + pl.col("Sublet")
         + pl.col("IGST") + pl.col("CGST") + pl.col("SGST")).alias("Total_Amt")
    )

    # --- Approve_Amount_by_HMI: 90–100% of Total_Amt for normal records ---
    total_amt_np = df["Total_Amt"].to_numpy()
    approve_pct = rng.uniform(0.90, 1.00, size=chunk_size)
    approve_amt = total_amt_np * approve_pct
    df = df.with_columns(
        pl.Series("Approve_Amount_by_HMI", approve_amt, dtype=pl.Float64)
    )

    return df


def _apply_anomaly_labels(df: pl.DataFrame) -> pl.DataFrame:
    """flag records as anomalous based on five different fraud/error patterns.
    
    the approach:
    1. first check which records naturally trigger any of the 5 patterns
    2. if we're below the target anomaly rate (0.3-1%), inject more by
       tweaking clean records to match anomaly conditions
    3. if we're above 1%, randomly un-flag some to bring it back down
    """
    n = len(df)
    rng = np.random.default_rng()

    # polars sometimes keeps dates as Object type, need to cast them
    for col_name in ["Claim_Date", "RO_Date", "Pdctn_Date"]:
        if df.schema[col_name] == pl.Object:
            date_list = df[col_name].to_list()
            df = df.with_columns(
                pl.Series(col_name, date_list, dtype=pl.Datetime("us"))
            )

    # we want roughly 0.5% anomalies — enough to train on, not so many it's unrealistic
    target_min = 0.003
    target_max = 0.010
    target_mid = 0.005

    def _detect_anomalies(frame: pl.DataFrame) -> pl.DataFrame:
        """check all 5 anomaly patterns and set the Is_Anomaly flag."""
        pre_tax = pl.col("Part_Cost") + pl.col("Labour") + pl.col("Sublet")
        actual_tax = pl.col("IGST") + pl.col("CGST") + pl.col("SGST")

        # pattern 1: suspiciously expensive parts or high cost on barely-driven cars
        cost_flag = (pl.col("Part_Cost") > 35000) | (
            (pl.col("Mileage") < 1000) & (pl.col("Part_Cost") > 8000)
        )

        # pattern 3: tax doesn't add up — should be 18% but isn't
        # guard against zero pre_tax so we don't divide by zero
        safe_pre_tax = pl.when(pre_tax > 1e-6).then(pre_tax).otherwise(1e-6)
        tax_mismatch_flag = ((actual_tax - pre_tax * 0.18).abs() / safe_pre_tax) > 0.01

        # pattern 4: regular warranty claim on a car that's over 5 years old — suspicious
        vehicle_age_us = (pl.col("Claim_Date") - pl.col("Pdctn_Date")).dt.total_microseconds()
        us_per_day = 86400 * 1_000_000
        temporal_flag = (vehicle_age_us > 1825 * us_per_day) & (pl.col("Claim_Type") == "Regular")

        # pattern 5: claim got accepted but manufacturer only approved a tiny fraction
        approval_flag = (
            (pl.col("Approve_Amount_by_HMI") < pl.col("Total_Amt") * 0.50)
            & (pl.col("Status") == "Accept")
        )

        # combine everything except duplicate VIN (that one needs row-level logic)
        frame = frame.with_columns(
            (cost_flag | tax_mismatch_flag | temporal_flag | approval_flag)
            .cast(pl.Int8)
            .alias("Is_Anomaly")
        )

        # pattern 2: same VIN showing up too many times in a short window
        # this one can't be done with polars expressions, need to iterate
        sorted_df = frame.sort(["VIN", "Claim_Date"])
        vins = sorted_df["VIN"].to_list()
        # Convert Polars datetime to Python datetime for day arithmetic
        dates_raw = sorted_df["Claim_Date"].to_list()
        dates = [d if isinstance(d, datetime) else d for d in dates_raw]
        anomaly_arr = sorted_df["Is_Anomaly"].to_numpy().copy()

        # Group by VIN and check 30-day windows
        i = 0
        while i < len(vins):
            j = i
            while j < len(vins) and vins[j] == vins[i]:
                j += 1
            if j - i > 3:
                group_dates = dates[i:j]
                for k in range(len(group_dates)):
                    count = 0
                    for m in range(len(group_dates)):
                        diff = abs((group_dates[k] - group_dates[m]).total_seconds()) / 86400
                        if diff <= 30:
                            count += 1
                    if count > 3:
                        for m in range(len(group_dates)):
                            diff = abs((group_dates[k] - group_dates[m]).total_seconds()) / 86400
                            if diff <= 30:
                                anomaly_arr[i + m] = 1
            i = j

        frame = sorted_df.with_columns(
            pl.Series("Is_Anomaly", anomaly_arr, dtype=pl.Int8)
        )
        return frame

    # step 1: see what's naturally anomalous
    df = _detect_anomalies(df)
    natural_count = int(df["Is_Anomaly"].sum())
    natural_rate = natural_count / n if n > 0 else 0

    # step 2: if we don't have enough anomalies, inject some by modifying clean records
    if natural_rate < target_min:
        needed_total = int(n * target_mid)
        inject_count = needed_total - natural_count
        if inject_count > 0:
            # Get indices of clean (non-anomalous) rows
            clean_indices = df.filter(pl.col("Is_Anomaly") == 0).with_row_index("_idx")
            clean_idx_list = clean_indices["_idx"].to_list()

            if len(clean_idx_list) >= inject_count:
                chosen = rng.choice(clean_idx_list, size=inject_count, replace=False)
            else:
                chosen = np.array(clean_idx_list)

            # Distribute injections across the 5 patterns roughly evenly
            n_inject = len(chosen)
            # Split: ~25% cost, ~20% dup VIN, ~20% tax mismatch, ~20% temporal, ~15% approval
            splits = [
                int(n_inject * 0.25),
                int(n_inject * 0.20),
                int(n_inject * 0.20),
                int(n_inject * 0.20),
            ]
            splits.append(n_inject - sum(splits))  # remainder to approval

            cost_idx = chosen[:splits[0]]
            dup_vin_idx = chosen[splits[0]:splits[0] + splits[1]]
            tax_idx = chosen[splits[0] + splits[1]:splits[0] + splits[1] + splits[2]]
            temporal_idx = chosen[splits[0] + splits[1] + splits[2]:splits[0] + splits[1] + splits[2] + splits[3]]
            approval_idx = chosen[splits[0] + splits[1] + splits[2] + splits[3]:]

            # Convert to numpy arrays for mutation
            part_cost_arr = df["Part_Cost"].to_numpy().copy()
            mileage_arr = df["Mileage"].to_numpy().copy()
            vin_list = df["VIN"].to_list()
            claim_date_list = [d if isinstance(d, datetime) else d for d in df["Claim_Date"].to_list()]
            igst_arr = df["IGST"].to_numpy().copy()
            cgst_arr = df["CGST"].to_numpy().copy()
            sgst_arr = df["SGST"].to_numpy().copy()
            pdctn_date_list = [d if isinstance(d, datetime) else d for d in df["Pdctn_Date"].to_list()]
            claim_type_list = df["Claim_Type"].to_list()
            approve_arr = df["Approve_Amount_by_HMI"].to_numpy().copy()
            status_list = df["Status"].to_list()
            total_amt_arr = df["Total_Amt"].to_numpy().copy()
            labour_arr = df["Labour"].to_numpy().copy()
            sublet_arr = df["Sublet"].to_numpy().copy()

            # --- Pattern 1: Cost-based injection ---
            for idx in cost_idx:
                # Half get high Part_Cost, half get low-mileage + moderate Part_Cost
                if rng.random() < 0.5:
                    part_cost_arr[idx] = rng.uniform(36000, 60000)
                else:
                    mileage_arr[idx] = rng.integers(0, 999)
                    part_cost_arr[idx] = rng.uniform(8500, 15000)

            # --- Pattern 2: Duplicate VIN injection ---
            # Create small clusters of 4-6 repeated VINs with close dates
            dup_cluster_size = 5
            num_clusters = max(1, len(dup_vin_idx) // dup_cluster_size)
            dup_vin_used = 0
            for c in range(num_clusters):
                cluster_start = c * dup_cluster_size
                cluster_end = min(cluster_start + dup_cluster_size, len(dup_vin_idx))
                cluster_indices = dup_vin_idx[cluster_start:cluster_end]
                if len(cluster_indices) < 4:
                    # Need at least 4 for >3 in 30-day window; pad with cost anomaly
                    for idx in cluster_indices:
                        part_cost_arr[idx] = rng.uniform(36000, 60000)
                    continue
                # Pick a shared VIN and base date
                shared_vin = f"MAL{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'), size=12))}"
                base_date = claim_date_list[cluster_indices[0]]
                for k, idx in enumerate(cluster_indices):
                    vin_list[idx] = shared_vin
                    # Spread dates within 0–15 days of base (all within 30-day window)
                    claim_date_list[idx] = base_date + timedelta(days=int(rng.integers(0, 16)))
                dup_vin_used += len(cluster_indices)

            # --- Pattern 3: Tax mismatch injection ---
            for idx in tax_idx:
                pre_tax_val = part_cost_arr[idx] + labour_arr[idx] + sublet_arr[idx]
                # Inject a wrong tax: add 5-15% error to the expected 18%
                error_factor = rng.choice([-1, 1]) * rng.uniform(0.03, 0.10)
                wrong_rate = 0.18 + error_factor
                wrong_tax = pre_tax_val * wrong_rate
                # Assign all to IGST (inter-state style) for simplicity
                igst_arr[idx] = wrong_tax
                cgst_arr[idx] = 0.0
                sgst_arr[idx] = 0.0

            # --- Pattern 4: Temporal injection ---
            for idx in temporal_idx:
                # Set Pdctn_Date to > 5 years before Claim_Date and Claim_Type to Regular
                claim_dt = claim_date_list[idx]
                age_days = rng.integers(1826, 2500)
                pdctn_date_list[idx] = claim_dt - timedelta(days=int(age_days))
                claim_type_list[idx] = "Regular"

            # --- Pattern 5: Approval amount injection ---
            for idx in approval_idx:
                # Set Approve < 50% of Total_Amt and Status to Accept
                approve_arr[idx] = total_amt_arr[idx] * rng.uniform(0.10, 0.45)
                status_list[idx] = "Accept"

            # Recalculate Total_Amt for tax-modified rows
            for idx in tax_idx:
                total_amt_arr[idx] = part_cost_arr[idx] + labour_arr[idx] + sublet_arr[idx] + igst_arr[idx] + cgst_arr[idx] + sgst_arr[idx]
            # Also recalculate for cost-modified rows (Part_Cost changed)
            for idx in cost_idx:
                pre_tax = part_cost_arr[idx] + labour_arr[idx] + sublet_arr[idx]
                # Recompute tax at 18% (inter-state) for simplicity
                igst_arr[idx] = pre_tax * 0.18
                cgst_arr[idx] = 0.0
                sgst_arr[idx] = 0.0
                total_amt_arr[idx] = pre_tax + igst_arr[idx]
                approve_arr[idx] = total_amt_arr[idx] * rng.uniform(0.90, 1.00)

            # Rebuild the DataFrame with mutated arrays
            df = df.with_columns([
                pl.Series("Part_Cost", part_cost_arr, dtype=pl.Float64),
                pl.Series("Mileage", mileage_arr),
                pl.Series("VIN", vin_list),
                pl.Series("Claim_Date", claim_date_list, dtype=pl.Datetime("us")),
                pl.Series("IGST", igst_arr, dtype=pl.Float64),
                pl.Series("CGST", cgst_arr, dtype=pl.Float64),
                pl.Series("SGST", sgst_arr, dtype=pl.Float64),
                pl.Series("Pdctn_Date", pdctn_date_list, dtype=pl.Datetime("us")),
                pl.Series("Claim_Type", claim_type_list),
                pl.Series("Approve_Amount_by_HMI", approve_arr, dtype=pl.Float64),
                pl.Series("Status", status_list),
                pl.Series("Total_Amt", total_amt_arr, dtype=pl.Float64),
            ])

            # Re-detect anomalies after injection
            df = df.drop("Is_Anomaly")
            df = _detect_anomalies(df)

    # step 3: if we overshot, randomly un-flag some to bring the rate back down
    current_count = int(df["Is_Anomaly"].sum())
    current_rate = current_count / len(df) if len(df) > 0 else 0

    if current_rate > target_max:
        # Randomly flip some anomaly labels to 0 to bring rate down
        max_anomalies = int(len(df) * target_max)
        excess = current_count - max_anomalies
        if excess > 0:
            anomaly_indices = df.with_row_index("_idx").filter(pl.col("Is_Anomaly") == 1)["_idx"].to_list()
            flip_indices = set(rng.choice(anomaly_indices, size=excess, replace=False).tolist())
            anomaly_arr = df["Is_Anomaly"].to_numpy().copy()
            for idx in flip_indices:
                anomaly_arr[idx] = 0
            df = df.with_columns(pl.Series("Is_Anomaly", anomaly_arr, dtype=pl.Int8))

    return df


def generate_big_data(total_records=10_000_000, chunk_size=1_000_000):
    """generate the full synthetic dataset as parquet chunks.
    each chunk gets all 28 columns (26 from the real schema + dealership + anomaly label).
    default is 10M rows in 1M-row chunks — takes about 5 min on a decent machine."""
    if total_records <= 0:
        raise ValueError(f"total_records must be > 0, got {total_records}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    total_chunks = (total_records + chunk_size - 1) // chunk_size

    for i in range(total_chunks):
        # Last chunk may be smaller than chunk_size
        current_chunk_size = min(chunk_size, total_records - i * chunk_size)

        chunk = _generate_chunk(current_chunk_size, i)
        chunk = _apply_anomaly_labels(chunk)
        chunk.write_parquet(f"claims_batch_{i}.parquet")
        print(f"✅ Chunk {i + 1}/{total_chunks} generated.")

# generate_big_data()
