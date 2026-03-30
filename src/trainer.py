import json

import polars as pl
import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score
import joblib

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# columns we need to turn into integers for the model
CATEGORICAL_COLS = ["Claim_Type", "Part_Type", "Cause", "Nature", "Status", "Dealership"]

# all 19 features the model sees — order matters
FEATURE_COLS = [
    # raw numerics straight from the claim data
    "Mileage", "Part_Cost", "Labour", "Sublet",
    "Total_Amt", "IGST", "CGST", "SGST",
    "Approve_Amount_by_HMI",
    # stuff we compute from dates and ratios
    "Vehicle_Age_Days", "Claim_RO_Gap_Days", "Tax_Rate", "Approval_Ratio",
    # integer-encoded categoricals
    "Claim_Type_idx", "Part_Type_idx", "Cause_idx", "Nature_idx",
    "Status_idx", "Dealership_idx",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _encode_categoricals(df: pl.DataFrame) -> pl.DataFrame:
    """turn each categorical column into sorted integer indices.
    we save the mapping to json so the api server can use the same encoding."""

    categorical_mappings: dict[str, dict[str, int]] = {}

    for col in CATEGORICAL_COLS:
        unique_vals = sorted(df[col].unique().to_list())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        categorical_mappings[col] = mapping

        idx_col = col + "_idx"
        df = df.with_columns(
            pl.col(col).replace_strict(mapping).cast(pl.Int64).alias(idx_col)
        )

    with open("categorical_mappings.json", "w") as f:
        json.dump(categorical_mappings, f, indent=2)

    return df


def _engineer_features(df: pl.DataFrame) -> pl.DataFrame:
    """compute the derived features we need for the model.
    uses safe division to avoid blowing up on zero denominators."""

    # how old is the car, and how long between the repair order and the claim?
    # polars gives us microseconds so we convert to days
    us_per_day = 86400 * 1_000_000

    df = df.with_columns([
        ((pl.col("Claim_Date") - pl.col("Pdctn_Date")).dt.total_microseconds() / us_per_day)
        .alias("Vehicle_Age_Days"),
        ((pl.col("Claim_Date") - pl.col("RO_Date")).dt.total_microseconds() / us_per_day)
        .alias("Claim_RO_Gap_Days"),
    ])

    # what percentage of the claim is tax? if there's nothing billable, just call it zero
    pre_tax = pl.col("Part_Cost") + pl.col("Labour") + pl.col("Sublet")
    safe_pre_tax = pl.when(pre_tax > 1e-6).then(pre_tax).otherwise(pl.lit(1e-6))
    total_tax = pl.col("IGST") + pl.col("CGST") + pl.col("SGST")

    df = df.with_columns(
        (total_tax / safe_pre_tax).alias("Tax_Rate")
    )

    # how much of the total did the manufacturer actually approve?
    safe_total = pl.when(pl.col("Total_Amt") > 1e-6).then(pl.col("Total_Amt")).otherwise(pl.lit(1e-6))

    df = df.with_columns(
        (pl.col("Approve_Amount_by_HMI") / safe_total).alias("Approval_Ratio")
    )

    return df


def _log_feature_importance(model: lgb.Booster) -> None:
    """print which features the model cares about most, ranked by gain.
    helps the dealership team understand what's driving the anomaly flags."""
    importance = model.feature_importance(importance_type="gain")
    feature_names = model.feature_name()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance_gain": importance}
    ).sort_values("importance_gain", ascending=False)

    max_gain = importance_df["importance_gain"].max()
    print("\n📊 Feature Importance (by gain):")
    print("=" * 60)
    for _, row in importance_df.iterrows():
        bar_len = int(row["importance_gain"] / max_gain * 30) if max_gain > 0 else 0
        bar = "█" * bar_len
        print(f"  {row['feature']:25s} {row['importance_gain']:12.1f}  {bar}")


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def run_training(data_glob: str = "claims_batch_*.parquet") -> None:
    """main training pipeline — loads the parquet data, builds features,
    runs 5-fold cv, trains the final model with early stopping, and saves it."""
    import glob

    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(
            f"No Parquet files found matching '{data_glob}'. "
            "Run data_engine.generate_big_data() first."
        )

    # Lazy-scan all matching parquet files and collect
    df = pl.scan_parquet(data_glob).collect()

    # --- Feature engineering pipeline ---
    df = _encode_categoricals(df)
    df = _engineer_features(df)

    # --- Prepare arrays ---
    X = df.select(FEATURE_COLS).to_pandas()
    y = df.select("Is_Anomaly").to_pandas().values.ravel()

    # need at least one of each class or the model can't learn anything
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    if pos_count == 0 or neg_count == 0:
        raise ValueError(
            "Labels have no class variation — cannot train. "
            f"pos={pos_count}, neg={neg_count}"
        )

    # Train / test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # weight the minority class so the model doesn't just predict "normal" for everything
    train_pos = int(y_train.sum())
    train_neg = len(y_train) - train_pos
    scale_pos_weight = train_neg / train_pos
    print(f"⚖️  scale_pos_weight: {scale_pos_weight:.2f} (neg={train_neg}, pos={train_pos})")

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "scale_pos_weight": scale_pos_weight,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "device": "cpu",
        "is_unbalance": False,
        "n_jobs": -1,
    }

    # --- 5-fold stratified cross-validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores: list[float] = []

    print("\n📊 5-Fold Stratified Cross-Validation:")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        fold_train = lgb.Dataset(X_train.iloc[train_idx], label=y_train[train_idx])
        fold_val = lgb.Dataset(
            X_train.iloc[val_idx], label=y_train[val_idx], reference=fold_train
        )

        fold_model = lgb.train(
            params,
            fold_train,
            num_boost_round=1000,
            valid_sets=[fold_val],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        y_val_pred = fold_model.predict(X_train.iloc[val_idx])
        fold_score = average_precision_score(y_train[val_idx], y_val_pred)
        cv_scores.append(fold_score)
        print(f"  Fold {fold + 1}/5 PR-AUC: {fold_score:.4f}")

    print(
        f"📊 Mean CV PR-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
    )

    # --- Final training with early stopping (90/10 split) ---
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    print("\n🚀 Training final model with early stopping (patience=50, max 1000 rounds)...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # --- Feature importance by gain ---
    _log_feature_importance(model)

    # --- Evaluate on held-out test set ---
    y_proba = model.predict(X_test)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"\n🎯 Test PR-AUC Score: {pr_auc:.4f}")

    # --- Save model ---
    joblib.dump(model, "warranty_model_v1.pkl")
    print("💾 Model saved as warranty_model_v1.pkl")


# run_training()
