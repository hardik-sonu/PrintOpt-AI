from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


# ============================================================
# PRINTOPT AI — STEP 8A
# SOURCE-AWARE CROSS-VALIDATION
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 8A")
print("SOURCE-AWARE CROSS-VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ti64_lpbf_processed.csv"
)


# ------------------------------------------------------------
# FEATURES / TARGETS
# ------------------------------------------------------------

FEATURE_COLUMNS = [
    "Powder_Size_um",
    "Laser_Spot_um",
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Hatch_Distance_um",
    "Layer_Thickness_um",
    "VED_J_mm3",
]

TARGET_COLUMNS = [
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
]


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMNS]
groups = df["Source_ID"]


print()
print("DATA")
print("-" * 70)

print(f"Total records : {len(df)}")
print(f"Total sources : {groups.nunique()}")
print(f"Features      : {len(FEATURE_COLUMNS)}")
print(f"Targets       : {len(TARGET_COLUMNS)}")


# ------------------------------------------------------------
# GROUP K-FOLD
# ------------------------------------------------------------

N_SPLITS = 5

cv = GroupKFold(
    n_splits=N_SPLITS
)


# ------------------------------------------------------------
# STORAGE
# ------------------------------------------------------------

fold_results = []


# ------------------------------------------------------------
# CROSS-VALIDATION
# ------------------------------------------------------------

print()
print("=" * 70)
print("CROSS-VALIDATION")
print("=" * 70)


for fold, (train_idx, test_idx) in enumerate(
    cv.split(
        X,
        y,
        groups=groups
    ),
    start=1,
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_sources = groups.iloc[train_idx].nunique()
    test_sources = groups.iloc[test_idx].nunique()

    print()
    print(f"FOLD {fold}")
    print("-" * 70)

    print(f"Training records : {len(train_idx)}")
    print(f"Testing records  : {len(test_idx)}")
    print(f"Training sources : {train_sources}")
    print(f"Testing sources  : {test_sources}")


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    base_model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(
        base_model
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    for i, target in enumerate(
        TARGET_COLUMNS
    ):

        actual = y_test.iloc[:, i]
        predicted = y_pred[:, i]

        r2 = r2_score(
            actual,
            predicted
        )

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        fold_results.append({
            "Fold": fold,
            "Target": target,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
        })

        print()
        print(target)
        print(f"  R²   : {r2:.4f}")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")


# ------------------------------------------------------------
# RESULTS DATAFRAME
# ------------------------------------------------------------

results_df = pd.DataFrame(
    fold_results
)


# ------------------------------------------------------------
# OVERALL RESULTS
# ------------------------------------------------------------

print()
print("=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)


summary = (
    results_df
    .groupby("Target")
    .agg(
        R2_Mean=("R2", "mean"),
        R2_Std=("R2", "std"),
        MAE_Mean=("MAE", "mean"),
        MAE_Std=("MAE", "std"),
        RMSE_Mean=("RMSE", "mean"),
        RMSE_Std=("RMSE", "std"),
    )
)


print(
    summary.to_string()
)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

OUTPUT_DIR = BASE_DIR / "evaluation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "source_cross_validation.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print()
print("STEP 8A COMPLETE")
print("=" * 70)