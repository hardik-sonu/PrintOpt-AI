from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PRINTOPT AI — STEP 7A
# TEST PREDICTION DIAGNOSTICS
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 7A")
print("TEST PREDICTION DIAGNOSTICS")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

RF_MODEL_PATH = BASE_DIR / "models" / "random_forest_baseline.joblib"
XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_baseline.joblib"

OUTPUT_DIR = BASE_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "test_predictions.csv"


# ------------------------------------------------------------
# FEATURES AND TARGETS
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

test_df = pd.read_csv(TEST_PATH)

X_test = test_df[FEATURE_COLUMNS]

y_test = test_df[TARGET_COLUMNS]


# ------------------------------------------------------------
# LOAD MODELS
# ------------------------------------------------------------

rf_model = joblib.load(RF_MODEL_PATH)
xgb_model = joblib.load(XGB_MODEL_PATH)


# ------------------------------------------------------------
# PREDICTIONS
# ------------------------------------------------------------

rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)


# ------------------------------------------------------------
# BUILD RESULTS TABLE
# ------------------------------------------------------------

results = test_df[
    [
        "Experiment_ID",
        "Source_ID",
        "Laser_Power_W",
        "Scanning_Speed_mm_s",
        "Hatch_Distance_um",
        "Layer_Thickness_um",
        "VED_J_mm3",
        "UTS_MPa",
        "YS_MPa",
        "Elongation_pct",
    ]
].copy()


# ------------------------------------------------------------
# RANDOM FOREST PREDICTIONS
# ------------------------------------------------------------

results["RF_UTS_Pred"] = rf_predictions[:, 0]
results["RF_YS_Pred"] = rf_predictions[:, 1]
results["RF_Elongation_Pred"] = rf_predictions[:, 2]


# ------------------------------------------------------------
# XGBOOST PREDICTIONS
# ------------------------------------------------------------

results["XGB_UTS_Pred"] = xgb_predictions[:, 0]
results["XGB_YS_Pred"] = xgb_predictions[:, 1]
results["XGB_Elongation_Pred"] = xgb_predictions[:, 2]


# ------------------------------------------------------------
# ABSOLUTE ERRORS
# ------------------------------------------------------------

results["XGB_UTS_Error"] = (
    results["XGB_UTS_Pred"] - results["UTS_MPa"]
).abs()

results["XGB_YS_Error"] = (
    results["XGB_YS_Pred"] - results["YS_MPa"]
).abs()

results["XGB_Elongation_Error"] = (
    results["XGB_Elongation_Pred"] - results["Elongation_pct"]
).abs()


# ------------------------------------------------------------
# SORT BY UTS ERROR
# ------------------------------------------------------------

results = results.sort_values(
    "XGB_UTS_Error",
    ascending=False
)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:.3f}".format)


print()
print("TEST SET PREDICTIONS")
print("-" * 70)

print(results.to_string(index=False))


# ------------------------------------------------------------
# VED ANALYSIS
# ------------------------------------------------------------

print()
print("=" * 70)
print("VED ANALYSIS")
print("=" * 70)

train_df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "train.csv"
)

train_ved_min = train_df["VED_J_mm3"].min()
train_ved_max = train_df["VED_J_mm3"].max()

results["VED_Outside_Training_Range"] = (
    (results["VED_J_mm3"] < train_ved_min)
    |
    (results["VED_J_mm3"] > train_ved_max)
)

outside_count = results[
    "VED_Outside_Training_Range"
].sum()

print(f"Training VED minimum : {train_ved_min:.3f}")
print(f"Training VED maximum : {train_ved_max:.3f}")
print(f"Test samples outside training VED range : {outside_count}")
print(f"Total test samples : {len(results)}")


# ------------------------------------------------------------
# SOURCE SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("TEST SOURCE SUMMARY")
print("=" * 70)

source_summary = (
    results
    .groupby("Source_ID")
    .agg(
        Samples=("Experiment_ID", "count"),
        Mean_VED=("VED_J_mm3", "mean"),
        Mean_XGB_UTS_Error=("XGB_UTS_Error", "mean"),
        Mean_XGB_YS_Error=("XGB_YS_Error", "mean"),
        Mean_XGB_Elongation_Error=("XGB_Elongation_Error", "mean"),
    )
    .sort_values(
        "Mean_XGB_UTS_Error",
        ascending=False
    )
)

print(
    source_summary.to_string()
)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print()
print("STEP 7A COMPLETE")
print("=" * 70)