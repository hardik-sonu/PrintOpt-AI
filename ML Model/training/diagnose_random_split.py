from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


# ============================================================
# PRINTOPT AI — STEP 7B
# RANDOM SPLIT DIAGNOSTIC
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 7B")
print("RANDOM SPLIT DIAGNOSTIC")
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

df = pd.read_csv(DATA_PATH)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMNS]


print()
print("DATA")
print("-" * 70)
print(f"Total records : {len(df)}")
print(f"Features      : {len(FEATURE_COLUMNS)}")
print(f"Targets       : {len(TARGET_COLUMNS)}")


# ------------------------------------------------------------
# RANDOM SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


print()
print("RANDOM SPLIT")
print("-" * 70)
print(f"Training records : {len(X_train)}")
print(f"Testing records  : {len(X_test)}")


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

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

model = MultiOutputRegressor(base_model)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

print()
print("TRAINING")
print("-" * 70)

model.fit(
    X_train,
    y_train
)

print("XGBoost training complete.")


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

print()
print("MODEL PERFORMANCE")
print("-" * 70)

results = []

for i, target in enumerate(TARGET_COLUMNS):

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

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    results.append({
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
# SUMMARY
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# VED RANGE COMPARISON
# ------------------------------------------------------------

print()
print("=" * 70)
print("VED RANGE COMPARISON")
print("=" * 70)

print(
    f"Training VED : "
    f"{X_train['VED_J_mm3'].min():.3f} → "
    f"{X_train['VED_J_mm3'].max():.3f}"
)

print(
    f"Testing VED  : "
    f"{X_test['VED_J_mm3'].min():.3f} → "
    f"{X_test['VED_J_mm3'].max():.3f}"
)

outside = (
    (X_test["VED_J_mm3"] < X_train["VED_J_mm3"].min())
    |
    (X_test["VED_J_mm3"] > X_train["VED_J_mm3"].max())
)

print(
    f"Test samples outside training VED range : "
    f"{outside.sum()}"
)


print()
print("STEP 7B COMPLETE")
print("=" * 70)