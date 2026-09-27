from pathlib import Path

import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PRINTOPT AI — STEP 6B
# XGBOOST BASELINE MODEL
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 6B")
print("XGBOOST BASELINE MODEL")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgboost_baseline.joblib"


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

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df[FEATURE_COLUMNS]
X_test = test_df[FEATURE_COLUMNS]

y_train = train_df[TARGET_COLUMNS]
y_test = test_df[TARGET_COLUMNS]


print()
print("DATA")
print("-" * 70)
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"Features         : {len(FEATURE_COLUMNS)}")
print(f"Targets          : {len(TARGET_COLUMNS)}")


# ------------------------------------------------------------
# CREATE XGBOOST MODEL
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

model.fit(X_train, y_train)

print("XGBoost training complete.")


# ------------------------------------------------------------
# PREDICTION
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

    r2 = r2_score(actual, predicted)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = mse ** 0.5

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

print(results_df.to_string(index=False))


# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

print()
print("FEATURE IMPORTANCE")
print("-" * 70)

importance_values = []

for estimator in model.estimators_:

    importance_values.append(
        estimator.feature_importances_
    )

importance_df = pd.DataFrame(
    importance_values,
    columns=FEATURE_COLUMNS,
    index=TARGET_COLUMNS,
)

print(importance_df.round(4).to_string())


# ------------------------------------------------------------
# AVERAGE FEATURE IMPORTANCE
# ------------------------------------------------------------

average_importance = importance_df.mean(axis=0)

average_importance_df = pd.DataFrame({
    "Feature": FEATURE_COLUMNS,
    "Average_Importance": average_importance.values,
}).sort_values(
    "Average_Importance",
    ascending=False
)

print()
print("AVERAGE FEATURE IMPORTANCE")
print("-" * 70)

print(
    average_importance_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------

joblib.dump(model, MODEL_PATH)

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_PATH)

print()
print("STEP 6B COMPLETE")
print("=" * 70)