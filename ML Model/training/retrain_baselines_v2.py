from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor


# ============================================================
# PRINTOPT AI — STEP 9F
# RETRAIN BASELINE MODELS V2
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9F")
print("RETRAIN BASELINE MODELS V2")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_v2.csv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_v2.csv"
)

MODEL_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"

RF_MODEL_FILE = (
    MODEL_DIR / "random_forest_v2.joblib"
)

XGB_MODEL_FILE = (
    MODEL_DIR / "xgboost_v2.joblib"
)

MLP_MODEL_FILE = (
    MODEL_DIR / "mlp_v2.joblib"
)

RESULTS_FILE = (
    EVALUATION_DIR
    / "baseline_results_v2.csv"
)

PREDICTIONS_FILE = (
    EVALUATION_DIR
    / "baseline_predictions_v2.csv"
)

FEATURE_IMPORTANCE_FILE = (
    EVALUATION_DIR
    / "baseline_feature_importance_v2.csv"
)


# ------------------------------------------------------------
# FEATURES AND TARGETS
# ------------------------------------------------------------

FEATURES = [
    "Powder_Size_um",
    "Laser_Spot_um",
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Hatch_Distance_um",
    "Layer_Thickness_um",
    "VED_J_mm3",
]

TARGETS = [
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
]


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print()
print("DATA")
print("-" * 70)

print(
    f"Training records : {len(train_df)}"
)

print(
    f"Testing records  : {len(test_df)}"
)

print(
    f"Training sources : "
    f"{train_df['Source_ID'].nunique()}"
)

print(
    f"Testing sources  : "
    f"{test_df['Source_ID'].nunique()}"
)


# ------------------------------------------------------------
# SOURCE LEAKAGE CHECK
# ------------------------------------------------------------

train_sources = set(
    train_df["Source_ID"]
)

test_sources = set(
    test_df["Source_ID"]
)

source_overlap = train_sources & test_sources

if source_overlap:
    raise ValueError(
        f"Source leakage detected: {source_overlap}"
    )


# ------------------------------------------------------------
# PREPARE X / Y
# ------------------------------------------------------------

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_train = train_df[TARGETS]
y_test = test_df[TARGETS]


# ------------------------------------------------------------
# HELPER
# ------------------------------------------------------------

def calculate_metrics(
    model_name,
    predictions
):
    rows = []

    for i, target in enumerate(TARGETS):

        actual = y_test[target].to_numpy()
        predicted = predictions[:, i]

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

        rows.append(
            {
                "Model": model_name,
                "Target": target,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

    return rows


# ============================================================
# MODEL 1 — RANDOM FOREST
# ============================================================

print()
print("=" * 70)
print("MODEL 1 — RANDOM FOREST")
print("=" * 70)

rf = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
)

rf.fit(
    X_train,
    y_train
)

rf_predictions = rf.predict(X_test)

rf_results = calculate_metrics(
    "Random Forest V2",
    rf_predictions
)

print()

for row in rf_results:
    print(
        f"{row['Target']:20s} "
        f"R²={row['R2']:.4f}  "
        f"MAE={row['MAE']:.4f}  "
        f"RMSE={row['RMSE']:.4f}"
    )


# ============================================================
# MODEL 2 — XGBOOST
# ============================================================

print()
print("=" * 70)
print("MODEL 2 — XGBOOST")
print("=" * 70)

xgb_base = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

xgb = MultiOutputRegressor(
    xgb_base
)

xgb.fit(
    X_train,
    y_train
)

xgb_predictions = xgb.predict(
    X_test
)

xgb_results = calculate_metrics(
    "XGBoost V2",
    xgb_predictions
)

print()

for row in xgb_results:
    print(
        f"{row['Target']:20s} "
        f"R²={row['R2']:.4f}  "
        f"MAE={row['MAE']:.4f}  "
        f"RMSE={row['RMSE']:.4f}"
    )


# ============================================================
# MODEL 3 — MLP
# ============================================================

print()
print("=" * 70)
print("MODEL 3 — MLP NEURAL NETWORK")
print("=" * 70)

mlp = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=100,
                random_state=42,
            ),
        ),
    ]
)

mlp.fit(
    X_train,
    y_train
)

mlp_predictions = mlp.predict(
    X_test
)

mlp_results = calculate_metrics(
    "MLP V2",
    mlp_predictions
)

print()

for row in mlp_results:
    print(
        f"{row['Target']:20s} "
        f"R²={row['R2']:.4f}  "
        f"MAE={row['MAE']:.4f}  "
        f"RMSE={row['RMSE']:.4f}"
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

all_results = (
    rf_results
    + xgb_results
    + mlp_results
)

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# PREDICTIONS TABLE
# ============================================================

predictions_df = test_df[
    [
        "Experiment_ID",
        "Source_ID",
        *FEATURES,
        *TARGETS,
    ]
].copy()

for i, target in enumerate(TARGETS):

    predictions_df[
        f"RF_{target}_Pred"
    ] = rf_predictions[:, i]

    predictions_df[
        f"XGB_{target}_Pred"
    ] = xgb_predictions[:, i]

    predictions_df[
        f"MLP_{target}_Pred"
    ] = mlp_predictions[:, i]


# ------------------------------------------------------------
# ABSOLUTE ERRORS
# ------------------------------------------------------------

for target in TARGETS:

    actual = test_df[target].to_numpy()

    predictions_df[
        f"XGB_{target}_AbsError"
    ] = np.abs(
        actual
        - xgb_predictions[
            :,
            TARGETS.index(target)
        ]
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_rows = []


# Random Forest importance
for target_index, target in enumerate(TARGETS):

    estimator = rf.estimators_[target_index]

    for feature, importance in zip(
        FEATURES,
        estimator.feature_importances_,
    ):
        importance_rows.append(
            {
                "Model": "Random Forest V2",
                "Target": target,
                "Feature": feature,
                "Importance": importance,
            }
        )


# XGBoost importance
for target_index, target in enumerate(TARGETS):

    estimator = xgb.estimators_[target_index]

    for feature, importance in zip(
        FEATURES,
        estimator.feature_importances_,
    ):
        importance_rows.append(
            {
                "Model": "XGBoost V2",
                "Target": target,
                "Feature": feature,
                "Importance": importance,
            }
        )


importance_df = pd.DataFrame(
    importance_rows
)


# ============================================================
# SAVE MODELS
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EVALUATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    rf,
    RF_MODEL_FILE
)

joblib.dump(
    xgb,
    XGB_MODEL_FILE
)

joblib.dump(
    mlp,
    MLP_MODEL_FILE
)


# ============================================================
# SAVE EVALUATION FILES
# ============================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

importance_df.to_csv(
    FEATURE_IMPORTANCE_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("BASELINE COMPARISON — V2")
print("=" * 70)

for model in [
    "Random Forest V2",
    "XGBoost V2",
    "MLP V2",
]:

    model_results = results_df[
        results_df["Model"] == model
    ]

    print()
    print(model)

    for _, row in model_results.iterrows():

        print(
            f"  {row['Target']:20s} "
            f"R²={row['R2']:.4f}  "
            f"MAE={row['MAE']:.4f}  "
            f"RMSE={row['RMSE']:.4f}"
        )


# ============================================================
# MLP TRAINING INFO
# ============================================================

try:

    mlp_model = mlp.named_steps["model"]

    print()
    print("MLP TRAINING")
    print("-" * 70)

    print(
        f"Iterations : "
        f"{mlp_model.n_iter_}"
    )

    print(
        f"Final loss : "
        f"{mlp_model.loss_:.6f}"
    )

except Exception:
    pass


# ============================================================
# FILES CREATED
# ============================================================

print()
print("FILES CREATED")
print("-" * 70)

print(
    f"Random Forest model:"
)

print(
    RF_MODEL_FILE
)

print()

print(
    f"XGBoost model:"
)

print(
    XGB_MODEL_FILE
)

print()

print(
    f"MLP model:"
)

print(
    MLP_MODEL_FILE
)

print()

print(
    f"Evaluation results:"
)

print(
    RESULTS_FILE
)

print()

print(
    f"Predictions:"
)

print(
    PREDICTIONS_FILE
)

print()

print(
    f"Feature importance:"
)

print(
    FEATURE_IMPORTANCE_FILE
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("FINAL VALIDATION")
print("-" * 70)

print(
    f"Training records : {len(train_df)}"
)

print(
    f"Testing records  : {len(test_df)}"
)

print(
    f"Source overlap   : {source_overlap}"
)

print(
    f"Results rows     : {len(results_df)}"
)

print(
    f"Prediction rows  : {len(predictions_df)}"
)


if len(results_df) != 9:
    raise ValueError(
        "Expected 9 evaluation rows "
        "(3 models × 3 targets)."
    )

if len(predictions_df) != len(test_df):
    raise ValueError(
        "Prediction row count does not "
        "match test dataset."
    )

if source_overlap:
    raise ValueError(
        "Source leakage detected."
    )


print()
print("IMPORTANT")
print("-" * 70)

print(
    "The original baseline models were NOT overwritten."
)

print(
    "The new models were trained only on "
    "train_v2.csv."
)

print(
    "Evaluation was performed only on "
    "test_v2.csv."
)

print(
    "The test sources were completely unseen "
    "during training."
)

print()
print("STEP 9F COMPLETE")
print("=" * 70)