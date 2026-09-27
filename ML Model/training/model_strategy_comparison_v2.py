from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


# ============================================================
# PRINTOPT AI — STEP 9L
# MODEL STRATEGY COMPARISON V2
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"
MODEL_DIR = BASE_DIR / "models"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_v2.csv"
TEST_FILE = DATA_DIR / "test_v2.csv"


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


# ============================================================
# FEATURES
# ============================================================

BASE_FEATURES = [
    "Powder_Size_um",
    "Laser_Spot_um",
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Hatch_Distance_um",
    "Layer_Thickness_um",
    "VED_J_mm3",
]


# Selective engineered features:
#
# Step 9J showed that all 6 engineered features together
# degraded UTS and YS performance.
#
# We therefore test only features with some physically
# interpretable signal and avoid the highly redundant
# Layer_to_Spot_Ratio and Layer_to_Powder_Ratio.
#
# These are still experiments, not assumed improvements.

SELECTIVE_ENGINEERED_FEATURES = [
    "Linear_Energy_Density_J_mm",
    "Hatch_to_Spot_Ratio",
    "Spot_to_Powder_Ratio",
    "Hatch_to_Powder_Ratio",
]


TARGETS = [
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df):
    df = df.copy()

    df["Linear_Energy_Density_J_mm"] = (
        df["Laser_Power_W"]
        / df["Scanning_Speed_mm_s"]
    )

    df["Hatch_to_Spot_Ratio"] = (
        df["Hatch_Distance_um"]
        / df["Laser_Spot_um"]
    )

    df["Spot_to_Powder_Ratio"] = (
        df["Laser_Spot_um"]
        / df["Powder_Size_um"]
    )

    df["Hatch_to_Powder_Ratio"] = (
        df["Hatch_Distance_um"]
        / df["Powder_Size_um"]
    )

    return df


train_df = add_engineered_features(train_df)
test_df = add_engineered_features(test_df)


# ============================================================
# VALIDATION
# ============================================================

for feature in BASE_FEATURES + SELECTIVE_ENGINEERED_FEATURES:

    if not np.isfinite(train_df[feature]).all():
        raise ValueError(
            f"Non-finite values found in training feature: {feature}"
        )

    if not np.isfinite(test_df[feature]).all():
        raise ValueError(
            f"Non-finite values found in testing feature: {feature}"
        )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9L")
print("MODEL STRATEGY COMPARISON V2")
print("=" * 70)

print("\nDATA")
print("-" * 70)
print(f"Training records : {len(train_df)}")
print(f"Testing records  : {len(test_df)}")
print(
    f"Training sources : "
    f"{train_df['Source_ID'].nunique()}"
)
print(
    f"Testing sources  : "
    f"{test_df['Source_ID'].nunique()}"
)

print("\nHeld-out test sources:")
print(", ".join(sorted(test_df["Source_ID"].unique())))


# ============================================================
# MODEL FACTORIES
# ============================================================

def make_random_forest():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def make_xgboost():
    return XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )


def make_knn():
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=7,
                    weights="distance",
                    p=2,
                ),
            ),
        ]
    )


# ============================================================
# STRATEGIES
# ============================================================

strategies = {
    "BASELINE_7": BASE_FEATURES,
    "SELECTIVE_ENGINEERED": (
        BASE_FEATURES
        + SELECTIVE_ENGINEERED_FEATURES
    ),
}


models = {
    "Random Forest": make_random_forest,
    "XGBoost": make_xgboost,
    "KNN": make_knn,
}


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_predictions(y_true, y_pred):

    rows = []

    for i, target in enumerate(TARGETS):

        actual = y_true[:, i]
        predicted = y_pred[:, i]

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

        r2 = r2_score(
            actual,
            predicted
        )

        rows.append(
            {
                "Target": target,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

    return rows


# ============================================================
# RUN STRATEGIES
# ============================================================

all_results = []
all_predictions = []


for strategy_name, features in strategies.items():

    print("\n" + "=" * 70)
    print(f"STRATEGY: {strategy_name}")
    print("=" * 70)

    print("\nFeatures:")

    for feature in features:
        print(f"  - {feature}")

    X_train = train_df[features].values
    X_test = test_df[features].values

    y_train = train_df[TARGETS].values
    y_test = test_df[TARGETS].values

    for model_name, model_factory in models.items():

        print("\n" + "-" * 70)
        print(f"{model_name}")

        model = model_factory()

        # ----------------------------------------------------
        # Fit one model for all targets
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(X_test)

        metric_rows = evaluate_predictions(
            y_test,
            predictions
        )

        for row in metric_rows:

            result = {
                "Strategy": strategy_name,
                "Model": model_name,
                **row,
            }

            all_results.append(result)

            print(
                f"  {row['Target']:<18} "
                f"R²={row['R2']:>9.4f} "
                f"MAE={row['MAE']:>9.4f} "
                f"RMSE={row['RMSE']:>9.4f}"
            )

        # ----------------------------------------------------
        # Save predictions
        # ----------------------------------------------------

        for i, experiment_id in enumerate(
            test_df["Experiment_ID"]
        ):

            prediction_row = {
                "Experiment_ID": experiment_id,
                "Source_ID": test_df.iloc[i]["Source_ID"],
                "Strategy": strategy_name,
                "Model": model_name,
            }

            for target_index, target in enumerate(
                TARGETS
            ):
                prediction_row[
                    f"Actual_{target}"
                ] = y_test[i, target_index]

                prediction_row[
                    f"Predicted_{target}"
                ] = predictions[
                    i,
                    target_index
                ]

            all_predictions.append(
                prediction_row
            )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        safe_strategy = (
            strategy_name.lower()
        )

        safe_model = (
            model_name.lower()
            .replace(" ", "_")
        )

        model_path = (
            MODEL_DIR
            / f"step9l_{safe_strategy}_{safe_model}.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "features": features,
                "targets": TARGETS,
                "training_sources": sorted(
                    train_df["Source_ID"].unique()
                ),
                "test_sources": sorted(
                    test_df["Source_ID"].unique()
                ),
            },
            model_path,
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)


# ============================================================
# STRATEGY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("STRATEGY COMPARISON")
print("=" * 70)

comparison_rows = []

for model_name in models.keys():

    for target in TARGETS:

        subset = results_df[
            (results_df["Model"] == model_name)
            & (results_df["Target"] == target)
        ].copy()

        baseline = subset[
            subset["Strategy"] == "BASELINE_7"
        ].iloc[0]

        selective = subset[
            subset["Strategy"]
            == "SELECTIVE_ENGINEERED"
        ].iloc[0]

        comparison_rows.append(
            {
                "Model": model_name,
                "Target": target,

                "Baseline_R2": baseline["R2"],
                "Selective_R2": selective["R2"],
                "R2_Change": (
                    selective["R2"]
                    - baseline["R2"]
                ),

                "Baseline_MAE": baseline["MAE"],
                "Selective_MAE": selective["MAE"],
                "MAE_Change": (
                    selective["MAE"]
                    - baseline["MAE"]
                ),

                "Baseline_RMSE": baseline["RMSE"],
                "Selective_RMSE": selective["RMSE"],
                "RMSE_Change": (
                    selective["RMSE"]
                    - baseline["RMSE"]
                ),
            }
        )

comparison_df = pd.DataFrame(
    comparison_rows
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL BY TARGET
# ============================================================

print("\n" + "=" * 70)
print("LOWEST-MAE MODEL BY TARGET")
print("=" * 70)

for target in TARGETS:

    subset = results_df[
        results_df["Target"] == target
    ].copy()

    best_index = subset["MAE"].idxmin()

    best = subset.loc[best_index]

    print(
        f"\n{target}"
    )

    print(
        f"  Model    : {best['Model']}"
    )

    print(
        f"  Strategy : {best['Strategy']}"
    )

    print(
        f"  MAE      : {best['MAE']:.4f}"
    )

    print(
        f"  RMSE     : {best['RMSE']:.4f}"
    )

    print(
        f"  R²       : {best['R2']:.4f}"
    )


# ============================================================
# ERROR BY HELD-OUT SOURCE
# ============================================================

print("\n" + "=" * 70)
print("SOURCE-LEVEL ERROR")
print("=" * 70)

source_error_rows = []

for (
    strategy,
    model_name,
), group in predictions_df.groupby(
    ["Strategy", "Model"]
):

    for source, source_group in group.groupby(
        "Source_ID"
    ):

        row = {
            "Strategy": strategy,
            "Model": model_name,
            "Source_ID": source,
            "Records": len(source_group),
        }

        for target in TARGETS:

            actual = source_group[
                f"Actual_{target}"
            ]

            predicted = source_group[
                f"Predicted_{target}"
            ]

            row[
                f"{target}_MAE"
            ] = mean_absolute_error(
                actual,
                predicted
            )

        source_error_rows.append(row)

source_error_df = pd.DataFrame(
    source_error_rows
)

print(
    source_error_df.to_string(
        index=False
    )
)


# ============================================================
# AGGREGATE MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("AGGREGATE PERFORMANCE")
print("=" * 70)

aggregate_rows = []

for (
    strategy,
    model_name,
), group in results_df.groupby(
    ["Strategy", "Model"]
):

    aggregate_rows.append(
        {
            "Strategy": strategy,
            "Model": model_name,

            "Mean_R2": group["R2"].mean(),
            "Mean_MAE": group["MAE"].mean(),
            "Mean_RMSE": group["RMSE"].mean(),

            "UTS_MAE": group.loc[
                group["Target"] == "UTS_MPa",
                "MAE",
            ].iloc[0],

            "YS_MAE": group.loc[
                group["Target"] == "YS_MPa",
                "MAE",
            ].iloc[0],

            "Elongation_MAE": group.loc[
                group["Target"] == "Elongation_pct",
                "MAE",
            ].iloc[0],
        }
    )

aggregate_df = pd.DataFrame(
    aggregate_rows
)

print(
    aggregate_df.sort_values(
        "Mean_MAE"
    ).to_string(index=False)
)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    """
This experiment compares model strategies using the exact same
source-held-out train/test split.

BASELINE_7:
  Uses the original seven process features.

SELECTIVE_ENGINEERED:
  Adds only a subset of physically interpretable engineered
  features. This is an experiment and is not assumed to be better.

All three targets are predicted directly from process features.

No target is used as an input to predict another target.

Source_ID is not used as a model feature.

No random row split is used.

The purpose of this experiment is to determine whether a different
modeling strategy can improve unseen-source generalization without
introducing target leakage or source leakage.
"""
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    EVAL_DIR
    / "model_strategy_results_v2.csv"
)

comparison_path = (
    EVAL_DIR
    / "model_strategy_comparison_v2.csv"
)

predictions_path = (
    EVAL_DIR
    / "model_strategy_predictions_v2.csv"
)

source_error_path = (
    EVAL_DIR
    / "model_strategy_source_error_v2.csv"
)

aggregate_path = (
    EVAL_DIR
    / "model_strategy_aggregate_v2.csv"
)


results_df.to_csv(
    results_path,
    index=False
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

source_error_df.to_csv(
    source_error_path,
    index=False
)

aggregate_df.to_csv(
    aggregate_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

for path in [
    results_path,
    comparison_path,
    predictions_path,
    source_error_path,
    aggregate_path,
]:
    print(path)

print("\n" + "=" * 70)
print("STEP 9L COMPLETE")
print("=" * 70)

print(
    """
Important:
 - Original dataset unchanged.
 - train_v2.csv unchanged.
 - test_v2.csv unchanged.
 - Targets unchanged.
 - Source_ID not used as a feature.
 - No target leakage.
 - No random split.
 - Same source-held-out test set used for every strategy.
 - Models saved separately.
"""
)