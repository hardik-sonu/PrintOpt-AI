"""
PRINTOPT AI — STEP 9J
FEATURE ENGINEERING & PHYSICS AUDIT V2

Purpose:
    Test whether physically meaningful engineered process features
    improve source-held-out prediction for Ti-6Al-4V LPBF.

Important:
    - Uses the EXACT existing train_v2 / test_v2 split.
    - Does not modify the original dataset.
    - Does not modify train_v2.csv or test_v2.csv.
    - Does not use target variables as features.
    - Does not use Source_ID as a feature.
    - Does not perform a new random split.
    - Preserves the original 7-feature baseline for comparison.

Models:
    1. Random Forest
    2. XGBoost

Feature sets:
    A. BASELINE_7
    B. ENGINEERED_13
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"

TRAIN_FILE = DATA_DIR / "train_v2.csv"
TEST_FILE = DATA_DIR / "test_v2.csv"

RESULTS_FILE = EVAL_DIR / "feature_engineering_results_v2.csv"
IMPORTANCE_FILE = EVAL_DIR / "feature_engineering_importance_v2.csv"
PREDICTIONS_FILE = EVAL_DIR / "feature_engineering_predictions_v2.csv"
SUMMARY_FILE = EVAL_DIR / "feature_engineering_summary_v2.csv"


# ============================================================
# ORIGINAL FEATURES
# ============================================================

BASELINE_FEATURES = [
    "Powder_Size_um",
    "Laser_Spot_um",
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Hatch_Distance_um",
    "Layer_Thickness_um",
    "VED_J_mm3",
]


# ============================================================
# ENGINEERED FEATURES
# ============================================================

ENGINEERED_FEATURES = [
    "Linear_Energy_Density_J_mm",
    "Hatch_to_Spot_Ratio",
    "Layer_to_Spot_Ratio",
    "Spot_to_Powder_Ratio",
    "Hatch_to_Powder_Ratio",
    "Layer_to_Powder_Ratio",
]


ALL_FEATURES = BASELINE_FEATURES + ENGINEERED_FEATURES


TARGETS = [
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df):
    """
    Add physics-informed derived process features.

    All features are calculated exclusively from process/material
    input variables. No target information is used.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Linear energy density
    #
    # P / v
    #
    # W / (mm/s) = J/mm
    # --------------------------------------------------------

    df["Linear_Energy_Density_J_mm"] = (
        df["Laser_Power_W"]
        / df["Scanning_Speed_mm_s"]
    )

    # --------------------------------------------------------
    # Geometric/process ratios
    # --------------------------------------------------------

    df["Hatch_to_Spot_Ratio"] = (
        df["Hatch_Distance_um"]
        / df["Laser_Spot_um"]
    )

    df["Layer_to_Spot_Ratio"] = (
        df["Layer_Thickness_um"]
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

    df["Layer_to_Powder_Ratio"] = (
        df["Layer_Thickness_um"]
        / df["Powder_Size_um"]
    )

    return df


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_random_forest():
    return MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
    )


def create_xgboost():
    return MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred, target):
    return {
        "Target": target,
        "R2": r2_score(
            y_true[target],
            y_pred[target],
        ),
        "MAE": mean_absolute_error(
            y_true[target],
            y_pred[target],
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true[target],
                y_pred[target],
            )
        ),
    }


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9J")
print("FEATURE ENGINEERING & PHYSICS AUDIT V2")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print()
print("DATA")
print("-" * 70)

print(f"Training records : {len(train)}")
print(f"Testing records  : {len(test)}")
print(f"Training sources : {train['Source_ID'].nunique()}")
print(f"Testing sources  : {test['Source_ID'].nunique()}")

print()
print("Held-out test sources:")
print(
    ", ".join(
        sorted(test["Source_ID"].unique())
    )
)


# ============================================================
# ADD ENGINEERED FEATURES
# ============================================================

train = add_engineered_features(train)
test = add_engineered_features(test)


# ============================================================
# PHYSICAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("1. ENGINEERED FEATURE VALIDATION")
print("=" * 70)

for feature in ENGINEERED_FEATURES:

    train_values = train[feature]
    test_values = test[feature]

    print()
    print(feature)

    print(
        f"  Train range : "
        f"{train_values.min():.6f} → "
        f"{train_values.max():.6f}"
    )

    print(
        f"  Test range  : "
        f"{test_values.min():.6f} → "
        f"{test_values.max():.6f}"
    )

    print(
        f"  Train mean  : "
        f"{train_values.mean():.6f}"
    )

    print(
        f"  Test mean   : "
        f"{test_values.mean():.6f}"
    )

    if not np.isfinite(train_values).all():
        raise ValueError(
            f"Non-finite values detected in training feature: {feature}"
        )

    if not np.isfinite(test_values).all():
        raise ValueError(
            f"Non-finite values detected in testing feature: {feature}"
        )


# ============================================================
# FEATURE CORRELATION / REDUNDANCY AUDIT
# ============================================================

print()
print("=" * 70)
print("2. FEATURE REDUNDANCY AUDIT")
print("=" * 70)

feature_corr = train[
    ALL_FEATURES
].corr()

high_corr_pairs = []

for i, feature_a in enumerate(ALL_FEATURES):

    for j, feature_b in enumerate(ALL_FEATURES):

        if j <= i:
            continue

        corr = feature_corr.loc[
            feature_a,
            feature_b
        ]

        if abs(corr) >= 0.90:

            high_corr_pairs.append({
                "Feature_A": feature_a,
                "Feature_B": feature_b,
                "Pearson_Correlation": corr,
            })


if high_corr_pairs:

    high_corr_df = pd.DataFrame(
        high_corr_pairs
    )

    print(
        high_corr_df.to_string(
            index=False
        )
    )

else:

    high_corr_df = pd.DataFrame(
        columns=[
            "Feature_A",
            "Feature_B",
            "Pearson_Correlation",
        ]
    )

    print(
        "No feature pairs with "
        "|Pearson correlation| >= 0.90."
    )


# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {
    "BASELINE_7": BASELINE_FEATURES,
    "ENGINEERED_13": ALL_FEATURES,
}


# ============================================================
# STORAGE
# ============================================================

results = []
importance_rows = []
prediction_frames = []


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

for feature_set_name, features in feature_sets.items():

    print()
    print("=" * 70)
    print(f"3. FEATURE SET: {feature_set_name}")
    print("=" * 70)

    print()
    print("Features:")
    for feature in features:
        print(f"  - {feature}")

    X_train = train[features]
    X_test = test[features]

    y_train = train[TARGETS]
    y_test = test[TARGETS]

    models = {
        "Random Forest": create_random_forest(),
        "XGBoost": create_xgboost(),
    }

    for model_name, model in models.items():

        print()
        print(
            f"Training {model_name} "
            f"with {len(features)} features..."
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        prediction_df = test[
            [
                "Experiment_ID",
                "Source_ID",
            ]
        ].copy()

        prediction_df[
            "Feature_Set"
        ] = feature_set_name

        prediction_df[
            "Model"
        ] = model_name

        for index, target in enumerate(TARGETS):

            prediction_df[
                f"{target}_Actual"
            ] = y_test[target].values

            prediction_df[
                f"{target}_Predicted"
            ] = predictions[:, index]

            prediction_df[
                f"{target}_AbsError"
            ] = np.abs(
                y_test[target].values
                - predictions[:, index]
            )

        prediction_frames.append(
            prediction_df
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        print()
        print(f"{model_name} results:")

        for target in TARGETS:

            metrics = calculate_metrics(
                y_test,
                pd.DataFrame(
                    predictions,
                    columns=TARGETS
                ),
                target,
            )

            results.append({
                "Feature_Set": feature_set_name,
                "Model": model_name,
                "Target": target,
                "R2": metrics["R2"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "Feature_Count": len(features),
            })

            print(
                f"  {target:<16} "
                f"R²={metrics['R2']:>9.4f}  "
                f"MAE={metrics['MAE']:>9.4f}  "
                f"RMSE={metrics['RMSE']:>9.4f}"
            )

        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        if model_name in [
            "Random Forest",
            "XGBoost",
        ]:

            for target_index, target in enumerate(TARGETS):

                estimator = model.estimators_[
                    target_index
                ]

                if hasattr(
                    estimator,
                    "feature_importances_"
                ):

                    importances = (
                        estimator.feature_importances_
                    )

                    for feature, importance in zip(
                        features,
                        importances,
                    ):

                        importance_rows.append({
                            "Feature_Set":
                                feature_set_name,

                            "Model":
                                model_name,

                            "Target":
                                target,

                            "Feature":
                                feature,

                            "Importance":
                                importance,
                        })


# ============================================================
# CREATE DATAFRAMES
# ============================================================

results_df = pd.DataFrame(results)

importance_df = pd.DataFrame(
    importance_rows
)

predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True
)


# ============================================================
# 4. RESULT COMPARISON
# ============================================================

print()
print("=" * 70)
print("4. BASELINE VS ENGINEERED FEATURE RESULTS")
print("=" * 70)

comparison = results_df.pivot_table(
    index=[
        "Model",
        "Target",
    ],
    columns="Feature_Set",
    values=[
        "R2",
        "MAE",
        "RMSE",
    ],
)

print(
    comparison.to_string()
)


# ============================================================
# 5. IMPROVEMENT CALCULATION
# ============================================================

print()
print("=" * 70)
print("5. ENGINEERED FEATURE IMPROVEMENT")
print("=" * 70)

improvement_rows = []

for model in results_df["Model"].unique():

    for target in TARGETS:

        baseline_row = results_df[
            (results_df["Model"] == model)
            &
            (results_df["Target"] == target)
            &
            (results_df["Feature_Set"] == "BASELINE_7")
        ]

        engineered_row = results_df[
            (results_df["Model"] == model)
            &
            (results_df["Target"] == target)
            &
            (results_df["Feature_Set"] == "ENGINEERED_13")
        ]

        if baseline_row.empty or engineered_row.empty:
            continue

        baseline = baseline_row.iloc[0]
        engineered = engineered_row.iloc[0]

        r2_change = (
            engineered["R2"]
            - baseline["R2"]
        )

        mae_change = (
            engineered["MAE"]
            - baseline["MAE"]
        )

        rmse_change = (
            engineered["RMSE"]
            - baseline["RMSE"]
        )

        if baseline["MAE"] != 0:
            mae_percent_change = (
                (
                    engineered["MAE"]
                    - baseline["MAE"]
                )
                / baseline["MAE"]
            ) * 100
        else:
            mae_percent_change = np.nan

        improvement_rows.append({
            "Model": model,
            "Target": target,

            "Baseline_R2":
                baseline["R2"],

            "Engineered_R2":
                engineered["R2"],

            "R2_Change":
                r2_change,

            "Baseline_MAE":
                baseline["MAE"],

            "Engineered_MAE":
                engineered["MAE"],

            "MAE_Change":
                mae_change,

            "MAE_Percent_Change":
                mae_percent_change,

            "Baseline_RMSE":
                baseline["RMSE"],

            "Engineered_RMSE":
                engineered["RMSE"],

            "RMSE_Change":
                rmse_change,
        })


improvement_df = pd.DataFrame(
    improvement_rows
)

print(
    improvement_df.to_string(
        index=False
    )
)


# ============================================================
# 6. INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("6. INTERPRETATION")
print("=" * 70)

for _, row in improvement_df.iterrows():

    model = row["Model"]
    target = row["Target"]

    r2_change = row["R2_Change"]
    mae_change = row["MAE_Change"]

    print()
    print(
        f"{model} — {target}"
    )

    print(
        f"  ΔR²   : {r2_change:+.4f}"
    )

    print(
        f"  ΔMAE  : {mae_change:+.4f}"
    )

    if r2_change > 0 and mae_change < 0:

        print(
            "  Direction: improved on both R² and MAE."
        )

    elif r2_change < 0 and mae_change > 0:

        print(
            "  Direction: degraded on both R² and MAE."
        )

    else:

        print(
            "  Direction: mixed change."
        )


# ============================================================
# 7. FEATURE IMPORTANCE — ENGINEERED MODEL
# ============================================================

print()
print("=" * 70)
print("7. ENGINEERED MODEL FEATURE IMPORTANCE")
print("=" * 70)

engineered_importance = importance_df[
    importance_df["Feature_Set"] == "ENGINEERED_13"
].copy()

if not engineered_importance.empty:

    for model in [
        "Random Forest",
        "XGBoost",
    ]:

        print()
        print(model)

        for target in TARGETS:

            subset = engineered_importance[
                (engineered_importance["Model"] == model)
                &
                (engineered_importance["Target"] == target)
            ].copy()

            subset = subset.sort_values(
                "Importance",
                ascending=False
            )

            print()
            print(f"  {target}")

            print(
                subset[
                    [
                        "Feature",
                        "Importance",
                    ]
                ].to_string(
                    index=False
                )
            )


# ============================================================
# 8. CHECK WHETHER ENGINEERED FEATURES HELP
# ============================================================

print()
print("=" * 70)
print("8. ENGINEERED FEATURE SIGNAL")
print("=" * 70)

for feature in ENGINEERED_FEATURES:

    correlations = []

    for target in TARGETS:

        corr = train[
            feature
        ].corr(
            train[target]
        )

        correlations.append(
            (
                target,
                corr
            )
        )

    print()
    print(feature)

    for target, corr in correlations:

        print(
            f"  {target:<16} "
            f"Pearson r = {corr:+.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

improvement_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(RESULTS_FILE)
print(IMPORTANCE_FILE)
print(PREDICTIONS_FILE)
print(SUMMARY_FILE)

print()
print("=" * 70)
print("STEP 9J COMPLETE")
print("=" * 70)

print()
print("Important:")
print(" - Original dataset unchanged.")
print(" - train_v2.csv unchanged.")
print(" - test_v2.csv unchanged.")
print(" - Targets unchanged.")
print(" - Source_ID was not used as a feature.")
print(" - Exact same source-held-out split was used.")
print(" - No random split was introduced.")
print("=" * 70)