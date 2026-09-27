from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# PRINTOPT AI — STEP 9M
# MODEL RELIABILITY & APPLICABILITY ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"
MODEL_DIR = BASE_DIR / "models"

EVAL_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = DATA_DIR / "train_v2.csv"
TEST_FILE = DATA_DIR / "test_v2.csv"


# ============================================================
# CORE FEATURES
# ============================================================

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


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


X_train = train_df[FEATURES].values
X_test = test_df[FEATURES].values

y_train = train_df[TARGETS].values
y_test = test_df[TARGETS].values


print("=" * 72)
print("PRINTOPT AI — STEP 9M")
print("MODEL RELIABILITY & APPLICABILITY ANALYSIS")
print("=" * 72)

print("\nDATA")
print("-" * 72)

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

print(
    "\nHeld-out sources:"
)

print(
    ", ".join(
        sorted(
            test_df["Source_ID"].unique()
        )
    )
)


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 72)
print("TRAINING RELIABILITY MODEL")
print("=" * 72)

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

rf.fit(
    X_train,
    y_train
)


# ============================================================
# 1. TRAINING DOMAIN RANGE
# ============================================================

print("\n" + "=" * 72)
print("1. TRAINING DOMAIN RANGE")
print("=" * 72)


feature_ranges = []

for feature_index, feature in enumerate(FEATURES):

    train_values = X_train[:, feature_index]

    minimum = np.min(train_values)
    maximum = np.max(train_values)

    feature_ranges.append(
        {
            "Feature": feature,
            "Train_Min": minimum,
            "Train_Max": maximum,
        }
    )

    print(
        f"{feature:<25} "
        f"{minimum:.6f} → {maximum:.6f}"
    )


range_df = pd.DataFrame(
    feature_ranges
)


# ============================================================
# 2. STANDARDIZED DOMAIN DISTANCE
# ============================================================

print("\n" + "=" * 72)
print("2. STANDARDIZED DOMAIN DISTANCE")
print("=" * 72)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ------------------------------------------------------------
# Mean absolute standardized distance
# ------------------------------------------------------------

train_abs_z = np.abs(
    X_train_scaled
)

test_abs_z = np.abs(
    X_test_scaled
)

train_mean_abs_z = (
    np.mean(
        train_abs_z,
        axis=1
    )
)

test_mean_abs_z = (
    np.mean(
        test_abs_z,
        axis=1
    )
)


# ------------------------------------------------------------
# Maximum standardized distance
# ------------------------------------------------------------

train_max_abs_z = np.max(
    train_abs_z,
    axis=1
)

test_max_abs_z = np.max(
    test_abs_z,
    axis=1
)


# ============================================================
# TRAINING-BASED DISTANCE THRESHOLDS
# ============================================================

# These thresholds are derived ONLY from training data.

mean_distance_threshold = np.percentile(
    train_mean_abs_z,
    95
)

max_distance_threshold = np.percentile(
    train_max_abs_z,
    95
)

print(
    f"\nTraining 95th percentile "
    f"mean |z|: "
    f"{mean_distance_threshold:.4f}"
)

print(
    f"Training 95th percentile "
    f"max |z|: "
    f"{max_distance_threshold:.4f}"
)


# ============================================================
# 3. NEAREST-NEIGHBOR DISTANCE
# ============================================================

print("\n" + "=" * 72)
print("3. NEAREST-NEIGHBOR DOMAIN DISTANCE")
print("=" * 72)


nearest_neighbor = NearestNeighbors(
    n_neighbors=5,
    metric="euclidean"
)

nearest_neighbor.fit(
    X_train_scaled
)


train_nn_distances, train_nn_indices = (
    nearest_neighbor.kneighbors(
        X_train_scaled
    )
)


test_nn_distances, test_nn_indices = (
    nearest_neighbor.kneighbors(
        X_test_scaled
    )
)


# Distance to closest training experiment
train_nearest_distance = (
    train_nn_distances[:, 0]
)

test_nearest_distance = (
    test_nn_distances[:, 0]
)


# Use leave-one-out nearest neighbor
# for training threshold calculation.

train_nn_loo_distance = (
    train_nn_distances[:, 1]
)


nearest_distance_threshold = np.percentile(
    train_nn_loo_distance,
    95
)


print(
    f"\nTraining 95th percentile "
    f"nearest-neighbor distance: "
    f"{nearest_distance_threshold:.4f}"
)


# ============================================================
# 4. INPUT RANGE COVERAGE
# ============================================================

print("\n" + "=" * 72)
print("4. INPUT RANGE COVERAGE")
print("=" * 72)


range_results = []

for row_index in range(
    len(test_df)
):

    experiment_id = test_df.iloc[
        row_index
    ]["Experiment_ID"]

    source_id = test_df.iloc[
        row_index
    ]["Source_ID"]

    outside_features = []

    inside_count = 0

    for feature_index, feature in enumerate(
        FEATURES
    ):

        value = X_test[
            row_index,
            feature_index
        ]

        minimum = range_df.loc[
            range_df["Feature"] == feature,
            "Train_Min",
        ].iloc[0]

        maximum = range_df.loc[
            range_df["Feature"] == feature,
            "Train_Max",
        ].iloc[0]

        if (
            value < minimum
            or value > maximum
        ):

            outside_features.append(
                feature
            )

        else:

            inside_count += 1

    coverage = (
        inside_count
        / len(FEATURES)
    )

    range_results.append(
        {
            "Experiment_ID": experiment_id,
            "Source_ID": source_id,
            "Range_Coverage": coverage,
            "Features_Inside_Range": inside_count,
            "Features_Outside_Range": len(
                outside_features
            ),
            "Outside_Range_Features": (
                ";".join(
                    outside_features
                )
                if outside_features
                else ""
            ),
        }
    )


range_results_df = pd.DataFrame(
    range_results
)


# ============================================================
# 5. RANDOM FOREST PREDICTION UNCERTAINTY
# ============================================================

print("\n" + "=" * 72)
print("5. RANDOM FOREST PREDICTION UNCERTAINTY")
print("=" * 72)


# Each Random Forest tree provides a prediction.
# Variation between trees is used as a model-uncertainty
# diagnostic. It is NOT a formal confidence interval.

tree_predictions = []

for estimator in rf.estimators_:

    prediction = estimator.predict(
        X_test
    )

    tree_predictions.append(
        prediction
    )


tree_predictions = np.stack(
    tree_predictions,
    axis=0
)


# Shape:
# n_trees × n_test × n_targets


prediction_mean = np.mean(
    tree_predictions,
    axis=0
)

prediction_std = np.std(
    tree_predictions,
    axis=0,
    ddof=1
)


# ============================================================
# TARGET-SPECIFIC UNCERTAINTY
# ============================================================

uncertainty_rows = []

for i in range(
    len(test_df)
):

    row = {
        "Experiment_ID": test_df.iloc[
            i
        ]["Experiment_ID"],

        "Source_ID": test_df.iloc[
            i
        ]["Source_ID"],
    }

    for target_index, target in enumerate(
        TARGETS
    ):

        row[
            f"{target}_Prediction"
        ] = prediction_mean[
            i,
            target_index
        ]

        row[
            f"{target}_RF_Std"
        ] = prediction_std[
            i,
            target_index
        ]

    uncertainty_rows.append(
        row
    )


uncertainty_df = pd.DataFrame(
    uncertainty_rows
)


# ============================================================
# 6. TRAINING-BASED UNCERTAINTY THRESHOLDS
# ============================================================

print("\n" + "=" * 72)
print("6. UNCERTAINTY THRESHOLDS")
print("=" * 72)


# Estimate tree uncertainty on training data.
# This is only used to establish a training-based reference.

train_tree_predictions = []

for estimator in rf.estimators_:

    prediction = estimator.predict(
        X_train
    )

    train_tree_predictions.append(
        prediction
    )


train_tree_predictions = np.stack(
    train_tree_predictions,
    axis=0
)


train_prediction_std = np.std(
    train_tree_predictions,
    axis=0,
    ddof=1
)


uncertainty_thresholds = {}

for target_index, target in enumerate(
    TARGETS
):

    threshold = np.percentile(
        train_prediction_std[
            :,
            target_index
        ],
        95
    )

    uncertainty_thresholds[
        target
    ] = threshold

    print(
        f"{target:<20} "
        f"95th percentile RF std = "
        f"{threshold:.6f}"
    )


# ============================================================
# 7. BUILD RELIABILITY TABLE
# ============================================================

print("\n" + "=" * 72)
print("7. BUILDING RELIABILITY TABLE")
print("=" * 72)


reliability_rows = []


for i in range(
    len(test_df)
):

    experiment_id = test_df.iloc[
        i
    ]["Experiment_ID"]

    source_id = test_df.iloc[
        i
    ]["Source_ID"]


    mean_abs_z = (
        test_mean_abs_z[i]
    )

    max_abs_z = (
        test_max_abs_z[i]
    )

    nearest_distance = (
        test_nearest_distance[i]
    )

    range_coverage = (
        range_results_df.iloc[i][
            "Range_Coverage"
        ]
    )

    outside_range_count = (
        range_results_df.iloc[i][
            "Features_Outside_Range"
        ]
    )


    # --------------------------------------------------------
    # Domain flags
    # --------------------------------------------------------

    mean_distance_flag = (
        mean_abs_z
        > mean_distance_threshold
    )

    max_distance_flag = (
        max_abs_z
        > max_distance_threshold
    )

    nearest_distance_flag = (
        nearest_distance
        > nearest_distance_threshold
    )

    range_flag = (
        outside_range_count > 0
    )


    # --------------------------------------------------------
    # Prediction uncertainty flags
    # --------------------------------------------------------

    uncertainty_flags = []

    for target_index, target in enumerate(
        TARGETS
    ):

        uncertainty = (
            prediction_std[
                i,
                target_index
            ]
        )

        threshold = (
            uncertainty_thresholds[
                target
            ]
        )

        if uncertainty > threshold:

            uncertainty_flags.append(
                target
            )


    # --------------------------------------------------------
    # Count warnings
    # --------------------------------------------------------

    domain_warning_count = sum(
        [
            mean_distance_flag,
            max_distance_flag,
            nearest_distance_flag,
            range_flag,
        ]
    )

    uncertainty_warning_count = len(
        uncertainty_flags
    )


    total_warning_count = (
        domain_warning_count
        + uncertainty_warning_count
    )


    # --------------------------------------------------------
    # Reliability classification
    # --------------------------------------------------------
    #
    # This is an engineering screening classification,
    # NOT a calibrated probability.
    #
    # HIGH:
    #   little evidence of extrapolation
    #
    # MODERATE:
    #   one or more domain/uncertainty warnings
    #
    # LOW:
    #   several independent warning signals
    # --------------------------------------------------------

    if (
        range_flag
        or total_warning_count >= 4
    ):

        reliability = "LOW"

    elif total_warning_count >= 2:

        reliability = "MODERATE"

    else:

        reliability = "HIGH"


    # --------------------------------------------------------
    # Human-readable warning
    # --------------------------------------------------------

    warnings = []

    if range_flag:

        warnings.append(
            "input_outside_training_range"
        )

    if mean_distance_flag:

        warnings.append(
            "high_mean_domain_distance"
        )

    if max_distance_flag:

        warnings.append(
            "high_single_feature_distance"
        )

    if nearest_distance_flag:

        warnings.append(
            "few_similar_training_experiments"
        )

    if uncertainty_flags:

        warnings.append(
            "high_model_uncertainty:"
            + ",".join(
                uncertainty_flags
            )
        )


    reliability_rows.append(
        {
            "Experiment_ID": experiment_id,
            "Source_ID": source_id,

            "Mean_Abs_Z": mean_abs_z,
            "Max_Abs_Z": max_abs_z,

            "Nearest_Training_Distance": (
                nearest_distance
            ),

            "Range_Coverage": (
                range_coverage
            ),

            "Outside_Range_Count": (
                outside_range_count
            ),

            "Domain_Warning_Count": (
                domain_warning_count
            ),

            "Uncertainty_Warning_Count": (
                uncertainty_warning_count
            ),

            "Total_Warning_Count": (
                total_warning_count
            ),

            "Reliability": reliability,

            "Warnings": (
                ";".join(warnings)
                if warnings
                else "none"
            ),
        }
    )


reliability_df = pd.DataFrame(
    reliability_rows
)


# ============================================================
# 8. MERGE EVERYTHING
# ============================================================

# reliability_df already contains:
# - Range_Coverage
# - Outside_Range_Count
#
# Therefore, do not merge those duplicate columns again.
# Only bring in the detailed list of features that were
# outside the training range.

range_detail_df = range_results_df[
    [
        "Experiment_ID",
        "Source_ID",
        "Outside_Range_Features",
    ]
].copy()


final_df = (
    reliability_df
    .merge(
        range_detail_df,
        on=[
            "Experiment_ID",
            "Source_ID",
        ],
        how="left",
    )
    .merge(
        uncertainty_df,
        on=[
            "Experiment_ID",
            "Source_ID",
        ],
        how="left",
    )
)


# ============================================================
# 9. ADD ACTUAL VALUES FOR DIAGNOSTIC PURPOSES
# ============================================================

for target in TARGETS:

    final_df[
        f"Actual_{target}"
    ] = test_df[target].values


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 72)
print("RELIABILITY RESULTS")
print("=" * 72)


display_columns = [
    "Experiment_ID",
    "Source_ID",
    "Mean_Abs_Z",
    "Max_Abs_Z",
    "Nearest_Training_Distance",
    "Range_Coverage",
    "Total_Warning_Count",
    "Reliability",
    "Warnings",
]


print(
    final_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# 11. RELIABILITY COUNTS
# ============================================================

print("\n" + "=" * 72)
print("RELIABILITY DISTRIBUTION")
print("=" * 72)


reliability_counts = (
    final_df[
        "Reliability"
    ]
    .value_counts()
)


for category in [
    "HIGH",
    "MODERATE",
    "LOW",
]:

    count = reliability_counts.get(
        category,
        0
    )

    print(
        f"{category:<10}: "
        f"{count}"
    )


# ============================================================
# 12. RANGE VIOLATIONS
# ============================================================

print("\n" + "=" * 72)
print("TRAINING-RANGE VIOLATIONS")
print("=" * 72)


range_violations = final_df[
    final_df[
        "Outside_Range_Count"
    ] > 0
]


if len(range_violations) == 0:

    print(
        "No test samples contain "
        "process values outside the "
        "training min/max ranges."
    )

else:

    print(
        range_violations[
            [
                "Experiment_ID",
                "Source_ID",
                "Outside_Range_Count",
                "Outside_Range_Features",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 13. HIGH-DISTANCE SAMPLES
# ============================================================

print("\n" + "=" * 72)
print("HIGHEST DOMAIN-DISTANCE TEST SAMPLES")
print("=" * 72)


print(
    final_df[
        [
            "Experiment_ID",
            "Source_ID",
            "Mean_Abs_Z",
            "Max_Abs_Z",
            "Nearest_Training_Distance",
            "Reliability",
        ]
    ]
    .sort_values(
        "Mean_Abs_Z",
        ascending=False,
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 14. MODEL UNCERTAINTY
# ============================================================

print("\n" + "=" * 72)
print("HIGHEST RANDOM FOREST UNCERTAINTY")
print("=" * 72)


for target in TARGETS:

    column = (
        f"{target}_RF_Std"
    )

    print(
        f"\n{target}"
    )

    print(
        final_df[
            [
                "Experiment_ID",
                "Source_ID",
                column,
                "Reliability",
            ]
        ]
        .sort_values(
            column,
            ascending=False,
        )
        .head(5)
        .to_string(
            index=False
        )
    )


# ============================================================
# 15. SOURCE-LEVEL RELIABILITY
# ============================================================

print("\n" + "=" * 72)
print("SOURCE-LEVEL RELIABILITY")
print("=" * 72)


source_summary = (
    final_df
    .groupby(
        "Source_ID"
    )
    .agg(
        Records=(
            "Experiment_ID",
            "count",
        ),

        Mean_Abs_Z=(
            "Mean_Abs_Z",
            "mean",
        ),

        Max_Abs_Z=(
            "Max_Abs_Z",
            "max",
        ),

        Mean_Nearest_Distance=(
            "Nearest_Training_Distance",
            "mean",
        ),

        Mean_Range_Coverage=(
            "Range_Coverage",
            "mean",
        ),

        Mean_Warnings=(
            "Total_Warning_Count",
            "mean",
        ),
    )
    .reset_index()
)


print(
    source_summary.to_string(
        index=False
    )
)


# ============================================================
# 16. SAVE FILES
# ============================================================

range_path = (
    EVAL_DIR
    / "reliability_training_feature_ranges_v2.csv"
)

reliability_path = (
    EVAL_DIR
    / "model_reliability_test_v2.csv"
)

source_path = (
    EVAL_DIR
    / "model_reliability_source_summary_v2.csv"
)

threshold_path = (
    EVAL_DIR
    / "model_reliability_thresholds_v2.csv"
)


range_df.to_csv(
    range_path,
    index=False
)


final_df.to_csv(
    reliability_path,
    index=False
)


source_summary.to_csv(
    source_path,
    index=False
)


threshold_rows = []

for target, threshold in (
    uncertainty_thresholds.items()
):

    threshold_rows.append(
        {
            "Metric": (
                f"{target}_RF_Std"
            ),
            "Training_95th_Percentile": (
                threshold
            ),
        }
    )


threshold_rows.extend(
    [
        {
            "Metric": "Mean_Abs_Z",
            "Training_95th_Percentile": (
                mean_distance_threshold
            ),
        },
        {
            "Metric": "Max_Abs_Z",
            "Training_95th_Percentile": (
                max_distance_threshold
            ),
        },
        {
            "Metric": "Nearest_Training_Distance",
            "Training_95th_Percentile": (
                nearest_distance_threshold
            ),
        },
    ]
)


pd.DataFrame(
    threshold_rows
).to_csv(
    threshold_path,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 72)
print("FILES CREATED")
print("=" * 72)

for path in [
    range_path,
    reliability_path,
    source_path,
    threshold_path,
]:

    print(path)


print("\n" + "=" * 72)
print("STEP 9M COMPLETE")
print("=" * 72)

print(
    """
Important interpretation:

This reliability analysis does NOT claim that the model has
a statistically calibrated probability of being correct.

The HIGH / MODERATE / LOW categories are engineering screening
categories based on training-domain distance, input-range
coverage, nearest-neighbor similarity, and Random Forest
tree-to-tree prediction variation.

The thresholds are calculated from the training data only.

The test data is not used to define the thresholds.

No Source_ID is used as a model feature.

No target variable is used as an input.

No original data values are modified.

No test samples are removed.

This layer is intended to support future PrintOpt AI UI
messages such as:

  Predicted UTS: 1217 MPa
  Reliability: MODERATE
  Domain coverage: 100%
  Warning: limited similarity to training experiments

rather than presenting every prediction as equally certain.
"""
)