from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PRINTOPT AI — STEP 9N
# RELIABILITY VALIDATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"


TRAIN_FILE = DATA_DIR / "train_v2.csv"


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
# SETTINGS
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 400,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "min_samples_leaf": 2,
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRAIN_FILE)

X = df[FEATURES].copy()
y = df[TARGETS].copy()
groups = df["Source_ID"].copy()


print("=" * 72)
print("PRINTOPT AI — STEP 9N")
print("RELIABILITY VALIDATION")
print("=" * 72)

print("\nDATA")
print("-" * 72)

print(f"Training records : {len(df)}")
print(f"Sources          : {df['Source_ID'].nunique()}")
print(f"Features         : {len(FEATURES)}")
print(f"Targets          : {len(TARGETS)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_domain_statistics(
    X_train,
    X_eval,
):
    """
    Calculate domain-distance metrics using ONLY X_train.

    Returns:
        mean_abs_z
        max_abs_z
        nearest_distance
        range_coverage
        outside_range_count
    """

    X_train_np = X_train[FEATURES].to_numpy(dtype=float)
    X_eval_np = X_eval[FEATURES].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Standardized distance
    # --------------------------------------------------------

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train_np)
    eval_scaled = scaler.transform(X_eval_np)

    abs_z = np.abs(eval_scaled)

    mean_abs_z = abs_z.mean(axis=1)
    max_abs_z = abs_z.max(axis=1)

    # --------------------------------------------------------
    # Nearest-neighbor distance
    # --------------------------------------------------------

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )

    nn.fit(train_scaled)

    nearest_distance = nn.kneighbors(
        eval_scaled,
        return_distance=True,
    )[0][:, 0]

    # --------------------------------------------------------
    # Training range coverage
    # --------------------------------------------------------

    train_min = X_train[FEATURES].min()
    train_max = X_train[FEATURES].max()

    below = X_eval[FEATURES].lt(train_min)
    above = X_eval[FEATURES].gt(train_max)

    outside = below | above

    outside_range_count = outside.sum(axis=1).to_numpy()

    range_coverage = (
        1.0
        - outside_range_count / len(FEATURES)
    )

    return {
        "Mean_Abs_Z": mean_abs_z,
        "Max_Abs_Z": max_abs_z,
        "Nearest_Training_Distance": nearest_distance,
        "Range_Coverage": range_coverage,
        "Outside_Range_Count": outside_range_count,
    }


def calculate_rf_predictions(
    X_train,
    y_train,
    X_eval,
):
    """
    Train one RF per target and calculate:

    - prediction
    - tree-to-tree prediction standard deviation
    """

    predictions = {}
    uncertainties = {}

    for target in TARGETS:

        model = RandomForestRegressor(
            **RF_PARAMS
        )

        model.fit(
            X_train[FEATURES],
            y_train[target],
        )

        tree_predictions = np.column_stack(
            [
                tree.predict(X_eval[FEATURES])
                for tree in model.estimators_
            ]
        )

        predictions[target] = tree_predictions.mean(axis=1)
        uncertainties[target] = tree_predictions.std(
            axis=1,
            ddof=1,
        )

    return predictions, uncertainties


# ============================================================
# GROUP K-FOLD
# ============================================================

print("\n" + "=" * 72)
print("1. SOURCE-AWARE OUT-OF-FOLD VALIDATION")
print("=" * 72)

gkf = GroupKFold(
    n_splits=N_SPLITS
)


fold_results = []


for fold_number, (train_idx, val_idx) in enumerate(
    gkf.split(
        X,
        y,
        groups=groups,
    ),
    start=1,
):

    print(
        f"\nFold {fold_number}"
        f" | train={len(train_idx)}"
        f" | validation={len(val_idx)}"
        f" | validation sources={groups.iloc[val_idx].nunique()}"
    )

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()

    # --------------------------------------------------------
    # Domain statistics
    # --------------------------------------------------------

    domain = calculate_domain_statistics(
        X_train,
        X_val,
    )

    # --------------------------------------------------------
    # RF predictions + uncertainty
    # --------------------------------------------------------

    predictions, uncertainties = calculate_rf_predictions(
        X_train,
        y_train,
        X_val,
    )

    fold_df = df.iloc[val_idx].copy()

    fold_df["Fold"] = fold_number

    fold_df["Mean_Abs_Z"] = domain["Mean_Abs_Z"]
    fold_df["Max_Abs_Z"] = domain["Max_Abs_Z"]
    fold_df["Nearest_Training_Distance"] = domain[
        "Nearest_Training_Distance"
    ]
    fold_df["Range_Coverage"] = domain[
        "Range_Coverage"
    ]
    fold_df["Outside_Range_Count"] = domain[
        "Outside_Range_Count"
    ]

    for target in TARGETS:

        fold_df[f"{target}_Prediction"] = predictions[target]

        fold_df[f"{target}_RF_Std"] = uncertainties[target]

        fold_df[f"{target}_Absolute_Error"] = np.abs(
            y_val[target].to_numpy()
            - predictions[target]
        )

    fold_results.append(
        fold_df
    )


oof_df = pd.concat(
    fold_results,
    ignore_index=True,
)


# ============================================================
# 2. TRAINING-DERIVED THRESHOLDS
# ============================================================

print("\n" + "=" * 72)
print("2. RELIABILITY THRESHOLDS")
print("=" * 72)

# These thresholds are calculated from OOF validation values.
# They are NOT calculated from the final test set.

mean_z_threshold = oof_df[
    "Mean_Abs_Z"
].quantile(0.95)

max_z_threshold = oof_df[
    "Max_Abs_Z"
].quantile(0.95)

nearest_threshold = oof_df[
    "Nearest_Training_Distance"
].quantile(0.95)


uncertainty_thresholds = {}

for target in TARGETS:

    uncertainty_thresholds[target] = oof_df[
        f"{target}_RF_Std"
    ].quantile(0.95)


print(
    f"\nOOF 95th percentile mean |z|:"
    f" {mean_z_threshold:.6f}"
)

print(
    f"OOF 95th percentile max |z|:"
    f" {max_z_threshold:.6f}"
)

print(
    f"OOF 95th percentile nearest distance:"
    f" {nearest_threshold:.6f}"
)

for target in TARGETS:

    print(
        f"{target:<20}"
        f"95th percentile RF std = "
        f"{uncertainty_thresholds[target]:.6f}"
    )


# ============================================================
# 3. RELIABILITY WARNINGS
# ============================================================

def classify_reliability(row):

    warnings = []

    # --------------------------------------------------------
    # Domain warnings
    # --------------------------------------------------------

    if row["Mean_Abs_Z"] > mean_z_threshold:
        warnings.append(
            "high_mean_domain_distance"
        )

    if row["Max_Abs_Z"] > max_z_threshold:
        warnings.append(
            "high_single_feature_distance"
        )

    if row["Nearest_Training_Distance"] > nearest_threshold:
        warnings.append(
            "few_similar_training_experiments"
        )

    if row["Range_Coverage"] < 1.0:
        warnings.append(
            "outside_training_range"
        )

    # --------------------------------------------------------
    # Model uncertainty warnings
    # --------------------------------------------------------

    high_uncertainty_targets = []

    for target in TARGETS:

        if (
            row[f"{target}_RF_Std"]
            > uncertainty_thresholds[target]
        ):
            high_uncertainty_targets.append(
                target
            )

    if high_uncertainty_targets:

        warnings.append(
            "high_model_uncertainty:"
            + ",".join(high_uncertainty_targets)
        )

    # --------------------------------------------------------
    # Screening category
    # --------------------------------------------------------

    warning_count = len(warnings)

    if warning_count >= 4:
        reliability = "LOW"

    elif warning_count >= 2:
        reliability = "MODERATE"

    else:
        reliability = "HIGH"

    return reliability, warning_count, warnings


classification_results = []

for _, row in oof_df.iterrows():

    reliability, warning_count, warnings = (
        classify_reliability(row)
    )

    classification_results.append(
        {
            "Reliability": reliability,
            "Total_Warning_Count": warning_count,
            "Warnings": ";".join(warnings)
            if warnings
            else "none",
        }
    )


classification_df = pd.DataFrame(
    classification_results
)


oof_df = pd.concat(
    [
        oof_df.reset_index(drop=True),
        classification_df,
    ],
    axis=1,
)


# ============================================================
# 4. OVERALL ERROR SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("3. OVERALL OUT-OF-FOLD ERROR")
print("=" * 72)

for target in TARGETS:

    mae = mean_absolute_error(
        oof_df[target],
        oof_df[f"{target}_Prediction"],
    )

    print(
        f"{target:<20}"
        f"MAE = {mae:.4f}"
    )


# ============================================================
# 5. ERROR BY RELIABILITY CATEGORY
# ============================================================

print("\n" + "=" * 72)
print("4. ERROR BY RELIABILITY CATEGORY")
print("=" * 72)


category_rows = []


for reliability in [
    "HIGH",
    "MODERATE",
    "LOW",
]:

    subset = oof_df[
        oof_df["Reliability"] == reliability
    ]

    if len(subset) == 0:
        continue

    row = {
        "Reliability": reliability,
        "Records": len(subset),
        "Mean_Warning_Count": subset[
            "Total_Warning_Count"
        ].mean(),
    }

    for target in TARGETS:

        row[
            f"{target}_MAE"
        ] = subset[
            f"{target}_Absolute_Error"
        ].mean()

    category_rows.append(row)

    print(
        f"\n{reliability}"
    )

    print(
        f"Records              : {len(subset)}"
    )

    print(
        f"Mean warning count   : "
        f"{subset['Total_Warning_Count'].mean():.3f}"
    )

    for target in TARGETS:

        print(
            f"{target:<20}: "
            f"{subset[f'{target}_Absolute_Error'].mean():.4f}"
        )


category_error_df = pd.DataFrame(
    category_rows
)


# ============================================================
# 6. ERROR BY WARNING COUNT
# ============================================================

print("\n" + "=" * 72)
print("5. ERROR BY WARNING COUNT")
print("=" * 72)


warning_rows = []


for count in sorted(
    oof_df["Total_Warning_Count"].unique()
):

    subset = oof_df[
        oof_df["Total_Warning_Count"] == count
    ]

    row = {
        "Warning_Count": int(count),
        "Records": len(subset),
    }

    for target in TARGETS:

        row[
            f"{target}_MAE"
        ] = subset[
            f"{target}_Absolute_Error"
        ].mean()

    warning_rows.append(row)

    print(
        f"\nWarnings = {count}"
    )

    print(
        f"Records : {len(subset)}"
    )

    for target in TARGETS:

        print(
            f"{target:<20}: "
            f"{row[f'{target}_MAE']:.4f}"
        )


warning_error_df = pd.DataFrame(
    warning_rows
)


# ============================================================
# 7. DOMAIN SIGNAL VS ERROR
# ============================================================

print("\n" + "=" * 72)
print("6. DOMAIN SIGNAL CORRELATION")
print("=" * 72)


domain_error_rows = []


domain_metrics = [
    "Mean_Abs_Z",
    "Max_Abs_Z",
    "Nearest_Training_Distance",
    "Outside_Range_Count",
    "Total_Warning_Count",
]


for metric in domain_metrics:

    row = {
        "Metric": metric
    }

    for target in TARGETS:

        error_column = (
            f"{target}_Absolute_Error"
        )

        correlation = oof_df[
            metric
        ].corr(
            oof_df[error_column]
        )

        row[
            f"{target}_Error_Correlation"
        ] = correlation

    domain_error_rows.append(row)

    print(
        f"\n{metric}"
    )

    for target in TARGETS:

        value = row[
            f"{target}_Error_Correlation"
        ]

        print(
            f"{target:<20}: {value:.4f}"
        )


domain_error_df = pd.DataFrame(
    domain_error_rows
)


# ============================================================
# 8. UNCERTAINTY VS ERROR
# ============================================================

print("\n" + "=" * 72)
print("7. RF UNCERTAINTY VS ABSOLUTE ERROR")
print("=" * 72)


uncertainty_rows = []


for target in TARGETS:

    uncertainty_column = (
        f"{target}_RF_Std"
    )

    error_column = (
        f"{target}_Absolute_Error"
    )

    correlation = oof_df[
        uncertainty_column
    ].corr(
        oof_df[error_column]
    )

    uncertainty_rows.append(
        {
            "Target": target,
            "Uncertainty_Error_Correlation": correlation,
        }
    )

    print(
        f"{target:<20}: {correlation:.4f}"
    )


uncertainty_error_df = pd.DataFrame(
    uncertainty_rows
)


# ============================================================
# 9. SOURCE-LEVEL VALIDATION
# ============================================================

print("\n" + "=" * 72)
print("8. SOURCE-LEVEL RELIABILITY VALIDATION")
print("=" * 72)


source_rows = []


for source_id, source_df in oof_df.groupby(
    "Source_ID"
):

    row = {
        "Source_ID": source_id,
        "Records": len(source_df),
        "Mean_Warnings": source_df[
            "Total_Warning_Count"
        ].mean(),
        "Mean_Abs_Z": source_df[
            "Mean_Abs_Z"
        ].mean(),
        "Max_Abs_Z": source_df[
            "Max_Abs_Z"
        ].max(),
        "Mean_Nearest_Distance": source_df[
            "Nearest_Training_Distance"
        ].mean(),
    }

    for target in TARGETS:

        row[
            f"{target}_MAE"
        ] = source_df[
            f"{target}_Absolute_Error"
        ].mean()

    source_rows.append(row)


source_validation_df = pd.DataFrame(
    source_rows
)

source_validation_df = source_validation_df.sort_values(
    by="Records",
    ascending=False,
)


print(
    source_validation_df.to_string(
        index=False
    )
)


# ============================================================
# 10. RELIABILITY DISTRIBUTION
# ============================================================

print("\n" + "=" * 72)
print("9. OOF RELIABILITY DISTRIBUTION")
print("=" * 72)


distribution = (
    oof_df["Reliability"]
    .value_counts()
)


for category in [
    "HIGH",
    "MODERATE",
    "LOW",
]:

    count = int(
        distribution.get(
            category,
            0
        )
    )

    percentage = (
        count / len(oof_df) * 100
    )

    print(
        f"{category:<10}: "
        f"{count:3d} "
        f"({percentage:5.1f}%)"
    )


# ============================================================
# 11. HIGH VS LOW ERROR RATIO
# ============================================================

print("\n" + "=" * 72)
print("10. HIGH VS LOW ERROR COMPARISON")
print("=" * 72)


comparison_rows = []


high_df = oof_df[
    oof_df["Reliability"] == "HIGH"
]

low_df = oof_df[
    oof_df["Reliability"] == "LOW"
]


for target in TARGETS:

    high_mae = (
        high_df[
            f"{target}_Absolute_Error"
        ].mean()
        if len(high_df)
        else np.nan
    )

    low_mae = (
        low_df[
            f"{target}_Absolute_Error"
        ].mean()
        if len(low_df)
        else np.nan
    )

    if (
        pd.notna(high_mae)
        and high_mae > 0
        and pd.notna(low_mae)
    ):
        ratio = low_mae / high_mae

    else:
        ratio = np.nan

    comparison_rows.append(
        {
            "Target": target,
            "HIGH_MAE": high_mae,
            "LOW_MAE": low_mae,
            "LOW_to_HIGH_MAE_Ratio": ratio,
        }
    )

    print(
        f"{target:<20}"
        f"HIGH MAE = {high_mae:.4f} | "
        f"LOW MAE = {low_mae:.4f} | "
        f"Ratio = {ratio:.4f}"
    )


high_low_comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

print("\n" + "=" * 72)
print("11. SAVING RESULTS")
print("=" * 72)


oof_output = (
    EVAL_DIR
    / "reliability_oof_predictions_v2.csv"
)

category_output = (
    EVAL_DIR
    / "reliability_error_by_category_v2.csv"
)

warning_output = (
    EVAL_DIR
    / "reliability_error_by_warning_count_v2.csv"
)

domain_output = (
    EVAL_DIR
    / "reliability_domain_error_correlation_v2.csv"
)

uncertainty_output = (
    EVAL_DIR
    / "reliability_uncertainty_error_correlation_v2.csv"
)

source_output = (
    EVAL_DIR
    / "reliability_source_validation_v2.csv"
)

comparison_output = (
    EVAL_DIR
    / "reliability_high_low_comparison_v2.csv"
)


oof_df.to_csv(
    oof_output,
    index=False,
)

category_error_df.to_csv(
    category_output,
    index=False,
)

warning_error_df.to_csv(
    warning_output,
    index=False,
)

domain_error_df.to_csv(
    domain_output,
    index=False,
)

uncertainty_error_df.to_csv(
    uncertainty_output,
    index=False,
)

source_validation_df.to_csv(
    source_output,
    index=False,
)

high_low_comparison_df.to_csv(
    comparison_output,
    index=False,
)


print(f"\n{oof_output}")
print(category_output)
print(warning_output)
print(domain_output)
print(uncertainty_output)
print(source_output)
print(comparison_output)


# ============================================================
# 13. FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 72)
print("STEP 9N COMPLETE")
print("=" * 72)

print(
    """
This analysis validates the reliability layer using
source-aware out-of-fold predictions.

Important:
- The final test set was NOT used.
- Source_ID was NOT used as a model feature.
- Targets were NOT used as input features.
- Reliability categories are NOT calibrated probabilities.
- HIGH / MODERATE / LOW remain engineering screening categories.
- No original dataset values were changed.
- No records were removed.

The next decision is whether the current reliability
signals provide useful discrimination between lower-error
and higher-error predictions.

If the LOW category consistently has higher validation
error than HIGH, the reliability layer has supporting
evidence for deployment.

If that relationship is weak, we will revise the
screening logic before connecting it to the UI.
"""
)