"""
PRINTOPT AI — STEP 9I
HELD-OUT SOURCE DOMAIN COMPARISON

Purpose:
    Investigate whether poor source-aware model performance is
    isolated to REF_16 or is a broader domain-shift problem.

This step:
    - compares all held-out test sources against training data
    - measures feature distribution shift
    - compares target distributions
    - connects distribution shift with model error
    - does NOT modify the dataset
    - does NOT remove records
    - does NOT retrain models
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"

TRAIN_FILE = DATA_DIR / "train_v2.csv"
TEST_FILE = DATA_DIR / "test_v2.csv"
PREDICTION_FILE = EVAL_DIR / "baseline_predictions_v2.csv"

OUTPUT_SOURCE = EVAL_DIR / "heldout_source_domain_summary_v2.csv"
OUTPUT_FEATURE = EVAL_DIR / "heldout_source_feature_shift_v2.csv"
OUTPUT_TARGET = EVAL_DIR / "heldout_source_target_shift_v2.csv"
OUTPUT_ERROR = EVAL_DIR / "heldout_source_error_vs_shift_v2.csv"
OUTPUT_SUMMARY = EVAL_DIR / "heldout_source_investigation_summary_v2.csv"


# ============================================================
# CONFIGURATION
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

MODEL_ERROR_COLUMNS = {
    "Random Forest V2": {
        "UTS_MPa": "RF_UTS_MPa_AbsError",
        "YS_MPa": "RF_YS_MPa_AbsError",
        "Elongation_pct": "RF_Elongation_pct_AbsError",
    },
    "XGBoost V2": {
        "UTS_MPa": "XGB_UTS_MPa_AbsError",
        "YS_MPa": "XGB_YS_MPa_AbsError",
        "Elongation_pct": "XGB_Elongation_pct_AbsError",
    },
    "MLP V2": {
        "UTS_MPa": "MLP_UTS_MPa_AbsError",
        "YS_MPa": "MLP_YS_MPa_AbsError",
        "Elongation_pct": "MLP_Elongation_pct_AbsError",
    },
}


# ============================================================
# HELPERS
# ============================================================

def safe_mean(series):
    return float(series.mean()) if len(series) else np.nan


def safe_std(series):
    return float(series.std()) if len(series) > 1 else 0.0


def safe_z(value, mean, std):
    if std == 0 or np.isnan(std):
        return 0.0
    return float((value - mean) / std)


def range_flag(value_min, value_max, train_min, train_max):
    below = value_min < train_min
    above = value_max > train_max

    if below and above:
        return "OUTSIDE_BOTH_SIDES"
    if below:
        return "BELOW_TRAINING_RANGE"
    if above:
        return "ABOVE_TRAINING_RANGE"

    return "WITHIN_TRAINING_RANGE"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9I")
print("HELD-OUT SOURCE DOMAIN COMPARISON")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)
pred = pd.read_csv(PREDICTION_FILE)

print()
print("INPUT DATA")
print("-" * 70)
print(f"Training records : {len(train)}")
print(f"Testing records  : {len(test)}")
print(f"Training sources : {train['Source_ID'].nunique()}")
print(f"Testing sources  : {test['Source_ID'].nunique()}")

test_sources = sorted(test["Source_ID"].unique())

print()
print("HELD-OUT SOURCES")
print("-" * 70)

for source in test_sources:
    count = (test["Source_ID"] == source).sum()
    print(f"{source:<10} {count:>3} records")


# ============================================================
# 1. TRAINING DISTRIBUTION REFERENCE
# ============================================================

train_stats = {}

for col in FEATURES + TARGETS:
    train_stats[col] = {
        "mean": train[col].mean(),
        "std": train[col].std(),
        "min": train[col].min(),
        "max": train[col].max(),
    }


# ============================================================
# 2. FEATURE SHIFT BY SOURCE
# ============================================================

feature_rows = []

for source in test_sources:

    source_df = test[test["Source_ID"] == source]

    for feature in FEATURES:

        values = source_df[feature]

        source_mean = values.mean()
        source_min = values.min()
        source_max = values.max()

        train_mean = train_stats[feature]["mean"]
        train_std = train_stats[feature]["std"]
        train_min = train_stats[feature]["min"]
        train_max = train_stats[feature]["max"]

        mean_z = safe_z(source_mean, train_mean, train_std)

        value_z = (
            (values - train_mean) / train_std
            if train_std != 0
            else pd.Series(0.0, index=values.index)
        )

        abs_z = value_z.abs()

        feature_rows.append({
            "Source_ID": source,
            "Feature": feature,
            "Records": len(source_df),

            "Source_Mean": source_mean,
            "Source_Min": source_min,
            "Source_Max": source_max,

            "Training_Mean": train_mean,
            "Training_Std": train_std,
            "Training_Min": train_min,
            "Training_Max": train_max,

            "Mean_Z_vs_Training": mean_z,
            "Max_Absolute_Z": abs_z.max(),
            "Mean_Absolute_Z": abs_z.mean(),

            "Values_Below_Training_Min":
                int((values < train_min).sum()),

            "Values_Above_Training_Max":
                int((values > train_max).sum()),

            "Values_Abs_Z_Greater_2":
                int((abs_z > 2).sum()),

            "Values_Abs_Z_Greater_3":
                int((abs_z > 3).sum()),

            "Range_Flag":
                range_flag(
                    source_min,
                    source_max,
                    train_min,
                    train_max
                ),
        })


feature_shift = pd.DataFrame(feature_rows)


# ============================================================
# 3. TARGET SHIFT BY SOURCE
# ============================================================

target_rows = []

for source in test_sources:

    source_df = test[test["Source_ID"] == source]

    for target in TARGETS:

        values = source_df[target]

        source_mean = values.mean()
        source_min = values.min()
        source_max = values.max()

        train_mean = train_stats[target]["mean"]
        train_std = train_stats[target]["std"]
        train_min = train_stats[target]["min"]
        train_max = train_stats[target]["max"]

        mean_z = safe_z(source_mean, train_mean, train_std)

        value_z = (
            (values - train_mean) / train_std
            if train_std != 0
            else pd.Series(0.0, index=values.index)
        )

        abs_z = value_z.abs()

        target_rows.append({
            "Source_ID": source,
            "Target": target,
            "Records": len(source_df),

            "Source_Mean": source_mean,
            "Source_Std": values.std() if len(values) > 1 else 0.0,
            "Source_Min": source_min,
            "Source_Max": source_max,

            "Training_Mean": train_mean,
            "Training_Std": train_std,
            "Training_Min": train_min,
            "Training_Max": train_max,

            "Mean_Z_vs_Training": mean_z,
            "Max_Absolute_Z": abs_z.max(),
            "Mean_Absolute_Z": abs_z.mean(),

            "Values_Below_Training_Min":
                int((values < train_min).sum()),

            "Values_Above_Training_Max":
                int((values > train_max).sum()),

            "Values_Abs_Z_Greater_2":
                int((abs_z > 2).sum()),

            "Values_Abs_Z_Greater_3":
                int((abs_z > 3).sum()),

            "Range_Flag":
                range_flag(
                    source_min,
                    source_max,
                    train_min,
                    train_max
                ),
        })


target_shift = pd.DataFrame(target_rows)


# ============================================================
# 4. SOURCE-LEVEL ERROR
# ============================================================

error_rows = []

for source in test_sources:

    source_pred = pred[pred["Source_ID"] == source]

    if source_pred.empty:
        continue

    row = {
        "Source_ID": source,
        "Records": len(source_pred),
    }

    for model_name, target_columns in MODEL_ERROR_COLUMNS.items():

        for target, error_col in target_columns.items():

            if error_col not in source_pred.columns:
                continue

            row[
                f"{model_name.replace(' ', '_')}_{target}_MAE"
            ] = source_pred[error_col].mean()

    error_rows.append(row)


error_by_source = pd.DataFrame(error_rows)


# ============================================================
# 5. AGGREGATED DOMAIN SHIFT SCORE
# ============================================================

# We calculate an interpretable source-level feature shift score:
#
# Mean absolute Z-score across the seven process features.
#
# This is NOT a formal statistical distance.
# It is a diagnostic indicator only.

shift_summary_rows = []

for source in test_sources:

    source_features = feature_shift[
        feature_shift["Source_ID"] == source
    ]

    source_targets = target_shift[
        target_shift["Source_ID"] == source
    ]

    process_mean_abs_z = source_features[
        "Mean_Absolute_Z"
    ].mean()

    process_max_abs_z = source_features[
        "Max_Absolute_Z"
    ].max()

    shifted_features = int(
        (source_features["Mean_Absolute_Z"] >= 2).sum()
    )

    outside_range_features = int(
        (
            source_features["Range_Flag"]
            != "WITHIN_TRAINING_RANGE"
        ).sum()
    )

    target_mean_abs_z = source_targets[
        "Mean_Absolute_Z"
    ].mean()

    target_max_abs_z = source_targets[
        "Max_Absolute_Z"
    ].max()

    shift_summary_rows.append({
        "Source_ID": source,
        "Records": len(test[test["Source_ID"] == source]),

        "Process_Mean_Absolute_Z": process_mean_abs_z,
        "Process_Max_Absolute_Z": process_max_abs_z,

        "Features_Shifted_GE_2SD":
            shifted_features,

        "Features_Outside_Training_Range":
            outside_range_features,

        "Target_Mean_Absolute_Z":
            target_mean_abs_z,

        "Target_Max_Absolute_Z":
            target_max_abs_z,
    })


source_summary = pd.DataFrame(shift_summary_rows)

source_summary = source_summary.merge(
    error_by_source,
    on=["Source_ID", "Records"],
    how="left"
)


# ============================================================
# 6. IDENTIFY MOST SHIFTED FEATURES
# ============================================================

print()
print("=" * 70)
print("1. SOURCE-LEVEL DOMAIN SHIFT SUMMARY")
print("=" * 70)

display_columns = [
    "Source_ID",
    "Records",
    "Process_Mean_Absolute_Z",
    "Process_Max_Absolute_Z",
    "Features_Shifted_GE_2SD",
    "Features_Outside_Training_Range",
    "Target_Mean_Absolute_Z",
]

print(
    source_summary[
        display_columns
    ].sort_values(
        "Process_Mean_Absolute_Z",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# 7. MOST SHIFTED FEATURE PER SOURCE
# ============================================================

print()
print("=" * 70)
print("2. MOST SHIFTED PROCESS FEATURES")
print("=" * 70)

most_shifted_rows = []

for source in test_sources:

    subset = feature_shift[
        feature_shift["Source_ID"] == source
    ].copy()

    subset = subset.sort_values(
        "Mean_Absolute_Z",
        ascending=False
    )

    top = subset.iloc[0]

    most_shifted_rows.append({
        "Source_ID": source,
        "Most_Shifted_Feature": top["Feature"],
        "Mean_Absolute_Z": top["Mean_Absolute_Z"],
        "Max_Absolute_Z": top["Max_Absolute_Z"],
        "Range_Flag": top["Range_Flag"],
    })

most_shifted = pd.DataFrame(most_shifted_rows)

print(
    most_shifted.to_string(index=False)
)


# ============================================================
# 8. TARGET SHIFT
# ============================================================

print()
print("=" * 70)
print("3. TARGET DISTRIBUTION SHIFT")
print("=" * 70)

target_display = target_shift[
    [
        "Source_ID",
        "Target",
        "Source_Mean",
        "Training_Mean",
        "Mean_Z_vs_Training",
        "Source_Min",
        "Source_Max",
        "Range_Flag",
    ]
]

print(
    target_display.to_string(index=False)
)


# ============================================================
# 9. MODEL ERROR VS DOMAIN SHIFT
# ============================================================

print()
print("=" * 70)
print("4. ERROR VS DOMAIN SHIFT")
print("=" * 70)

comparison = source_summary.copy()

for model_name, target_columns in MODEL_ERROR_COLUMNS.items():

    prefix = model_name.replace(" ", "_")

    error_cols = [
        f"{prefix}_{target}_MAE"
        for target in target_columns
        if f"{prefix}_{target}_MAE" in comparison.columns
    ]

    if error_cols:
        comparison[
            f"{prefix}_Mean_Target_MAE"
        ] = comparison[error_cols].mean(axis=1)


comparison_display_columns = [
    "Source_ID",
    "Records",
    "Process_Mean_Absolute_Z",
    "Process_Max_Absolute_Z",
    "Features_Shifted_GE_2SD",
]

for col in comparison.columns:
    if "Mean_Target_MAE" in col:
        comparison_display_columns.append(col)

print(
    comparison[
        comparison_display_columns
    ].sort_values(
        "Process_Mean_Absolute_Z",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# 10. CORRELATION BETWEEN SHIFT AND ERROR
# ============================================================

print()
print("=" * 70)
print("5. SHIFT / ERROR CORRELATION")
print("=" * 70)

correlation_rows = []

shift_metrics = [
    "Process_Mean_Absolute_Z",
    "Process_Max_Absolute_Z",
    "Features_Shifted_GE_2SD",
    "Features_Outside_Training_Range",
]

error_metrics = [
    col
    for col in comparison.columns
    if "Mean_Target_MAE" in col
]

for shift_metric in shift_metrics:

    for error_metric in error_metrics:

        valid = comparison[
            [shift_metric, error_metric]
        ].dropna()

        if len(valid) >= 3:
            corr = valid[
                shift_metric
            ].corr(valid[error_metric])
        else:
            corr = np.nan

        correlation_rows.append({
            "Shift_Metric": shift_metric,
            "Error_Metric": error_metric,
            "Sources_Used": len(valid),
            "Pearson_Correlation": corr,
        })


correlation_df = pd.DataFrame(correlation_rows)

print(
    correlation_df.to_string(index=False)
)


# ============================================================
# 11. SOURCE RANKING BY DIAGNOSTIC SEVERITY
# ============================================================

print()
print("=" * 70)
print("6. SOURCE DIAGNOSTIC PRIORITY")
print("=" * 70)

priority = source_summary.copy()

# Diagnostic score only.
# This is NOT a model quality score and must not be interpreted
# as a ranking of scientific quality.

priority["Domain_Shift_Indicator"] = (
    priority["Process_Mean_Absolute_Z"]
    + 0.5 * priority["Features_Shifted_GE_2SD"]
    + 0.5 * priority["Features_Outside_Training_Range"]
)

priority_display = priority[
    [
        "Source_ID",
        "Records",
        "Process_Mean_Absolute_Z",
        "Features_Shifted_GE_2SD",
        "Features_Outside_Training_Range",
        "Domain_Shift_Indicator",
    ]
].sort_values(
    "Domain_Shift_Indicator",
    ascending=False
)

print(
    priority_display.to_string(index=False)
)


# ============================================================
# 12. INTERPRETATION
# ============================================================

print()
print("=" * 70)
print("7. AUTOMATIC INTERPRETATION")
print("=" * 70)

highest_shift_source = source_summary.loc[
    source_summary["Process_Mean_Absolute_Z"].idxmax()
]

print(
    f"Highest process-distribution shift: "
    f"{highest_shift_source['Source_ID']}"
)

print(
    f"Mean absolute process Z-score: "
    f"{highest_shift_source['Process_Mean_Absolute_Z']:.3f}"
)

highest_shift_features = feature_shift[
    feature_shift["Mean_Absolute_Z"]
    >= 2
]

print(
    f"Total source-feature combinations shifted "
    f"by >=2 SD: {len(highest_shift_features)}"
)

outside_range = feature_shift[
    feature_shift["Range_Flag"]
    != "WITHIN_TRAINING_RANGE"
]

print(
    f"Total source-feature combinations outside "
    f"training range: {len(outside_range)}"
)

print()
print("Interpretation rules:")
print(" - Z-score measures distribution shift relative to training data.")
print(" - >=2 SD is treated as a diagnostic warning, not an automatic error.")
print(" - Outside-range values indicate extrapolation in that feature.")
print(" - This analysis does not prove causation.")
print(" - Source differences may also reflect unmeasured variables.")


# ============================================================
# 13. SAVE FILES
# ============================================================

feature_shift.to_csv(
    OUTPUT_FEATURE,
    index=False
)

target_shift.to_csv(
    OUTPUT_TARGET,
    index=False
)

error_by_source.to_csv(
    OUTPUT_ERROR,
    index=False
)

source_summary.to_csv(
    OUTPUT_SOURCE,
    index=False
)

summary_rows = []

for _, row in source_summary.iterrows():

    source = row["Source_ID"]

    top_feature_row = most_shifted[
        most_shifted["Source_ID"] == source
    ].iloc[0]

    summary_rows.append({
        "Source_ID": source,
        "Records": row["Records"],

        "Process_Mean_Absolute_Z":
            row["Process_Mean_Absolute_Z"],

        "Process_Max_Absolute_Z":
            row["Process_Max_Absolute_Z"],

        "Features_Shifted_GE_2SD":
            row["Features_Shifted_GE_2SD"],

        "Features_Outside_Training_Range":
            row["Features_Outside_Training_Range"],

        "Most_Shifted_Feature":
            top_feature_row["Most_Shifted_Feature"],

        "Most_Shifted_Feature_Z":
            top_feature_row["Mean_Absolute_Z"],

        "Target_Mean_Absolute_Z":
            row["Target_Mean_Absolute_Z"],

        "Target_Max_Absolute_Z":
            row["Target_Max_Absolute_Z"],
    })

summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(OUTPUT_SOURCE)
print(OUTPUT_FEATURE)
print(OUTPUT_TARGET)
print(OUTPUT_ERROR)
print(OUTPUT_SUMMARY)

print()
print("=" * 70)
print("STEP 9I COMPLETE")
print("=" * 70)

print(
    f"Held-out sources analyzed : {len(test_sources)}"
)

print(
    "No records were removed."
)

print(
    "No target values were modified."
)

print(
    "No models were retrained."
)

print(
    "No dataset files were modified."
)

print("=" * 70)