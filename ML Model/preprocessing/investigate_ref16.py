from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PRINTOPT AI — STEP 9H
# REF_16 SOURCE INVESTIGATION
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9H")
print("REF_16 SOURCE / DOMAIN INVESTIGATION")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ti64_lpbf_training_candidate.csv"
)

SOURCE_ERROR_FILE = (
    BASE_DIR
    / "evaluation"
    / "baseline_source_error_v2.csv"
)

EXPERIMENT_ERROR_FILE = (
    BASE_DIR
    / "evaluation"
    / "baseline_experiment_error_v2.csv"
)

OUTPUT_DIR = BASE_DIR / "evaluation"

REF16_RECORDS_FILE = (
    OUTPUT_DIR
    / "ref16_records_v2.csv"
)

REF16_COMPARISON_FILE = (
    OUTPUT_DIR
    / "ref16_vs_training_comparison_v2.csv"
)

REF16_ZSCORE_FILE = (
    OUTPUT_DIR
    / "ref16_feature_target_zscores_v2.csv"
)

REF16_SUMMARY_FILE = (
    OUTPUT_DIR
    / "ref16_investigation_summary_v2.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print()
print("INPUT DATA")
print("-" * 70)
print(f"Records : {len(df)}")
print(f"Sources : {df['Source_ID'].nunique()}")


# ------------------------------------------------------------
# COLUMN DEFINITIONS
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


# ============================================================
# 1. REF_16 RECORDS
# ============================================================

print()
print("=" * 70)
print("1. REF_16 RECORDS")
print("=" * 70)


ref16 = df[
    df["Source_ID"] == "REF_16"
].copy()

training_sources = df[
    df["Source_ID"] != "REF_16"
].copy()


print()
print(f"REF_16 records : {len(ref16)}")

if len(ref16) == 0:
    raise ValueError(
        "REF_16 was not found in the training candidate dataset."
    )


# ------------------------------------------------------------
# SAVE FULL REF_16 RECORDS
# ------------------------------------------------------------

ref16_display_columns = [
    "Experiment_ID",
    "Source_ID",
    *FEATURES,
    *TARGETS,
]

if "Reference" in ref16.columns:
    ref16_display_columns.append("Reference")

ref16[
    ref16_display_columns
].sort_values(
    "Experiment_ID"
).to_csv(
    REF16_RECORDS_FILE,
    index=False,
)


print()
print(
    ref16[
        ref16_display_columns
    ].sort_values(
        "Experiment_ID"
    ).to_string(index=False)
)


# ============================================================
# 2. REF_16 DESCRIPTIVE STATISTICS
# ============================================================

print()
print("=" * 70)
print("2. REF_16 DESCRIPTIVE STATISTICS")
print("=" * 70)


ref16_stats = ref16[
    FEATURES + TARGETS
].describe().T[
    [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
]

print()
print(
    ref16_stats.to_string()
)


# ============================================================
# 3. TRAINING DATA DESCRIPTIVE STATISTICS
# ============================================================

print()
print("=" * 70)
print("3. NON-REF_16 TRAINING DATA STATISTICS")
print("=" * 70)


training_stats = training_sources[
    FEATURES + TARGETS
].describe().T[
    [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
]

print()
print(
    training_stats.to_string()
)


# ============================================================
# 4. RANGE COMPARISON
# ============================================================

print()
print("=" * 70)
print("4. REF_16 VS TRAINING RANGE COMPARISON")
print("=" * 70)


comparison_rows = []


for column in FEATURES + TARGETS:

    ref_min = ref16[column].min()
    ref_max = ref16[column].max()

    train_min = training_sources[column].min()
    train_max = training_sources[column].max()

    ref_mean = ref16[column].mean()
    train_mean = training_sources[column].mean()

    ref_std = ref16[column].std()
    train_std = training_sources[column].std()

    # Determine whether REF_16 has values outside
    # the range observed in the other training sources.
    below_count = int(
        (ref16[column] < train_min).sum()
    )

    above_count = int(
        (ref16[column] > train_max).sum()
    )

    outside_count = (
        below_count + above_count
    )

    # Mean difference expressed relative to
    # the training standard deviation.
    if train_std > 0:
        mean_z = (
            ref_mean - train_mean
        ) / train_std
    else:
        mean_z = np.nan

    comparison_rows.append(
        {
            "Variable": column,
            "REF16_Min": ref_min,
            "REF16_Max": ref_max,
            "Training_Min": train_min,
            "Training_Max": train_max,
            "REF16_Mean": ref_mean,
            "Training_Mean": train_mean,
            "REF16_Std": ref_std,
            "Training_Std": train_std,
            "REF16_Values_Below_Training_Min": below_count,
            "REF16_Values_Above_Training_Max": above_count,
            "REF16_Values_Outside_Training_Range": outside_count,
            "REF16_Mean_Z_vs_Training": mean_z,
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    REF16_COMPARISON_FILE,
    index=False,
)


print()
print(
    comparison_df.to_string(index=False)
)


# ============================================================
# 5. REF_16 Z-SCORE ANALYSIS
# ============================================================

print()
print("=" * 70)
print("5. REF_16 STANDARDIZED DISTANCE FROM TRAINING DISTRIBUTION")
print("=" * 70)


zscore_rows = []


for column in FEATURES + TARGETS:

    train_mean = training_sources[
        column
    ].mean()

    train_std = training_sources[
        column
    ].std()

    if train_std == 0:
        zscores = np.zeros(
            len(ref16)
        )
    else:
        zscores = (
            ref16[column].values
            - train_mean
        ) / train_std

    abs_zscores = np.abs(
        zscores
    )

    zscore_rows.append(
        {
            "Variable": column,
            "Max_Absolute_Z": abs_zscores.max(),
            "Mean_Absolute_Z": abs_zscores.mean(),
            "Count_Abs_Z_Greater_2": int(
                (abs_zscores > 2).sum()
            ),
            "Count_Abs_Z_Greater_3": int(
                (abs_zscores > 3).sum()
            ),
        }
    )


zscore_df = pd.DataFrame(
    zscore_rows
)

zscore_df.to_csv(
    REF16_ZSCORE_FILE,
    index=False,
)


print()
print(
    zscore_df.to_string(index=False)
)


# ============================================================
# 6. PROCESS-PARAMETER COMPARISON
# ============================================================

print()
print("=" * 70)
print("6. PROCESS PARAMETER COMPARISON")
print("=" * 70)


process_features = [
    "Powder_Size_um",
    "Laser_Spot_um",
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Hatch_Distance_um",
    "Layer_Thickness_um",
    "VED_J_mm3",
]


for feature in process_features:

    print()
    print(f"{feature}")
    print("-" * 70)

    print(
        f"REF_16 mean : "
        f"{ref16[feature].mean():.4f}"
    )

    print(
        f"Other-source training mean : "
        f"{training_sources[feature].mean():.4f}"
    )

    print(
        f"REF_16 range : "
        f"{ref16[feature].min():.4f} "
        f"to "
        f"{ref16[feature].max():.4f}"
    )

    print(
        f"Training range : "
        f"{training_sources[feature].min():.4f} "
        f"to "
        f"{training_sources[feature].max():.4f}"
    )


# ============================================================
# 7. TARGET DISTRIBUTION COMPARISON
# ============================================================

print()
print("=" * 70)
print("7. TARGET DISTRIBUTION COMPARISON")
print("=" * 70)


for target in TARGETS:

    print()
    print(target)
    print("-" * 70)

    print(
        f"REF_16 mean : "
        f"{ref16[target].mean():.4f}"
    )

    print(
        f"Other-source training mean : "
        f"{training_sources[target].mean():.4f}"
    )

    print(
        f"REF_16 range : "
        f"{ref16[target].min():.4f} "
        f"to "
        f"{ref16[target].max():.4f}"
    )

    print(
        f"Training range : "
        f"{training_sources[target].min():.4f} "
        f"to "
        f"{training_sources[target].max():.4f}"
    )


# ============================================================
# 8. SOURCE-LEVEL COMPARISON
# ============================================================

print()
print("=" * 70)
print("8. REF_16 VS OTHER SOURCES")
print("=" * 70)


source_summary = (
    df.groupby("Source_ID")
    .agg(
        Records=("Experiment_ID", "count"),
        Powder_Mean=("Powder_Size_um", "mean"),
        Spot_Mean=("Laser_Spot_um", "mean"),
        Power_Mean=("Laser_Power_W", "mean"),
        Speed_Mean=("Scanning_Speed_mm_s", "mean"),
        Hatch_Mean=("Hatch_Distance_um", "mean"),
        Layer_Mean=("Layer_Thickness_um", "mean"),
        VED_Mean=("VED_J_mm3", "mean"),
        UTS_Mean=("UTS_MPa", "mean"),
        YS_Mean=("YS_MPa", "mean"),
        Elongation_Mean=("Elongation_pct", "mean"),
    )
    .sort_values(
        "Records",
        ascending=False,
    )
)


print()
print(
    source_summary.to_string()
)


# ============================================================
# 9. REF_16 ERROR DATA FROM STEP 9G
# ============================================================

print()
print("=" * 70)
print("9. REF_16 MODEL ERROR FROM STEP 9G")
print("=" * 70)


if EXPERIMENT_ERROR_FILE.exists():

    experiment_error = pd.read_csv(
        EXPERIMENT_ERROR_FILE
    )

    ref16_errors = experiment_error[
        experiment_error["Source_ID"] == "REF_16"
    ].copy()

    print()

    error_columns = [
        "Experiment_ID",
        "Source_ID",
        "VED_J_mm3",
        "XGB_UTS_MPa_AbsError",
        "XGB_YS_MPa_AbsError",
        "XGB_Elongation_pct_AbsError",
        "XGB_Mean_Absolute_Error",
    ]

    available_error_columns = [
        column
        for column in error_columns
        if column in ref16_errors.columns
    ]

    print(
        ref16_errors[
            available_error_columns
        ].sort_values(
            "Experiment_ID"
        ).to_string(index=False)
    )

else:

    print(
        "Step 9G experiment-error file was not found."
    )


# ============================================================
# 10. REF_16 INTERNAL VARIATION
# ============================================================

print()
print("=" * 70)
print("10. REF_16 INTERNAL VARIATION")
print("=" * 70)


print()
print("Coefficient of variation (CV)")
print("-" * 70)

for column in FEATURES + TARGETS:

    mean = ref16[column].mean()
    std = ref16[column].std()

    if mean != 0:
        cv = (
            std / abs(mean)
        ) * 100
    else:
        cv = np.nan

    print(
        f"{column:<25} CV = {cv:8.2f}%"
    )


# ============================================================
# 11. SUMMARY FLAGS
# ============================================================

print()
print("=" * 70)
print("11. DIAGNOSTIC FLAGS")
print("=" * 70)


summary_rows = []


for _, row in comparison_df.iterrows():

    variable = row["Variable"]

    outside = int(
        row[
            "REF16_Values_Outside_Training_Range"
        ]
    )

    mean_z = row[
        "REF16_Mean_Z_vs_Training"
    ]

    if outside > 0:
        range_flag = (
            "OUTSIDE_TRAINING_RANGE"
        )
    else:
        range_flag = (
            "WITHIN_TRAINING_RANGE"
        )

    if pd.isna(mean_z):
        distribution_flag = "UNDEFINED"

    elif abs(mean_z) >= 3:
        distribution_flag = (
            "STRONGLY_SHIFTED"
        )

    elif abs(mean_z) >= 2:
        distribution_flag = (
            "SHIFTED"
        )

    elif abs(mean_z) >= 1:
        distribution_flag = (
            "MODERATELY_SHIFTED"
        )

    else:
        distribution_flag = (
            "CLOSE_TO_TRAINING_MEAN"
        )

    summary_rows.append(
        {
            "Variable": variable,
            "Range_Flag": range_flag,
            "Distribution_Flag": distribution_flag,
            "Values_Outside_Training_Range": outside,
            "Mean_Z_vs_Training": mean_z,
        }
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    REF16_SUMMARY_FILE,
    index=False,
)


print()
print(
    summary_df.to_string(index=False)
)


# ============================================================
# 12. FINAL INTERPRETATION FLAGS
# ============================================================

print()
print("=" * 70)
print("12. FINAL REF_16 DIAGNOSTIC INDICATORS")
print("=" * 70)


feature_comparison = comparison_df[
    comparison_df["Variable"].isin(
        FEATURES
    )
]

target_comparison = comparison_df[
    comparison_df["Variable"].isin(
        TARGETS
    )
]


outside_feature_count = int(
    (
        feature_comparison[
            "REF16_Values_Outside_Training_Range"
        ]
        > 0
    ).sum()
)

outside_target_count = int(
    (
        target_comparison[
            "REF16_Values_Outside_Training_Range"
        ]
        > 0
    ).sum()
)


strong_shift_count = int(
    (
        feature_comparison[
            "REF16_Mean_Z_vs_Training"
        ].abs()
        >= 2
    ).sum()
)


print(
    f"REF_16 records                  : "
    f"{len(ref16)}"
)

print(
    f"Process features outside range  : "
    f"{outside_feature_count} / {len(FEATURES)}"
)

print(
    f"Targets outside range           : "
    f"{outside_target_count} / {len(TARGETS)}"
)

print(
    f"Process features shifted >=2 SD : "
    f"{strong_shift_count} / {len(FEATURES)}"
)


# ------------------------------------------------------------
# REF_16 TARGET RELATIONSHIPS
# ------------------------------------------------------------

print()
print("REF_16 target correlations")
print("-" * 70)

print(
    ref16[
        TARGETS
    ].corr().to_string()
)


# ------------------------------------------------------------
# REF_16 PROCESS-TARGET CORRELATIONS
# ------------------------------------------------------------

print()
print("REF_16 process-target correlations")
print("-" * 70)

print(
    ref16[
        FEATURES + TARGETS
    ].corr()[
        TARGETS
    ].loc[
        FEATURES
    ].to_string()
)


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    REF16_RECORDS_FILE
)

print(
    REF16_COMPARISON_FILE
)

print(
    REF16_ZSCORE_FILE
)

print(
    REF16_SUMMARY_FILE
)


print()
print("=" * 70)
print("STEP 9H COMPLETE")
print("=" * 70)

print(
    "No records were removed."
)

print(
    "No target values were modified."
)

print(
    "No models were retrained."
)