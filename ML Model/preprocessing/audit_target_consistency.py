from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PRINTOPT AI — STEP 9A
# TARGET CONSISTENCY & DATA-QUALITY AUDIT
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9A")
print("TARGET CONSISTENCY & DATA-QUALITY AUDIT")
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

OUTPUT_DIR = BASE_DIR / "evaluation"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

AUDIT_PATH = (
    OUTPUT_DIR
    / "target_consistency_audit.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ------------------------------------------------------------
# IMPORTANT COLUMNS
# ------------------------------------------------------------

PROCESS_COLUMNS = [
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

IDENTIFICATION_COLUMNS = [
    "Experiment_ID",
    "Source_ID",
]


# ------------------------------------------------------------
# BASIC DATASET INFORMATION
# ------------------------------------------------------------

print()
print("DATASET")
print("-" * 70)

print(
    f"Records : {len(df)}"
)

print(
    f"Columns : {len(df.columns)}"
)

print(
    f"Sources : {df['Source_ID'].nunique()}"
)


# ============================================================
# AUDIT 1 — MISSING VALUES
# ============================================================

print()
print("=" * 70)
print("AUDIT 1 — MISSING VALUES")
print("=" * 70)

missing = df.isna().sum()

missing = missing[
    missing > 0
]

if len(missing) == 0:

    print()
    print("No missing values detected.")

else:

    print()
    print(missing.to_string())


# ============================================================
# AUDIT 2 — NON-FINITE NUMERIC VALUES
# ============================================================

print()
print("=" * 70)
print("AUDIT 2 — NON-FINITE NUMERIC VALUES")
print("=" * 70)

numeric_columns = (
    PROCESS_COLUMNS
    + TARGET_COLUMNS
)

nonfinite_rows = []

for column in numeric_columns:

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    count = (
        ~np.isfinite(values)
    ).sum()

    if count > 0:

        nonfinite_rows.append(
            {
                "Column": column,
                "Nonfinite_Count": int(count),
            }
        )

if len(nonfinite_rows) == 0:

    print()
    print("No non-finite numeric values detected.")

else:

    print(
        pd.DataFrame(
            nonfinite_rows
        ).to_string(index=False)
    )


# ============================================================
# AUDIT 3 — YS > UTS
# ============================================================

print()
print("=" * 70)
print("AUDIT 3 — YIELD STRENGTH GREATER THAN UTS")
print("=" * 70)

ys_gt_uts = (
    df["YS_MPa"]
    > df["UTS_MPa"]
)

ys_gt_uts_df = df[
    ys_gt_uts
].copy()

print()
print(
    f"Records where YS > UTS : "
    f"{len(ys_gt_uts_df)}"
)


if len(ys_gt_uts_df) > 0:

    display_columns = [
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

    print()

    print(
        ys_gt_uts_df[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# AUDIT 4 — YS/UTS RATIO
# ============================================================

print()
print("=" * 70)
print("AUDIT 4 — YS / UTS RATIO")
print("=" * 70)

df["YS_UTS_Ratio"] = (
    df["YS_MPa"]
    / df["UTS_MPa"]
)

print()
print(
    "Ratio statistics:"
)

print(
    df["YS_UTS_Ratio"]
    .describe()
    .to_string()
)

high_ratio = (
    df["YS_UTS_Ratio"]
    > 1.0
)

print()
print(
    f"YS/UTS ratio > 1.0 : "
    f"{high_ratio.sum()}"
)

print(
    f"YS/UTS ratio > 0.95: "
    f"{(df['YS_UTS_Ratio'] > 0.95).sum()}"
)


# ============================================================
# AUDIT 5 — TARGET RANGE CHECK
# ============================================================

print()
print("=" * 70)
print("AUDIT 5 — TARGET RANGES")
print("=" * 70)

for target in TARGET_COLUMNS:

    print()
    print(target)

    print(
        f"  Minimum : "
        f"{df[target].min():.3f}"
    )

    print(
        f"  Maximum : "
        f"{df[target].max():.3f}"
    )

    print(
        f"  Mean    : "
        f"{df[target].mean():.3f}"
    )

    print(
        f"  Median  : "
        f"{df[target].median():.3f}"
    )

    print(
        f"  Std     : "
        f"{df[target].std():.3f}"
    )


# ============================================================
# AUDIT 6 — PROCESS PARAMETER RANGES
# ============================================================

print()
print("=" * 70)
print("AUDIT 6 — PROCESS PARAMETER RANGES")
print("=" * 70)

for feature in PROCESS_COLUMNS:

    print()
    print(feature)

    print(
        f"  Minimum : "
        f"{df[feature].min():.3f}"
    )

    print(
        f"  Maximum : "
        f"{df[feature].max():.3f}"
    )

    print(
        f"  Mean    : "
        f"{df[feature].mean():.3f}"
    )

    print(
        f"  Median  : "
        f"{df[feature].median():.3f}"
    )


# ============================================================
# AUDIT 7 — EXTREME PROCESS DOMAIN
# ============================================================

print()
print("=" * 70)
print("AUDIT 7 — EXTREME PROCESS-DOMAIN RECORDS")
print("=" * 70)


# VED above the 95th percentile
ved_95 = df[
    "VED_J_mm3"
].quantile(0.95)

high_ved = df[
    df["VED_J_mm3"] > ved_95
].copy()

print()
print(
    f"VED 95th percentile : "
    f"{ved_95:.3f}"
)

print(
    f"Records above it    : "
    f"{len(high_ved)}"
)

if len(high_ved) > 0:

    print()

    print(
        high_ved[
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
        ]
        .sort_values(
            "VED_J_mm3",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# AUDIT 8 — EXACT DUPLICATE PROCESS CONDITIONS
# ============================================================

print()
print("=" * 70)
print("AUDIT 8 — REPEATED PROCESS CONDITIONS")
print("=" * 70)

condition_columns = [
    "Powder_Size_um",
    "Laser_Spot_um",
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Hatch_Distance_um",
    "Layer_Thickness_um",
]

condition_counts = (
    df
    .groupby(condition_columns)
    .size()
    .reset_index(
        name="Record_Count"
    )
)

repeated_conditions = (
    condition_counts[
        condition_counts["Record_Count"] > 1
    ]
    .sort_values(
        "Record_Count",
        ascending=False
    )
)

print()
print(
    f"Unique process conditions : "
    f"{len(condition_counts)}"
)

print(
    f"Repeated process conditions: "
    f"{len(repeated_conditions)}"
)

if len(repeated_conditions) > 0:

    print()

    print(
        repeated_conditions
        .to_string(index=False)
    )


# ============================================================
# AUDIT 9 — EXACT DUPLICATE FULL RECORDS
# ============================================================

print()
print("=" * 70)
print("AUDIT 9 — DUPLICATE RECORD CHECK")
print("=" * 70)

duplicate_mask = (
    df
    .drop(
        columns=[
            "Experiment_ID",
            "YS_UTS_Ratio",
        ],
        errors="ignore"
    )
    .duplicated(
        keep=False
    )
)

duplicate_count = (
    duplicate_mask.sum()
)

print()
print(
    f"Duplicate rows detected : "
    f"{duplicate_count}"
)


# ============================================================
# AUDIT 10 — SOURCE-LEVEL YS > UTS
# ============================================================

print()
print("=" * 70)
print("AUDIT 10 — YS > UTS BY SOURCE")
print("=" * 70)

source_ys_gt_uts = (
    df
    .groupby("Source_ID")
    .agg(
        Records=(
            "Experiment_ID",
            "count"
        ),
        YS_GT_UTS=(
            "YS_UTS_Ratio",
            lambda x: (x > 1.0).sum()
        ),
        Mean_YS_UTS_Ratio=(
            "YS_UTS_Ratio",
            "mean"
        ),
    )
)

source_ys_gt_uts = (
    source_ys_gt_uts[
        source_ys_gt_uts["YS_GT_UTS"] > 0
    ]
    .sort_values(
        "YS_GT_UTS",
        ascending=False
    )
)

if len(source_ys_gt_uts) == 0:

    print()
    print(
        "No source contains YS > UTS."
    )

else:

    print()

    print(
        source_ys_gt_uts
        .to_string()
    )


# ============================================================
# AUDIT 11 — TARGET CORRELATION
# ============================================================

print()
print("=" * 70)
print("AUDIT 11 — TARGET CORRELATION")
print("=" * 70)

target_correlation = (
    df[TARGET_COLUMNS]
    .corr()
)

print()
print(
    target_correlation
    .to_string()
)


# ============================================================
# AUDIT 12 — FLAG RECORDS
# ============================================================

print()
print("=" * 70)
print("AUDIT 12 — RECORD FLAGS")
print("=" * 70)


audit = df[
    [
        "Experiment_ID",
        "Source_ID",
        *PROCESS_COLUMNS,
        *TARGET_COLUMNS,
        "YS_UTS_Ratio",
    ]
].copy()


audit["FLAG_YS_GT_UTS"] = (
    audit["YS_MPa"]
    > audit["UTS_MPa"]
)


audit["FLAG_HIGH_VED"] = (
    audit["VED_J_mm3"]
    > ved_95
)


audit["FLAG_ANY"] = (
    audit["FLAG_YS_GT_UTS"]
    | audit["FLAG_HIGH_VED"]
)


flagged = audit[
    audit["FLAG_ANY"]
].copy()


print()
print(
    f"Total flagged records : "
    f"{len(flagged)}"
)

print(
    f"  YS > UTS flags       : "
    f"{audit['FLAG_YS_GT_UTS'].sum()}"
)

print(
    f"  High VED flags       : "
    f"{audit['FLAG_HIGH_VED'].sum()}"
)


# ------------------------------------------------------------
# SAVE AUDIT
# ------------------------------------------------------------

audit.to_csv(
    AUDIT_PATH,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    AUDIT_PATH
)

print()
print("IMPORTANT")
print("-" * 70)

print(
    "No records were deleted or modified."
)

print(
    "Flags are diagnostic only."
)

print()
print("STEP 9A COMPLETE")
print("=" * 70)