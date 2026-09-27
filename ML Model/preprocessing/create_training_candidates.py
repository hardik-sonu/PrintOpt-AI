from pathlib import Path
import pandas as pd


# ============================================================
# PRINTOPT AI — STEP 9D
# CREATE CONTROLLED TRAINING CANDIDATE DATASETS
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9D")
print("CREATE CONTROLLED TRAINING CANDIDATE DATASETS")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ti64_lpbf_verified.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

TRAINING_CANDIDATE_FILE = (
    OUTPUT_DIR
    / "ti64_lpbf_training_candidate.csv"
)

SENSITIVITY_FILE = (
    OUTPUT_DIR
    / "ti64_lpbf_full_sensitivity.csv"
)

SUMMARY_FILE = (
    BASE_DIR
    / "evaluation"
    / "training_candidate_summary.csv"
)


# ------------------------------------------------------------
# LOAD VERIFIED DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print()
print("INPUT DATASET")
print("-" * 70)
print(f"Records : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"Sources : {df['Source_ID'].nunique()}")


# ------------------------------------------------------------
# CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "Experiment_ID",
    "Source_ID",
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
    "VED_J_mm3",
    "Data_Status",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )


# ------------------------------------------------------------
# DATASET A
# CONSERVATIVE TRAINING CANDIDATE
#
# Exclude only records explicitly marked
# REVIEW_REQUIRED.
#
# Nothing is deleted from the original dataset.
# ------------------------------------------------------------

training_candidate = df[
    df["Data_Status"] != "REVIEW_REQUIRED"
].copy()

training_candidate.reset_index(drop=True, inplace=True)


# ------------------------------------------------------------
# DATASET B
# FULL SENSITIVITY DATASET
#
# Keep all records.
#
# This dataset is NOT the primary training dataset.
# It is retained for later sensitivity analysis.
# ------------------------------------------------------------

full_sensitivity = df.copy()
full_sensitivity.reset_index(drop=True, inplace=True)


# ------------------------------------------------------------
# VERIFY TRAINING CANDIDATE
# ------------------------------------------------------------

print()
print("DATASET A — CONSERVATIVE TRAINING CANDIDATE")
print("-" * 70)

print(
    f"Records retained : {len(training_candidate)}"
)

print(
    f"Records excluded : "
    f"{len(df) - len(training_candidate)}"
)

print(
    f"Sources retained : "
    f"{training_candidate['Source_ID'].nunique()}"
)


# ------------------------------------------------------------
# VERIFY NO REVIEW_REQUIRED RECORDS REMAIN
# ------------------------------------------------------------

remaining_review = training_candidate[
    training_candidate["Data_Status"] == "REVIEW_REQUIRED"
]

if len(remaining_review) != 0:
    raise ValueError(
        "ERROR: REVIEW_REQUIRED records remain "
        "inside the training candidate."
    )


# ------------------------------------------------------------
# CHECK TARGETS
# ------------------------------------------------------------

target_columns = [
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
]

missing_target_values = (
    training_candidate[target_columns]
    .isna()
    .sum()
)

if missing_target_values.sum() != 0:
    raise ValueError(
        "ERROR: Missing target values found "
        "in training candidate."
    )


# ------------------------------------------------------------
# CHECK PHYSICAL CONSISTENCY
# ------------------------------------------------------------

ys_gt_uts = (
    training_candidate["YS_MPa"]
    > training_candidate["UTS_MPa"]
)

print()
print("PHYSICAL CONSISTENCY CHECK")
print("-" * 70)
print(
    f"Training records with YS > UTS : "
    f"{ys_gt_uts.sum()}"
)

if ys_gt_uts.sum() != 0:
    raise ValueError(
        "ERROR: YS > UTS record remains "
        "in the conservative training dataset."
    )


# ------------------------------------------------------------
# SOURCE DISTRIBUTION
# ------------------------------------------------------------

print()
print("SOURCE DISTRIBUTION")
print("-" * 70)

source_counts = (
    training_candidate["Source_ID"]
    .value_counts()
    .sort_index()
)

print(source_counts.to_string())


# ------------------------------------------------------------
# TARGET STATISTICS
# ------------------------------------------------------------

print()
print("TARGET STATISTICS — TRAINING CANDIDATE")
print("-" * 70)

for target in target_columns:
    series = training_candidate[target]

    print(
        f"{target:20s} "
        f"min={series.min():.3f}  "
        f"max={series.max():.3f}  "
        f"mean={series.mean():.3f}  "
        f"std={series.std():.3f}"
    )


# ------------------------------------------------------------
# FEATURE / TARGET RANGES
# ------------------------------------------------------------

feature_columns = [
    "Powder_Size_um",
    "Laser_Spot_um",
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Hatch_Distance_um",
    "Layer_Thickness_um",
    "VED_J_mm3",
]

print()
print("FEATURE RANGES — TRAINING CANDIDATE")
print("-" * 70)

for feature in feature_columns:
    series = training_candidate[feature]

    print(
        f"{feature:25s} "
        f"min={series.min():.3f}  "
        f"max={series.max():.3f}"
    )


# ------------------------------------------------------------
# SOURCE OVERLAP INFORMATION
# ------------------------------------------------------------

original_sources = set(df["Source_ID"])
candidate_sources = set(
    training_candidate["Source_ID"]
)

excluded_sources = sorted(
    original_sources - candidate_sources
)

print()
print("SOURCE COVERAGE")
print("-" * 70)

print(
    f"Original sources       : "
    f"{len(original_sources)}"
)

print(
    f"Training sources       : "
    f"{len(candidate_sources)}"
)

print(
    f"Completely excluded sources : "
    f"{len(excluded_sources)}"
)

if excluded_sources:
    print(
        "Excluded source IDs:"
    )
    print(
        ", ".join(excluded_sources)
    )


# ------------------------------------------------------------
# CREATE SUMMARY TABLE
# ------------------------------------------------------------

summary_rows = [
    {
        "Dataset": "Original processed",
        "Records": len(df),
        "Sources": df["Source_ID"].nunique(),
        "Review_Required": int(
            (
                df["Data_Status"]
                == "REVIEW_REQUIRED"
            ).sum()
        ),
        "YS_GT_UTS": int(
            (
                df["YS_MPa"]
                > df["UTS_MPa"]
            ).sum()
        ),
        "Purpose": (
            "Original processed dataset; "
            "preserved unchanged."
        ),
    },
    {
        "Dataset": "Conservative training candidate",
        "Records": len(training_candidate),
        "Sources": training_candidate[
            "Source_ID"
        ].nunique(),
        "Review_Required": int(
            (
                training_candidate["Data_Status"]
                == "REVIEW_REQUIRED"
            ).sum()
        ),
        "YS_GT_UTS": int(
            (
                training_candidate["YS_MPa"]
                > training_candidate["UTS_MPa"]
            ).sum()
        ),
        "Purpose": (
            "Primary candidate for model "
            "training and validation."
        ),
    },
    {
        "Dataset": "Full sensitivity",
        "Records": len(full_sensitivity),
        "Sources": full_sensitivity[
            "Source_ID"
        ].nunique(),
        "Review_Required": int(
            (
                full_sensitivity["Data_Status"]
                == "REVIEW_REQUIRED"
            ).sum()
        ),
        "YS_GT_UTS": int(
            (
                full_sensitivity["YS_MPa"]
                > full_sensitivity["UTS_MPa"]
            ).sum()
        ),
        "Purpose": (
            "Sensitivity analysis only; "
            "all records retained."
        ),
    },
]

summary_df = pd.DataFrame(summary_rows)


# ------------------------------------------------------------
# SAVE FILES
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUMMARY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

training_candidate.to_csv(
    TRAINING_CANDIDATE_FILE,
    index=False
)

full_sensitivity.to_csv(
    SENSITIVITY_FILE,
    index=False
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ------------------------------------------------------------
# FINAL VERIFICATION
# ------------------------------------------------------------

print()
print("FILES CREATED")
print("-" * 70)

print(
    f"Training candidate:"
)

print(
    TRAINING_CANDIDATE_FILE
)

print()

print(
    f"Full sensitivity:"
)

print(
    SENSITIVITY_FILE
)

print()

print(
    f"Dataset summary:"
)

print(
    SUMMARY_FILE
)


print()
print("FINAL DATASET STATUS")
print("-" * 70)

print(
    f"Original dataset       : {len(df)} records"
)

print(
    f"Training candidate     : "
    f"{len(training_candidate)} records"
)

print(
    f"Full sensitivity       : "
    f"{len(full_sensitivity)} records"
)

print(
    f"Review records removed "
    f"from training candidate: "
    f"{len(df) - len(training_candidate)}"
)

print()
print("IMPORTANT")
print("-" * 70)

print(
    "The original processed dataset was NOT modified."
)

print(
    "The verified dataset was NOT modified."
)

print(
    "No target values were changed."
)

print(
    "No suspicious records were deleted from the "
    "project datasets."
)

print(
    "Only the conservative training candidate excludes "
    "REVIEW_REQUIRED records."
)

print()
print("STEP 9D COMPLETE")
print("=" * 70)