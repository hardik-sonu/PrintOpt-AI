from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PRINTOPT AI — STEP 9E
# SOURCE-AWARE TRAIN / TEST SPLIT V2
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9E")
print("SOURCE-AWARE TRAIN / TEST SPLIT V2")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "ti64_lpbf_training_candidate.csv"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"

TRAIN_FILE = (
    OUTPUT_DIR
    / "train_v2.csv"
)

TEST_FILE = (
    OUTPUT_DIR
    / "test_v2.csv"
)


# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print()
print("INPUT DATASET")
print("-" * 70)

print(f"Records : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"Sources : {df['Source_ID'].nunique()}")


# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

required_columns = [
    "Experiment_ID",
    "Source_ID",
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )


# ------------------------------------------------------------
# CHECK REVIEW-REQUIRED RECORDS
# ------------------------------------------------------------

if "Data_Status" in df.columns:

    review_required = (
        df["Data_Status"] == "REVIEW_REQUIRED"
    ).sum()

    print()
    print("DATA QUALITY CHECK")
    print("-" * 70)

    print(
        f"REVIEW_REQUIRED records : "
        f"{review_required}"
    )

    if review_required != 0:
        raise ValueError(
            "Training candidate still contains "
            "REVIEW_REQUIRED records."
        )


# ------------------------------------------------------------
# CHECK YS <= UTS
# ------------------------------------------------------------

ys_gt_uts = (
    df["YS_MPa"] > df["UTS_MPa"]
)

print(
    f"YS > UTS records       : "
    f"{ys_gt_uts.sum()}"
)

if ys_gt_uts.sum() != 0:
    raise ValueError(
        "YS > UTS records detected in the "
        "training candidate."
    )


# ------------------------------------------------------------
# SOURCE-AWARE SPLIT
#
# IMPORTANT:
# Entire sources are assigned to either train or test.
# No source can appear in both.
# ------------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df["Source_ID"]
    )
)


train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()


# ------------------------------------------------------------
# RESET INDICES
# ------------------------------------------------------------

train_df.reset_index(
    drop=True,
    inplace=True
)

test_df.reset_index(
    drop=True,
    inplace=True
)


# ------------------------------------------------------------
# SOURCE SETS
# ------------------------------------------------------------

train_sources = set(
    train_df["Source_ID"]
)

test_sources = set(
    test_df["Source_ID"]
)

source_overlap = (
    train_sources & test_sources
)


# ------------------------------------------------------------
# VALIDATE SOURCE SEPARATION
# ------------------------------------------------------------

if source_overlap:
    raise ValueError(
        "SOURCE LEAKAGE DETECTED: "
        f"{sorted(source_overlap)}"
    )


# ------------------------------------------------------------
# RECORD OVERLAP CHECK
# ------------------------------------------------------------

train_ids = set(
    train_df["Experiment_ID"]
)

test_ids = set(
    test_df["Experiment_ID"]
)

record_overlap = (
    train_ids & test_ids
)

if record_overlap:
    raise ValueError(
        "RECORD OVERLAP DETECTED: "
        f"{sorted(record_overlap)}"
    )


# ------------------------------------------------------------
# SOURCE DISTRIBUTION
# ------------------------------------------------------------

train_source_counts = (
    train_df["Source_ID"]
    .value_counts()
    .sort_index()
)

test_source_counts = (
    test_df["Source_ID"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print()
print("SOURCE-AWARE SPLIT")
print("-" * 70)

print(
    f"Total records       : {len(df)}"
)

print(
    f"Training records    : {len(train_df)}"
)

print(
    f"Testing records     : {len(test_df)}"
)

print()

print(
    f"Total sources       : "
    f"{df['Source_ID'].nunique()}"
)

print(
    f"Training sources    : "
    f"{len(train_sources)}"
)

print(
    f"Testing sources     : "
    f"{len(test_sources)}"
)

print()

print(
    f"Source overlap      : "
    f"{source_overlap}"
)

print(
    f"Record overlap      : "
    f"{record_overlap}"
)


# ------------------------------------------------------------
# TRAINING SOURCE DISTRIBUTION
# ------------------------------------------------------------

print()
print("TRAINING SOURCE DISTRIBUTION")
print("-" * 70)

print(
    train_source_counts.to_string()
)


# ------------------------------------------------------------
# TEST SOURCE DISTRIBUTION
# ------------------------------------------------------------

print()
print("TESTING SOURCE DISTRIBUTION")
print("-" * 70)

print(
    test_source_counts.to_string()
)


# ------------------------------------------------------------
# TARGET RANGES
# ------------------------------------------------------------

targets = [
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
]

print()
print("TARGET RANGES")
print("-" * 70)

for target in targets:

    print(
        f"{target:20s} "
        f"train={train_df[target].min():.3f}"
        f"–{train_df[target].max():.3f}    "
        f"test={test_df[target].min():.3f}"
        f"–{test_df[target].max():.3f}"
    )


# ------------------------------------------------------------
# FEATURE RANGES
# ------------------------------------------------------------

features = [
    "Powder_Size_um",
    "Laser_Spot_um",
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Hatch_Distance_um",
    "Layer_Thickness_um",
    "VED_J_mm3",
]

print()
print("FEATURE RANGES")
print("-" * 70)

for feature in features:

    print(
        f"{feature:25s} "
        f"train={train_df[feature].min():.3f}"
        f"–{train_df[feature].max():.3f}    "
        f"test={test_df[feature].min():.3f}"
        f"–{test_df[feature].max():.3f}"
    )


# ------------------------------------------------------------
# VED DOMAIN CHECK
# ------------------------------------------------------------

train_ved_min = train_df["VED_J_mm3"].min()
train_ved_max = train_df["VED_J_mm3"].max()

test_outside_ved = test_df[
    (test_df["VED_J_mm3"] < train_ved_min)
    | (test_df["VED_J_mm3"] > train_ved_max)
]

print()
print("VED DOMAIN CHECK")
print("-" * 70)

print(
    f"Training VED range : "
    f"{train_ved_min:.3f} – "
    f"{train_ved_max:.3f} J/mm³"
)

print(
    f"Test records outside training VED range : "
    f"{len(test_outside_ved)}"
)


# ------------------------------------------------------------
# SAVE FILES
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print()
print("FILES CREATED")
print("-" * 70)

print(
    f"Training:"
)

print(
    TRAIN_FILE
)

print()

print(
    f"Testing:"
)

print(
    TEST_FILE
)


print()
print("FINAL VALIDATION")
print("-" * 70)

print(
    f"Source overlap : {source_overlap}"
)

print(
    f"Record overlap : {record_overlap}"
)

print(
    f"YS > UTS in train : "
    f"{(train_df['YS_MPa'] > train_df['UTS_MPa']).sum()}"
)

print(
    f"YS > UTS in test  : "
    f"{(test_df['YS_MPa'] > test_df['UTS_MPa']).sum()}"
)


if source_overlap:
    raise ValueError(
        "Final validation failed: source overlap."
    )

if record_overlap:
    raise ValueError(
        "Final validation failed: record overlap."
    )


print()
print("IMPORTANT")
print("-" * 70)

print(
    "This split was created ONLY from the "
    "167-record conservative training candidate."
)

print(
    "The original 173-record split was NOT overwritten."
)

print(
    "Entire sources are held out from testing."
)

print(
    "No source appears in both training and testing."
)

print()
print("STEP 9E COMPLETE")
print("=" * 70)