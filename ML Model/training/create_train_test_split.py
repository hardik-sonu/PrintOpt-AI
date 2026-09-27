from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "ti64_lpbf_processed.csv"
TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("PRINTOPT AI — TRAIN/TEST SPLIT")
print("=" * 70)

print(f"\nInput dataset: {INPUT_FILE}")
print(f"Total records: {len(df)}")
print(f"Total sources: {df['Source_ID'].nunique()}")


# ============================================================
# BASIC VALIDATION
# ============================================================

if "Source_ID" not in df.columns:
    raise ValueError("Source_ID column not found in processed dataset.")

if df["Source_ID"].isna().any():
    raise ValueError("Source_ID contains missing values.")

if df["Source_ID"].nunique() < 5:
    raise ValueError("Too few source groups for source-aware splitting.")


# ============================================================
# SOURCE-AWARE TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df["Source_ID"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()


# ============================================================
# VERIFY SOURCE SEPARATION
# ============================================================

train_sources = set(train_df["Source_ID"])
test_sources = set(test_df["Source_ID"])

overlap = train_sources.intersection(test_sources)

if overlap:
    raise RuntimeError(
        f"Source leakage detected! "
        f"These sources appear in both sets: {sorted(overlap)}"
    )


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# ============================================================
# REPORT
# ============================================================

print("\n" + "-" * 70)
print("SPLIT RESULTS")
print("-" * 70)

print(f"Training records : {len(train_df)}")
print(f"Testing records  : {len(test_df)}")

print(f"\nTraining sources : {train_df['Source_ID'].nunique()}")
print(f"Testing sources  : {test_df['Source_ID'].nunique()}")

print("\nTraining sources:")
print(
    train_df["Source_ID"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nTesting sources:")
print(
    test_df["Source_ID"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nSource overlap:")
print(overlap)

print("\nSaved files:")
print(f"  {TRAIN_FILE}")
print(f"  {TEST_FILE}")

print("\n" + "=" * 70)
print("STEP 5A COMPLETE")
print("=" * 70)