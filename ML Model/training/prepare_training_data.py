from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test.csv"


# ============================================================
# FEATURES AND TARGETS
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)


print("=" * 70)
print("PRINTOPT AI — STEP 5B")
print("FEATURE / TARGET PREPARATION")
print("=" * 70)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = FEATURE_COLUMNS + TARGET_COLUMNS + ["Source_ID"]

for column in required_columns:
    if column not in train_df.columns:
        raise ValueError(
            f"Missing required column in training data: {column}"
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Missing required column in testing data: {column}"
        )


# ============================================================
# CREATE FEATURE MATRICES
# ============================================================

X_train = train_df[FEATURE_COLUMNS].copy()
X_test = test_df[FEATURE_COLUMNS].copy()


# ============================================================
# CREATE TARGET MATRICES
# ============================================================

y_train = train_df[TARGET_COLUMNS].copy()
y_test = test_df[TARGET_COLUMNS].copy()


# ============================================================
# NUMERIC VALIDATION
# ============================================================

for column in FEATURE_COLUMNS:
    if not pd.api.types.is_numeric_dtype(X_train[column]):
        raise TypeError(
            f"Training feature is not numeric: {column}"
        )

    if not pd.api.types.is_numeric_dtype(X_test[column]):
        raise TypeError(
            f"Testing feature is not numeric: {column}"
        )


for column in TARGET_COLUMNS:
    if not pd.api.types.is_numeric_dtype(y_train[column]):
        raise TypeError(
            f"Training target is not numeric: {column}"
        )

    if not pd.api.types.is_numeric_dtype(y_test[column]):
        raise TypeError(
            f"Testing target is not numeric: {column}"
        )


# ============================================================
# MISSING / INFINITE VALUE CHECK
# ============================================================

if X_train.isna().any().any():
    raise ValueError("Missing values detected in X_train.")

if X_test.isna().any().any():
    raise ValueError("Missing values detected in X_test.")

if y_train.isna().any().any():
    raise ValueError("Missing values detected in y_train.")

if y_test.isna().any().any():
    raise ValueError("Missing values detected in y_test.")


if X_train.isin([float("inf"), float("-inf")]).any().any():
    raise ValueError("Infinite values detected in X_train.")

if X_test.isin([float("inf"), float("-inf")]).any().any():
    raise ValueError("Infinite values detected in X_test.")


# ============================================================
# SOURCE LEAKAGE CHECK
# ============================================================

train_sources = set(train_df["Source_ID"])
test_sources = set(test_df["Source_ID"])

source_overlap = train_sources.intersection(test_sources)

if source_overlap:
    raise RuntimeError(
        f"Source leakage detected: {sorted(source_overlap)}"
    )


# ============================================================
# REPORT
# ============================================================

print("\nFEATURES")
print("-" * 70)

for i, feature in enumerate(FEATURE_COLUMNS, start=1):
    print(f"{i}. {feature}")


print("\nTARGETS")
print("-" * 70)

for i, target in enumerate(TARGET_COLUMNS, start=1):
    print(f"{i}. {target}")


print("\nMATRIX SHAPES")
print("-" * 70)

print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y_train : {y_train.shape}")
print(f"y_test  : {y_test.shape}")


print("\nTRAINING FEATURE RANGES")
print("-" * 70)
print(X_train.describe().T[["min", "max"]].to_string())


print("\nTRAINING TARGET RANGES")
print("-" * 70)
print(y_train.describe().T[["min", "max"]].to_string())


print("\nTESTING FEATURE RANGES")
print("-" * 70)
print(X_test.describe().T[["min", "max"]].to_string())


print("\nTESTING TARGET RANGES")
print("-" * 70)
print(y_test.describe().T[["min", "max"]].to_string())


print("\nSOURCE LEAKAGE CHECK")
print("-" * 70)
print(f"Training sources: {len(train_sources)}")
print(f"Testing sources : {len(test_sources)}")
print(f"Overlap         : {source_overlap}")


print("\n" + "=" * 70)
print("STEP 5B COMPLETE")
print("=" * 70)