from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder


# ============================================================
# PRINTOPT AI — STEP 8C
# SOURCE CLASSIFICATION DIAGNOSTIC
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 8C")
print("SOURCE CLASSIFICATION DIAGNOSTIC")
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


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ------------------------------------------------------------
# FEATURES
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


X = df[FEATURES]
y_text = df["Source_ID"]


# ------------------------------------------------------------
# ENCODE SOURCE LABELS
# ------------------------------------------------------------

encoder = LabelEncoder()

y = encoder.fit_transform(y_text)


# ------------------------------------------------------------
# DATA SUMMARY
# ------------------------------------------------------------

print()
print("DATA")
print("-" * 70)

print(f"Total records : {len(df)}")
print(f"Total sources : {df['Source_ID'].nunique()}")
print(f"Features      : {len(FEATURES)}")

print()
print("SOURCE DISTRIBUTION")
print("-" * 70)

print(
    df["Source_ID"]
    .value_counts()
    .sort_values(ascending=False)
    .to_string()
)


# ------------------------------------------------------------
# GROUP-AWARE CROSS-VALIDATION
# ------------------------------------------------------------

groups = df["Source_ID"]

splitter = GroupKFold(
    n_splits=5
)


accuracies = []
balanced_accuracies = []


print()
print("=" * 70)
print("CROSS-VALIDATION")
print("=" * 70)


for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups=groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    train_sources = set(
        groups.iloc[train_idx]
    )

    test_sources = set(
        groups.iloc[test_idx]
    )

    print()
    print(f"FOLD {fold}")
    print("-" * 70)

    print(
        f"Training records : {len(train_idx)}"
    )

    print(
        f"Testing records  : {len(test_idx)}"
    )

    print(
        f"Training sources : {len(train_sources)}"
    )

    print(
        f"Testing sources  : {len(test_sources)}"
    )

    print(
        f"Test source(s)   : "
        f"{sorted(test_sources)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        predictions
    )

    accuracies.append(
        accuracy
    )

    balanced_accuracies.append(
        balanced_accuracy
    )

    print()
    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.4f}"
    )


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print()
print(
    f"Mean Accuracy          : "
    f"{sum(accuracies) / len(accuracies):.4f}"
)

print(
    f"Mean Balanced Accuracy : "
    f"{sum(balanced_accuracies) / len(balanced_accuracies):.4f}"
)

print()
print(
    "Important:"
)

print(
    "This is a diagnostic only."
)

print(
    "It is NOT a production PrintOpt AI model."
)

print("=" * 70)
print("STEP 8C COMPLETE")
print("=" * 70)