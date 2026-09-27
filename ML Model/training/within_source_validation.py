from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PRINTOPT AI — STEP 8D
# WITHIN-SOURCE VALIDATION
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 8D")
print("WITHIN-SOURCE VALIDATION")
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

OUTPUT_PATH = (
    OUTPUT_DIR
    / "within_source_validation.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ------------------------------------------------------------
# FEATURES AND TARGETS
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


# ------------------------------------------------------------
# SELECT SOURCES WITH ENOUGH RECORDS
# ------------------------------------------------------------

source_counts = (
    df["Source_ID"]
    .value_counts()
)

eligible_sources = source_counts[
    source_counts >= 4
].index.tolist()


print()
print("DATA")
print("-" * 70)

print(
    f"Total records : {len(df)}"
)

print(
    f"Total sources : "
    f"{df['Source_ID'].nunique()}"
)

print(
    f"Eligible sources "
    f"(>= 4 records): "
    f"{len(eligible_sources)}"
)

print()
print("ELIGIBLE SOURCES")
print("-" * 70)

for source in eligible_sources:

    print(
        f"{source:<10} "
        f"{source_counts[source]} records"
    )


# ------------------------------------------------------------
# CREATE WITHIN-SOURCE TRAIN / VALIDATION SPLITS
# ------------------------------------------------------------

train_parts = []
validation_parts = []


for source in eligible_sources:

    source_df = df[
        df["Source_ID"] == source
    ].copy()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    # Use Experiment_ID as a unique group so that
    # GroupShuffleSplit performs a record-level split.
    groups = source_df["Experiment_ID"]

    train_idx, validation_idx = next(
        splitter.split(
            source_df,
            groups=groups
        )
    )

    train_parts.append(
        source_df.iloc[train_idx]
    )

    validation_parts.append(
        source_df.iloc[validation_idx]
    )


train_df = pd.concat(
    train_parts,
    ignore_index=True
)

validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)


# ------------------------------------------------------------
# DISPLAY SPLIT
# ------------------------------------------------------------

print()
print("=" * 70)
print("WITHIN-SOURCE SPLIT")
print("=" * 70)

print()
print(
    f"Training records   : "
    f"{len(train_df)}"
)

print(
    f"Validation records : "
    f"{len(validation_df)}"
)

print(
    f"Training sources   : "
    f"{train_df['Source_ID'].nunique()}"
)

print(
    f"Validation sources : "
    f"{validation_df['Source_ID'].nunique()}"
)


# ------------------------------------------------------------
# VERIFY SOURCE OVERLAP
# ------------------------------------------------------------

train_sources = set(
    train_df["Source_ID"]
)

validation_sources = set(
    validation_df["Source_ID"]
)

overlap = (
    train_sources
    & validation_sources
)


print()
print(
    f"Source overlap     : "
    f"{len(overlap)} sources"
)

print(
    f"Expected           : "
    f"{len(eligible_sources)} sources"
)


# ------------------------------------------------------------
# PREPARE FEATURES/TARGETS
# ------------------------------------------------------------

X_train = train_df[
    FEATURES
]

X_validation = validation_df[
    FEATURES
]

y_train = train_df[
    TARGETS
]

y_validation = validation_df[
    TARGETS
]


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

print()
print("=" * 70)
print("MODEL TRAINING")
print("=" * 70)

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
)

model.fit(
    X_train,
    y_train
)

print()
print("Random Forest training complete.")


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

predictions = model.predict(
    X_validation
)

predictions = pd.DataFrame(
    predictions,
    columns=TARGETS,
    index=y_validation.index
)


# ------------------------------------------------------------
# OVERALL METRICS
# ------------------------------------------------------------

print()
print("=" * 70)
print("OVERALL PERFORMANCE")
print("=" * 70)


results = []


for target in TARGETS:

    actual = y_validation[target]

    predicted = predictions[target]

    r2 = r2_score(
        actual,
        predicted
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    results.append(
        {
            "Target": target,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
        }
    )

    print()
    print(target)

    print(
        f"  R²   : {r2:.4f}"
    )

    print(
        f"  MAE  : {mae:.4f}"
    )

    print(
        f"  RMSE : {rmse:.4f}"
    )


# ------------------------------------------------------------
# SOURCE-BY-SOURCE PERFORMANCE
# ------------------------------------------------------------

print()
print("=" * 70)
print("SOURCE-BY-SOURCE PERFORMANCE")
print("=" * 70)


source_results = []


for source in eligible_sources:

    mask = (
        validation_df["Source_ID"]
        == source
    )

    source_actual = (
        y_validation.loc[mask]
    )

    source_predicted = (
        predictions.loc[mask]
    )

    print()
    print(source)

    print(
        f"  Validation records : "
        f"{len(source_actual)}"
    )

    for target in TARGETS:

        actual = source_actual[target]

        predicted = source_predicted[target]

        mae = mean_absolute_error(
            actual,
            predicted
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

        # R² is undefined for one validation
        # sample, so only calculate when possible.
        if len(actual) >= 2:
            r2 = r2_score(
                actual,
                predicted
            )
        else:
            r2 = np.nan

        source_results.append(
            {
                "Source_ID": source,
                "Records": len(actual),
                "Target": target,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

        print(
            f"  {target:<18} "
            f"MAE={mae:.3f} "
            f"RMSE={rmse:.3f}"
        )


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)

source_results_df = pd.DataFrame(
    source_results
)

source_results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------

print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    OUTPUT_PATH
)

print()
print("STEP 8D COMPLETE")
print("=" * 70)