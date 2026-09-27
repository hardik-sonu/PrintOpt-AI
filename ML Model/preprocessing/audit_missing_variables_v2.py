from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PRINTOPT AI — STEP 9K
# MISSING-VARIABLE & SOURCE-GENERALIZATION AUDIT V2
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train_v2.csv"
TEST_FILE = DATA_DIR / "test_v2.csv"

TRAIN = pd.read_csv(TRAIN_FILE)
TEST = pd.read_csv(TEST_FILE)

FULL = pd.concat([TRAIN, TEST], ignore_index=True)


# ============================================================
# CONFIGURATION
# ============================================================

PROCESS_FEATURES = [
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

# Scientific variables that would ideally be available for
# stronger cross-source LPBF modeling.
MISSING_VARIABLES = {
    "Scan_Strategy": (
        "Raster/stripe/island/contour strategy and scan rotation."
    ),
    "Build_Orientation": (
        "Part orientation relative to the build direction."
    ),
    "Preheat_Build_Temperature": (
        "Powder-bed/build-platform preheating conditions."
    ),
    "Atmosphere": (
        "Chamber gas, oxygen level and other atmosphere conditions."
    ),
    "Powder_Chemistry": (
        "Detailed Ti-6Al-4V powder chemical composition."
    ),
    "Powder_Morphology": (
        "Particle shape, sphericity, morphology and flowability."
    ),
    "Powder_Size_Distribution": (
        "Full particle-size distribution rather than a single powder-size value."
    ),
    "Machine_System": (
        "Machine/manufacturer/platform and laser/system characteristics."
    ),
    "Laser_Wavelength": (
        "Laser wavelength where relevant."
    ),
    "Laser_Profile": (
        "Laser mode/profile and detailed beam characteristics."
    ),
    "Heat_Treatment": (
        "Post-build heat treatment or HIP conditions."
    ),
    "Cooling_History": (
        "Cooling conditions/history during and after fabrication."
    ),
    "Microstructure": (
        "Phase, grain morphology, alpha/beta structure and texture."
    ),
    "Porosity_Defects": (
        "Porosity, lack-of-fusion, keyhole or other defect information."
    ),
    "Surface_Roughness": (
        "Surface roughness measurements such as Ra."
    ),
    "Hardness": (
        "Vickers or other hardness measurements."
    ),
    "Tensile_Test_Method": (
        "Specimen standard, strain rate, test direction and methodology."
    ),
    "Specimen_Geometry": (
        "Detailed specimen dimensions and geometry."
    ),
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9K")
print("MISSING-VARIABLE & SOURCE-GENERALIZATION AUDIT V2")
print("=" * 70)


# ============================================================
# 1. DATASET STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET STRUCTURE")
print("=" * 70)

print(f"Training records : {len(TRAIN)}")
print(f"Testing records  : {len(TEST)}")
print(f"Total records    : {len(FULL)}")

print(f"Training sources : {TRAIN['Source_ID'].nunique()}")
print(f"Testing sources  : {TEST['Source_ID'].nunique()}")

print("\nTraining sources:")
print(", ".join(sorted(TRAIN["Source_ID"].unique())))

print("\nHeld-out sources:")
print(", ".join(sorted(TEST["Source_ID"].unique())))


# ============================================================
# 2. AVAILABLE VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("2. VARIABLES CURRENTLY AVAILABLE")
print("=" * 70)

available_rows = []

for feature in PROCESS_FEATURES:
    available_rows.append(
        {
            "Variable": feature,
            "Status": "AVAILABLE",
            "Type": "Process / derived process feature",
            "Missing_Count": int(FULL[feature].isna().sum()),
            "Unique_Values": int(FULL[feature].nunique()),
        }
    )

for target in TARGETS:
    available_rows.append(
        {
            "Variable": target,
            "Status": "AVAILABLE",
            "Type": "Mechanical-property target",
            "Missing_Count": int(FULL[target].isna().sum()),
            "Unique_Values": int(FULL[target].nunique()),
        }
    )

available_df = pd.DataFrame(available_rows)

print(available_df.to_string(index=False))


# ============================================================
# 3. MISSING SCIENTIFIC VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("3. SCIENTIFIC VARIABLES NOT CURRENTLY AVAILABLE")
print("=" * 70)

missing_rows = []

for variable, description in MISSING_VARIABLES.items():
    missing_rows.append(
        {
            "Variable": variable,
            "Status": "NOT_AVAILABLE",
            "Description": description,
        }
    )

missing_df = pd.DataFrame(missing_rows)

print(missing_df.to_string(index=False))


# ============================================================
# 4. SOURCE SIZE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("4. SOURCE DISTRIBUTION")
print("=" * 70)

train_source_counts = (
    TRAIN["Source_ID"]
    .value_counts()
    .rename("Training_Records")
)

test_source_counts = (
    TEST["Source_ID"]
    .value_counts()
    .rename("Testing_Records")
)

source_counts = pd.concat(
    [train_source_counts, test_source_counts],
    axis=1
).fillna(0)

source_counts["Training_Records"] = (
    source_counts["Training_Records"].astype(int)
)

source_counts["Testing_Records"] = (
    source_counts["Testing_Records"].astype(int)
)

source_counts["Total_Records"] = (
    source_counts["Training_Records"]
    + source_counts["Testing_Records"]
)

source_counts["Held_Out"] = (
    source_counts["Testing_Records"] > 0
)

source_counts = source_counts.sort_values(
    "Total_Records",
    ascending=False
)

print(source_counts.to_string())


# ============================================================
# 5. SOURCE-LEVEL PROCESS VARIATION
# ============================================================

print("\n" + "=" * 70)
print("5. SOURCE-LEVEL PROCESS VARIATION")
print("=" * 70)

source_variation_rows = []

for source, group in FULL.groupby("Source_ID"):

    row = {
        "Source_ID": source,
        "Records": len(group),
    }

    for feature in PROCESS_FEATURES:
        row[f"{feature}_Unique"] = group[feature].nunique()

    # Number of process variables that remain constant
    # within the source.
    constant_features = 0

    for feature in PROCESS_FEATURES:
        if group[feature].nunique() <= 1:
            constant_features += 1

    row["Constant_Process_Features"] = constant_features
    row["Variable_Process_Features"] = (
        len(PROCESS_FEATURES) - constant_features
    )

    source_variation_rows.append(row)

source_variation_df = pd.DataFrame(source_variation_rows)

print(
    source_variation_df[
        [
            "Source_ID",
            "Records",
            "Constant_Process_Features",
            "Variable_Process_Features",
        ]
    ].sort_values("Records", ascending=False).to_string(index=False)
)


# ============================================================
# 6. SOURCE-LEVEL TARGET VARIATION
# ============================================================

print("\n" + "=" * 70)
print("6. SOURCE-LEVEL TARGET VARIATION")
print("=" * 70)

target_rows = []

for source, group in FULL.groupby("Source_ID"):

    row = {
        "Source_ID": source,
        "Records": len(group),
    }

    for target in TARGETS:
        row[f"{target}_Mean"] = group[target].mean()
        row[f"{target}_Std"] = group[target].std()
        row[f"{target}_Min"] = group[target].min()
        row[f"{target}_Max"] = group[target].max()

    target_rows.append(row)

target_variation_df = pd.DataFrame(target_rows)

display_columns = ["Source_ID", "Records"]

for target in TARGETS:
    display_columns.extend(
        [
            f"{target}_Mean",
            f"{target}_Std",
            f"{target}_Min",
            f"{target}_Max",
        ]
    )

print(
    target_variation_df[
        display_columns
    ].sort_values("Records", ascending=False).to_string(index=False)
)


# ============================================================
# 7. SOURCE-TO-SOURCE TARGET SHIFT
# ============================================================

print("\n" + "=" * 70)
print("7. SOURCE-TO-SOURCE TARGET SHIFT")
print("=" * 70)

global_target_stats = {}

for target in TARGETS:
    global_target_stats[target] = {
        "mean": FULL[target].mean(),
        "std": FULL[target].std(),
        "min": FULL[target].min(),
        "max": FULL[target].max(),
    }

target_shift_rows = []

for source, group in FULL.groupby("Source_ID"):

    row = {
        "Source_ID": source,
        "Records": len(group),
    }

    z_values = []

    for target in TARGETS:

        global_mean = global_target_stats[target]["mean"]
        global_std = global_target_stats[target]["std"]

        source_mean = group[target].mean()

        if global_std > 0:
            z = (source_mean - global_mean) / global_std
        else:
            z = 0.0

        row[f"{target}_Mean_Z"] = z
        z_values.append(abs(z))

    row["Mean_Absolute_Target_Z"] = np.mean(z_values)
    row["Max_Absolute_Target_Z"] = np.max(z_values)

    target_shift_rows.append(row)

target_shift_df = pd.DataFrame(target_shift_rows)

print(
    target_shift_df.sort_values(
        "Mean_Absolute_Target_Z",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# 8. SOURCE PROCESS SHIFT
# ============================================================

print("\n" + "=" * 70)
print("8. SOURCE PROCESS DISTRIBUTION SHIFT")
print("=" * 70)

train_feature_stats = {}

for feature in PROCESS_FEATURES:

    train_feature_stats[feature] = {
        "mean": TRAIN[feature].mean(),
        "std": TRAIN[feature].std(),
        "min": TRAIN[feature].min(),
        "max": TRAIN[feature].max(),
    }

process_shift_rows = []

for source, group in TEST.groupby("Source_ID"):

    row = {
        "Source_ID": source,
        "Records": len(group),
    }

    abs_z_values = []
    outside_range_count = 0
    shifted_feature_count = 0

    for feature in PROCESS_FEATURES:

        train_mean = train_feature_stats[feature]["mean"]
        train_std = train_feature_stats[feature]["std"]
        train_min = train_feature_stats[feature]["min"]
        train_max = train_feature_stats[feature]["max"]

        source_mean = group[feature].mean()

        if train_std > 0:
            z = (source_mean - train_mean) / train_std
        else:
            z = 0.0

        abs_z = abs(z)

        row[f"{feature}_Z"] = z

        abs_z_values.append(abs_z)

        if abs_z >= 2:
            shifted_feature_count += 1

        if (
            group[feature].min() < train_min
            or group[feature].max() > train_max
        ):
            outside_range_count += 1

    row["Mean_Absolute_Z"] = np.mean(abs_z_values)
    row["Max_Absolute_Z"] = np.max(abs_z_values)
    row["Features_GE_2SD"] = shifted_feature_count
    row["Features_Outside_Train_Range"] = outside_range_count

    process_shift_rows.append(row)

process_shift_df = pd.DataFrame(process_shift_rows)

print(
    process_shift_df.sort_values(
        "Mean_Absolute_Z",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# 9. SOURCE GENERALIZATION RISK INDICATORS
# ============================================================

print("\n" + "=" * 70)
print("9. SOURCE GENERALIZATION RISK INDICATORS")
print("=" * 70)

risk_rows = []

for _, row in process_shift_df.iterrows():

    source = row["Source_ID"]

    target_row = target_shift_df[
        target_shift_df["Source_ID"] == source
    ]

    if len(target_row) == 1:
        target_row = target_row.iloc[0]

        target_shift = target_row[
            "Mean_Absolute_Target_Z"
        ]

        target_max_shift = target_row[
            "Max_Absolute_Target_Z"
        ]

    else:
        target_shift = np.nan
        target_max_shift = np.nan

    source_row = source_variation_df[
        source_variation_df["Source_ID"] == source
    ]

    if len(source_row) == 1:
        source_row = source_row.iloc[0]

        constant_features = source_row[
            "Constant_Process_Features"
        ]

    else:
        constant_features = np.nan

    risk_rows.append(
        {
            "Source_ID": source,
            "Records": int(row["Records"]),
            "Process_Mean_Absolute_Z": row[
                "Mean_Absolute_Z"
            ],
            "Process_Max_Absolute_Z": row[
                "Max_Absolute_Z"
            ],
            "Process_Features_GE_2SD": int(
                row["Features_GE_2SD"]
            ),
            "Process_Features_Outside_Train_Range": int(
                row["Features_Outside_Train_Range"]
            ),
            "Target_Mean_Absolute_Z": target_shift,
            "Target_Max_Absolute_Z": target_max_shift,
            "Constant_Process_Features": constant_features,
        }
    )

risk_df = pd.DataFrame(risk_rows)

print(
    risk_df.sort_values(
        "Process_Max_Absolute_Z",
        ascending=False
    ).to_string(index=False)
)


# ============================================================
# 10. DATA COVERAGE SCORE
# ============================================================

print("\n" + "=" * 70)
print("10. DATA COVERAGE ASSESSMENT")
print("=" * 70)

coverage_rows = []

for variable in PROCESS_FEATURES:
    coverage_rows.append(
        {
            "Variable": variable,
            "Available": True,
            "Directly_Measured_or_Derived": (
                "Derived" if variable == "VED_J_mm3"
                else "Measured/Reported"
            ),
            "Potential_Source_Dependence": (
                "Low/Moderate"
            ),
        }
    )

for variable in MISSING_VARIABLES:
    coverage_rows.append(
        {
            "Variable": variable,
            "Available": False,
            "Directly_Measured_or_Derived": "Not available",
            "Potential_Source_Dependence": "Potentially High",
        }
    )

coverage_df = pd.DataFrame(coverage_rows)

print(
    coverage_df.to_string(index=False)
)


# ============================================================
# 11. SOURCE GENERALIZATION INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("11. INTERPRETATION")
print("=" * 70)

print(
    """
The current dataset contains a useful set of LPBF process variables,
but it does not contain many variables that can change the
process -> microstructure -> mechanical-property relationship.

The strongest limitation identified by this audit is therefore
not simply the number of process parameters.

The dataset represents multiple literature sources that may differ
in machine, powder, scan strategy, thermal history, specimen
orientation, post-processing, microstructure and testing methodology.

Because Source_ID is deliberately excluded from the ML features,
the model must attempt to generalize across these differences
without directly observing them.

This creates a difficult unseen-source prediction problem.

The audit does NOT conclude that any particular source is incorrect.

A large source shift or model error should be interpreted as
possible domain shift, missing-variable effects, or source-specific
relationships unless the original experimental source proves
otherwise.
"""
)


# ============================================================
# 12. RECOMMENDED MODELING STRATEGY
# ============================================================

print("\n" + "=" * 70)
print("12. RECOMMENDED MODELING STRATEGY")
print("=" * 70)

recommendations = [
    (
        "Primary baseline",
        "Keep the original 7-feature process representation."
    ),
    (
        "Engineered features",
        "Keep the 13-feature experiment as an audit result, "
        "but do not adopt it as the primary model."
    ),
    (
        "Validation",
        "Continue using source-held-out validation for "
        "unseen-literature generalization."
    ),
    (
        "Within-source modeling",
        "Treat within-source validation as a separate modeling "
        "scenario because it answers a different question."
    ),
    (
        "Future data enrichment",
        "Prioritize scan strategy, build orientation, heat treatment, "
        "machine/system, powder characterization, defects/porosity, "
        "microstructure and tensile-test methodology."
    ),
    (
        "Source_ID",
        "Do not use Source_ID as an input feature for a model "
        "intended to predict completely unseen sources."
    ),
    (
        "Target quality",
        "Keep unresolved records flagged rather than silently "
        "correcting or deleting them."
    ),
]

for title, description in recommendations:
    print(f"\n{title}:")
    print(f"  {description}")


# ============================================================
# 13. SAVE RESULTS
# ============================================================

available_df.to_csv(
    EVAL_DIR / "available_variables_v2.csv",
    index=False
)

missing_df.to_csv(
    EVAL_DIR / "missing_variables_v2.csv",
    index=False
)

source_counts.to_csv(
    EVAL_DIR / "source_distribution_v2.csv"
)

source_variation_df.to_csv(
    EVAL_DIR / "source_process_variation_v2.csv",
    index=False
)

target_variation_df.to_csv(
    EVAL_DIR / "source_target_variation_v2.csv",
    index=False
)

target_shift_df.to_csv(
    EVAL_DIR / "source_target_shift_v2.csv",
    index=False
)

process_shift_df.to_csv(
    EVAL_DIR / "source_process_shift_v2.csv",
    index=False
)

risk_df.to_csv(
    EVAL_DIR / "source_generalization_risk_v2.csv",
    index=False
)

coverage_df.to_csv(
    EVAL_DIR / "dataset_variable_coverage_v2.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

files_created = [
    "available_variables_v2.csv",
    "missing_variables_v2.csv",
    "source_distribution_v2.csv",
    "source_process_variation_v2.csv",
    "source_target_variation_v2.csv",
    "source_target_shift_v2.csv",
    "source_process_shift_v2.csv",
    "source_generalization_risk_v2.csv",
    "dataset_variable_coverage_v2.csv",
]

for filename in files_created:
    print(EVAL_DIR / filename)

print("\n" + "=" * 70)
print("STEP 9K COMPLETE")
print("=" * 70)

print(
    """
Important:
 - Original dataset unchanged.
 - train_v2.csv unchanged.
 - test_v2.csv unchanged.
 - No targets modified.
 - No records removed.
 - Source_ID was not used as an ML feature.
 - No new model was trained.
 - This step is an audit of data coverage and source generalization.
"""
)