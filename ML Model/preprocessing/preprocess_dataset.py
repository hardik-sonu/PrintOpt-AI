from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PRINTOPT AI
# Ti-6Al-4V LPBF Dataset Preprocessing
# ============================================================

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "ti64_lpbf_dataset.xlsx"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "ti64_lpbf_processed.csv"


def create_source_id(reference):
    """
    Convert the original Reference field into a clean Source_ID.

    Literature references:
        [1] ...  -> REF_01
        [2] ...  -> REF_02
        ...

    Laboratory records:
        Experimental data from the laboratory -> LAB
    """

    reference = str(reference).strip()

    if reference.lower().startswith("experimental data from the laboratory"):
        return "LAB"

    if reference.startswith("["):
        closing_bracket = reference.find("]")

        if closing_bracket != -1:
            number_text = reference[1:closing_bracket]

            if number_text.isdigit():
                return f"REF_{int(number_text):02d}"

    return "UNKNOWN"


def main():

    print("=" * 80)
    print("PRINTOPT AI — DATASET PREPROCESSING")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Check raw dataset
    # --------------------------------------------------------

    print("\n[1] Checking raw dataset...")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_FILE}"
        )

    print(f"Raw dataset: {RAW_FILE}")

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    print("\n[2] Loading dataset...")

    df = pd.read_excel(RAW_FILE)

    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # --------------------------------------------------------
    # 3. Rename columns
    # --------------------------------------------------------

    print("\n[3] Standardizing column names...")

    rename_map = {
        "Alloy": "Experiment_ID",
        "Powder Size (μm)": "Powder_Size_um",
        "Laser Spot (μm)": "Laser_Spot_um",
        "Laser Power (W)": "Laser_Power_W",
        "Scanning Speed (mm/s)": "Scanning_Speed_mm_s",
        "Hatch Distance (μm)": "Hatch_Distance_um",
        "Layer Thickness (μm)": "Layer_Thickness_um",
        "Gauge Area (mm²)": "Gauge_Area_mm2",
        "Gauge Length (mm)": "Gauge_Length_mm",
        "UTS (MPa)": "UTS_MPa",
        "YS (MPa)": "YS_MPa",
        "EF (%)": "Elongation_pct",
        "Reference": "Reference",
    }

    df = df.rename(columns=rename_map)

    print("Column names standardized.")

    # --------------------------------------------------------
    # 4. Create Source_ID
    # --------------------------------------------------------

    print("\n[4] Creating Source_ID...")

    df["Source_ID"] = df["Reference"].apply(create_source_id)

    print("Source distribution:")
    print(
        df["Source_ID"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # 5. Preserve original record ID
    # --------------------------------------------------------

    print("\n[5] Validating Experiment_ID...")

    if df["Experiment_ID"].duplicated().any():
        duplicate_ids = df.loc[
            df["Experiment_ID"].duplicated(keep=False),
            "Experiment_ID",
        ].tolist()

        raise ValueError(
            f"Duplicate Experiment_ID values detected: {duplicate_ids}"
        )

    print(
        f"Experiment_ID range: "
        f"{df['Experiment_ID'].min()} → "
        f"{df['Experiment_ID'].max()}"
    )

    # --------------------------------------------------------
    # 6. Convert units for VED calculation
    # --------------------------------------------------------

    print("\n[6] Converting μm → mm for VED calculation...")

    df["Hatch_Distance_mm"] = (
        df["Hatch_Distance_um"] / 1000.0
    )

    df["Layer_Thickness_mm"] = (
        df["Layer_Thickness_um"] / 1000.0
    )

    # --------------------------------------------------------
    # 7. Calculate VED
    # --------------------------------------------------------

    print("\n[7] Calculating Volumetric Energy Density (VED)...")

    denominator = (
        df["Scanning_Speed_mm_s"]
        * df["Hatch_Distance_mm"]
        * df["Layer_Thickness_mm"]
    )

    if (denominator <= 0).any():
        raise ValueError(
            "Invalid VED denominator detected. "
            "Scanning speed, hatch distance, and layer thickness "
            "must all be greater than zero."
        )

    df["VED_J_mm3"] = (
        df["Laser_Power_W"] / denominator
    )

    print(
        "VED calculation completed."
    )

    print(
        f"VED range: "
        f"{df['VED_J_mm3'].min():.3f} → "
        f"{df['VED_J_mm3'].max():.3f} J/mm³"
    )

    # --------------------------------------------------------
    # 8. Create a process/material feature list
    # --------------------------------------------------------

    print("\n[8] Defining ML feature groups...")

    process_features = [
        "Powder_Size_um",
        "Laser_Spot_um",
        "Laser_Power_W",
        "Scanning_Speed_mm_s",
        "Hatch_Distance_um",
        "Layer_Thickness_um",
        "VED_J_mm3",
    ]

    testing_metadata = [
        "Gauge_Area_mm2",
        "Gauge_Length_mm",
    ]

    targets = [
        "UTS_MPa",
        "YS_MPa",
        "Elongation_pct",
    ]

    print("\nProcess/material features:")
    for feature in process_features:
        print(f"  - {feature}")

    print("\nTesting metadata:")
    for feature in testing_metadata:
        print(f"  - {feature}")

    print("\nPrediction targets:")
    for target in targets:
        print(f"  - {target}")

    # --------------------------------------------------------
    # 9. Validate required columns
    # --------------------------------------------------------

    print("\n[9] Validating required columns...")

    required_columns = (
        [
            "Experiment_ID",
            "Source_ID",
            "Reference",
        ]
        + process_features
        + testing_metadata
        + targets
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("All required columns are present.")

    # --------------------------------------------------------
    # 10. Check missing values
    # --------------------------------------------------------

    print("\n[10] Checking missing values...")

    missing_values = df[required_columns].isnull().sum()

    if missing_values.sum() > 0:

        print(
            missing_values[
                missing_values > 0
            ].to_string()
        )

        raise ValueError(
            "Missing values detected. "
            "Dataset requires further cleaning."
        )

    print("No missing values detected.")

    # --------------------------------------------------------
    # 11. Check infinite values
    # --------------------------------------------------------

    print("\n[11] Checking infinite values...")

    numerical_columns = (
        process_features
        + testing_metadata
        + targets
    )

    infinite_counts = np.isinf(
        df[numerical_columns].to_numpy()
    ).sum()

    if infinite_counts > 0:
        raise ValueError(
            "Infinite numerical values detected."
        )

    print("No infinite values detected.")

    # --------------------------------------------------------
    # 12. Validate physically meaningful ranges
    # --------------------------------------------------------

    print("\n[12] Checking physical parameter ranges...")

    positive_columns = [
        "Powder_Size_um",
        "Laser_Spot_um",
        "Laser_Power_W",
        "Scanning_Speed_mm_s",
        "Hatch_Distance_um",
        "Layer_Thickness_um",
        "Gauge_Area_mm2",
        "Gauge_Length_mm",
    ]

    for column in positive_columns:

        invalid_count = (
            df[column] <= 0
        ).sum()

        if invalid_count > 0:
            raise ValueError(
                f"{column} contains "
                f"{invalid_count} non-positive values."
            )

    print(
        "All checked physical quantities "
        "contain positive values."
    )

    # --------------------------------------------------------
    # 13. Check target ranges
    # --------------------------------------------------------

    print("\n[13] Target variable ranges...")

    for target in targets:

        print(
            f"{target}: "
            f"{df[target].min():.3f} → "
            f"{df[target].max():.3f}"
        )

    # --------------------------------------------------------
    # 14. Arrange final columns
    # --------------------------------------------------------

    print("\n[14] Arranging processed dataset...")

    final_columns = [
        "Experiment_ID",
        "Source_ID",
        "Powder_Size_um",
        "Laser_Spot_um",
        "Laser_Power_W",
        "Scanning_Speed_mm_s",
        "Hatch_Distance_um",
        "Layer_Thickness_um",
        "Hatch_Distance_mm",
        "Layer_Thickness_mm",
        "VED_J_mm3",
        "Gauge_Area_mm2",
        "Gauge_Length_mm",
        "UTS_MPa",
        "YS_MPa",
        "Elongation_pct",
        "Reference",
    ]

    df_processed = df[final_columns].copy()

    # --------------------------------------------------------
    # 15. Sort by Experiment_ID
    # --------------------------------------------------------

    df_processed = df_processed.sort_values(
        "Experiment_ID"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 16. Save processed dataset
    # --------------------------------------------------------

    print("\n[15] Saving processed dataset...")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df_processed.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved to:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # 17. Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    print(f"Rows       : {len(df_processed)}")
    print(f"Columns    : {len(df_processed.columns)}")
    print(
        f"VED range  : "
        f"{df_processed['VED_J_mm3'].min():.3f} → "
        f"{df_processed['VED_J_mm3'].max():.3f} J/mm³"
    )

    print("\nOutput columns:")

    for index, column in enumerate(
        df_processed.columns,
        start=1
    ):
        print(f"{index:2}. {column}")

    print("\nOriginal Excel dataset was not modified.")
    print("=" * 80)


if __name__ == "__main__":
    main()