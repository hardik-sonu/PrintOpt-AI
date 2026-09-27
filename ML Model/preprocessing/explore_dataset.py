from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PRINTOPT AI
# Ti-6Al-4V LPBF Exploratory Data Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ti64_lpbf_processed.csv"
)


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():

    print_section(
        "PRINTOPT AI — EXPLORATORY DATA ANALYSIS"
    )

    # --------------------------------------------------------
    # 1. Load processed dataset
    # --------------------------------------------------------

    print_section("[1] Loading processed dataset")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Define variables
    # --------------------------------------------------------

    process_features = [
        "Powder_Size_um",
        "Laser_Spot_um",
        "Laser_Power_W",
        "Scanning_Speed_mm_s",
        "Hatch_Distance_um",
        "Layer_Thickness_um",
        "VED_J_mm3",
    ]

    targets = [
        "UTS_MPa",
        "YS_MPa",
        "Elongation_pct",
    ]

    # --------------------------------------------------------
    # 3. Descriptive statistics
    # --------------------------------------------------------

    print_section("[2] Descriptive statistics")

    analysis_columns = process_features + targets

    stats = df[analysis_columns].describe().T

    stats["median"] = df[analysis_columns].median()

    stats = stats[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "median",
            "75%",
            "max",
        ]
    ]

    print(stats.round(3).to_string())

    # --------------------------------------------------------
    # 4. Correlation: inputs vs targets
    # --------------------------------------------------------

    print_section(
        "[3] Correlation between process features and targets"
    )

    correlation = df[
        process_features + targets
    ].corr()

    input_target_correlation = correlation.loc[
        process_features,
        targets,
    ]

    print(
        input_target_correlation
        .round(3)
        .to_string()
    )

    # --------------------------------------------------------
    # 5. Target-to-target correlations
    # --------------------------------------------------------

    print_section(
        "[4] Correlation between prediction targets"
    )

    target_correlation = df[targets].corr()

    print(
        target_correlation
        .round(3)
        .to_string()
    )

    # --------------------------------------------------------
    # 6. VED relationship with targets
    # --------------------------------------------------------

    print_section(
        "[5] VED correlation with mechanical properties"
    )

    for target in targets:

        correlation_value = (
            df["VED_J_mm3"]
            .corr(df[target])
        )

        print(
            f"VED vs {target}: "
            f"{correlation_value:.4f}"
        )

    # --------------------------------------------------------
    # 7. Detect statistical outliers using IQR
    # --------------------------------------------------------

    print_section(
        "[6] Potential statistical outliers — IQR method"
    )

    for column in analysis_columns:

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = (
            (df[column] < lower)
            | (df[column] > upper)
        )

        count = mask.sum()

        print(
            f"{column:25} "
            f"outliers={count:3} "
            f"range=[{lower:.3f}, {upper:.3f}]"
        )

    # --------------------------------------------------------
    # 8. Investigate highest YS records
    # --------------------------------------------------------

    print_section(
        "[7] Highest Yield Strength records"
    )

    high_ys = (
        df[
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
            "YS_MPa",
            ascending=False
        )
        .head(10)
    )

    print(
        high_ys.to_string(index=False)
    )

    # --------------------------------------------------------
    # 9. Source-level statistics
    # --------------------------------------------------------

    print_section(
        "[8] Source-level statistics"
    )

    source_summary = (
        df.groupby("Source_ID")
        .agg(
            Records=("Experiment_ID", "count"),
            UTS_Mean=("UTS_MPa", "mean"),
            UTS_Min=("UTS_MPa", "min"),
            UTS_Max=("UTS_MPa", "max"),
            YS_Mean=("YS_MPa", "mean"),
            YS_Min=("YS_MPa", "min"),
            YS_Max=("YS_MPa", "max"),
            EF_Mean=("Elongation_pct", "mean"),
            EF_Min=("Elongation_pct", "min"),
            EF_Max=("Elongation_pct", "max"),
        )
        .sort_values(
            "Records",
            ascending=False
        )
    )

    print(
        source_summary.round(3).to_string()
    )

    # --------------------------------------------------------
    # 10. Repeated process conditions
    # --------------------------------------------------------

    print_section(
        "[9] Repeated process conditions"
    )

    process_condition_columns = [
        "Powder_Size_um",
        "Laser_Spot_um",
        "Laser_Power_W",
        "Scanning_Speed_mm_s",
        "Hatch_Distance_um",
        "Layer_Thickness_um",
    ]

    grouped = (
        df.groupby(process_condition_columns)
        .size()
        .reset_index(name="Record_Count")
    )

    repeated = (
        grouped[
            grouped["Record_Count"] > 1
        ]
        .sort_values(
            "Record_Count",
            ascending=False
        )
    )

    print(
        f"Unique process conditions: {len(grouped)}"
    )

    print(
        f"Repeated process conditions: "
        f"{len(repeated)}"
    )

    if len(repeated) > 0:
        print(
            "\nMost repeated conditions:"
        )

        print(
            repeated.head(20)
            .to_string(index=False)
        )

    # --------------------------------------------------------
    # 11. Target ranges
    # --------------------------------------------------------

    print_section(
        "[10] Target ranges"
    )

    for target in targets:

        minimum = df[target].min()
        maximum = df[target].max()

        print(
            f"{target:20} "
            f"{minimum:.3f} → {maximum:.3f}"
        )

    # --------------------------------------------------------
    # 12. Final EDA summary
    # --------------------------------------------------------

    print_section(
        "EDA COMPLETE"
    )

    print(
        "No rows were modified or removed."
    )

    print(
        "No model training was performed."
    )

    print(
        "This analysis is descriptive only."
    )


if __name__ == "__main__":
    main()