from pathlib import Path

import pandas as pd


# ============================================================
# PRINTOPT AI — STEP 8B
# SOURCE VARIATION ANALYSIS
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 8B")
print("SOURCE VARIATION ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# PATH
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
    / "source_variation.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)


# ------------------------------------------------------------
# COLUMNS
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


# ------------------------------------------------------------
# SOURCE SUMMARY
# ------------------------------------------------------------

summary = (
    df
    .groupby("Source_ID")
    .agg(
        Records=("Experiment_ID", "count"),

        Powder_Size_Min=("Powder_Size_um", "min"),
        Powder_Size_Max=("Powder_Size_um", "max"),

        Laser_Spot_Min=("Laser_Spot_um", "min"),
        Laser_Spot_Max=("Laser_Spot_um", "max"),

        Laser_Power_Min=("Laser_Power_W", "min"),
        Laser_Power_Max=("Laser_Power_W", "max"),

        Scan_Speed_Min=("Scanning_Speed_mm_s", "min"),
        Scan_Speed_Max=("Scanning_Speed_mm_s", "max"),

        Hatch_Min=("Hatch_Distance_um", "min"),
        Hatch_Max=("Hatch_Distance_um", "max"),

        Layer_Min=("Layer_Thickness_um", "min"),
        Layer_Max=("Layer_Thickness_um", "max"),

        VED_Min=("VED_J_mm3", "min"),
        VED_Max=("VED_J_mm3", "max"),

        UTS_Mean=("UTS_MPa", "mean"),
        UTS_Std=("UTS_MPa", "std"),

        YS_Mean=("YS_MPa", "mean"),
        YS_Std=("YS_MPa", "std"),

        Elongation_Mean=("Elongation_pct", "mean"),
        Elongation_Std=("Elongation_pct", "std"),
    )
    .sort_values(
        "Records",
        ascending=False
    )
)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    240
)

pd.set_option(
    "display.float_format",
    "{:.3f}".format
)


print()
print("SOURCE SUMMARY")
print("-" * 70)

print(
    summary.to_string()
)


# ------------------------------------------------------------
# GLOBAL STATISTICS
# ------------------------------------------------------------

print()
print("=" * 70)
print("GLOBAL TARGET STATISTICS")
print("=" * 70)

for target in TARGET_COLUMNS:

    print()
    print(target)

    print(
        f"  Global mean : "
        f"{df[target].mean():.3f}"
    )

    print(
        f"  Global std  : "
        f"{df[target].std():.3f}"
    )

    print(
        f"  Global min  : "
        f"{df[target].min():.3f}"
    )

    print(
        f"  Global max  : "
        f"{df[target].max():.3f}"
    )


# ------------------------------------------------------------
# SOURCE MEAN RANGE
# ------------------------------------------------------------

print()
print("=" * 70)
print("SOURCE MEAN RANGES")
print("=" * 70)

for target in TARGET_COLUMNS:

    source_means = (
        df
        .groupby("Source_ID")[target]
        .mean()
    )

    print()
    print(target)

    print(
        f"  Lowest source mean  : "
        f"{source_means.min():.3f}"
    )

    print(
        f"  Highest source mean : "
        f"{source_means.max():.3f}"
    )

    print(
        f"  Difference           : "
        f"{source_means.max() - source_means.min():.3f}"
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

summary.to_csv(
    OUTPUT_PATH
)

print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(OUTPUT_PATH)

print()
print("STEP 8B COMPLETE")
print("=" * 70)