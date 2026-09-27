from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PRINTOPT AI — STEP 9G
# BASELINE MODEL ERROR DIAGNOSTICS V2
# ============================================================

print("=" * 70)
print("PRINTOPT AI — STEP 9G")
print("BASELINE MODEL ERROR DIAGNOSTICS V2")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "baseline_predictions_v2.csv"
)

RESULTS_FILE = (
    BASE_DIR
    / "evaluation"
    / "baseline_results_v2.csv"
)

OUTPUT_DIR = BASE_DIR / "evaluation"

SOURCE_ERROR_FILE = (
    OUTPUT_DIR
    / "baseline_source_error_v2.csv"
)

EXPERIMENT_ERROR_FILE = (
    OUTPUT_DIR
    / "baseline_experiment_error_v2.csv"
)

TARGET_BIAS_FILE = (
    OUTPUT_DIR
    / "baseline_target_bias_v2.csv"
)

REGION_ERROR_FILE = (
    OUTPUT_DIR
    / "baseline_region_error_v2.csv"
)

MODEL_COMPARISON_FILE = (
    OUTPUT_DIR
    / "baseline_model_error_comparison_v2.csv"
)

DIAGNOSTIC_SUMMARY_FILE = (
    OUTPUT_DIR
    / "baseline_diagnostic_summary_v2.csv"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

pred = pd.read_csv(PREDICTIONS_FILE)
results = pd.read_csv(RESULTS_FILE)

print()
print("INPUT FILES")
print("-" * 70)

print(
    f"Prediction records : {len(pred)}"
)

print(
    f"Result rows        : {len(results)}"
)


# ------------------------------------------------------------
# TARGETS
# ------------------------------------------------------------

TARGETS = [
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
]


# ------------------------------------------------------------
# MODEL PREDICTION COLUMNS
# ------------------------------------------------------------

MODEL_COLUMNS = {
    "Random Forest V2": {
        target: f"RF_{target}_Pred"
        for target in TARGETS
    },
    "XGBoost V2": {
        target: f"XGB_{target}_Pred"
        for target in TARGETS
    },
    "MLP V2": {
        target: f"MLP_{target}_Pred"
        for target in TARGETS
    },
}


# ------------------------------------------------------------
# BASIC VALIDATION
# ------------------------------------------------------------

required_columns = [
    "Experiment_ID",
    "Source_ID",
    "VED_J_mm3",
    *TARGETS,
]

for model_columns in MODEL_COLUMNS.values():

    required_columns.extend(
        model_columns.values()
    )

missing_columns = [
    column
    for column in required_columns
    if column not in pred.columns
]

if missing_columns:

    raise ValueError(
        "Missing required prediction columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# 1. INDIVIDUAL EXPERIMENT ERROR
# ============================================================

print()
print("=" * 70)
print("1. INDIVIDUAL EXPERIMENT ERROR")
print("=" * 70)


experiment_error = pred[
    [
        "Experiment_ID",
        "Source_ID",
        "VED_J_mm3",
        *TARGETS,
    ]
].copy()


for model_name, model_targets in MODEL_COLUMNS.items():

    short_name = (
        "RF"
        if model_name == "Random Forest V2"
        else
        "XGB"
        if model_name == "XGBoost V2"
        else
        "MLP"
    )

    for target in TARGETS:

        actual = pred[target]

        predicted = pred[
            model_targets[target]
        ]

        experiment_error[
            f"{short_name}_{target}_Error"
        ] = predicted - actual

        experiment_error[
            f"{short_name}_{target}_AbsError"
        ] = np.abs(
            predicted - actual
        )


# ------------------------------------------------------------
# OVERALL ERROR SCORE
# ------------------------------------------------------------

for model_name in [
    "RF",
    "XGB",
    "MLP",
]:

    absolute_columns = [
        f"{model_name}_{target}_AbsError"
        for target in TARGETS
    ]

    experiment_error[
        f"{model_name}_Mean_Absolute_Error"
    ] = experiment_error[
        absolute_columns
    ].mean(axis=1)


# ------------------------------------------------------------
# SORT BY XGBOOST ERROR
# ------------------------------------------------------------

experiment_error = experiment_error.sort_values(
    "XGB_Mean_Absolute_Error",
    ascending=False,
)

experiment_error.to_csv(
    EXPERIMENT_ERROR_FILE,
    index=False,
)


print()
print("Highest-error experiments — XGBoost")
print("-" * 70)

display_columns = [
    "Experiment_ID",
    "Source_ID",
    "VED_J_mm3",
    "XGB_UTS_MPa_AbsError",
    "XGB_YS_MPa_AbsError",
    "XGB_Elongation_pct_AbsError",
    "XGB_Mean_Absolute_Error",
]

print(
    experiment_error[
        display_columns
    ].head(10).to_string(index=False)
)


# ============================================================
# 2. TARGET BIAS
# ============================================================

print()
print("=" * 70)
print("2. TARGET BIAS / SYSTEMATIC ERROR")
print("=" * 70)


bias_rows = []


for model_name, model_targets in MODEL_COLUMNS.items():

    for target in TARGETS:

        actual = pred[target]

        predicted = pred[
            model_targets[target]
        ]

        error = predicted - actual

        bias_rows.append(
            {
                "Model": model_name,
                "Target": target,
                "Mean_Signed_Error": error.mean(),
                "Mean_Absolute_Error": np.abs(
                    error
                ).mean(),
                "Median_Signed_Error": np.median(
                    error
                ),
                "Std_Error": error.std(),
                "Min_Error": error.min(),
                "Max_Error": error.max(),
            }
        )


bias_df = pd.DataFrame(
    bias_rows
)

bias_df.to_csv(
    TARGET_BIAS_FILE,
    index=False,
)


print(
    bias_df.to_string(index=False)
)


# ============================================================
# 3. ERROR BY SOURCE
# ============================================================

print()
print("=" * 70)
print("3. ERROR BY HELD-OUT SOURCE")
print("=" * 70)


source_rows = []


for source, group in pred.groupby(
    "Source_ID"
):

    row = {
        "Source_ID": source,
        "Records": len(group),
    }

    for model_name, model_targets in MODEL_COLUMNS.items():

        short_name = (
            "RF"
            if model_name == "Random Forest V2"
            else
            "XGB"
            if model_name == "XGBoost V2"
            else
            "MLP"
        )

        for target in TARGETS:

            error = (
                group[
                    model_targets[target]
                ]
                - group[target]
            )

            row[
                f"{short_name}_{target}_MAE"
            ] = np.abs(error).mean()

            row[
                f"{short_name}_{target}_RMSE"
            ] = np.sqrt(
                np.mean(
                    error ** 2
                )
            )

    source_rows.append(row)


source_error_df = pd.DataFrame(
    source_rows
)

source_error_df.to_csv(
    SOURCE_ERROR_FILE,
    index=False,
)


# ------------------------------------------------------------
# DISPLAY XGBOOST SOURCE ERROR
# ------------------------------------------------------------

print()
print("XGBoost source-level error")
print("-" * 70)

xgb_source_display = [
    "Source_ID",
    "Records",
    "XGB_UTS_MPa_MAE",
    "XGB_YS_MPa_MAE",
    "XGB_Elongation_pct_MAE",
]

print(
    source_error_df[
        xgb_source_display
    ].sort_values(
        "XGB_UTS_MPa_MAE",
        ascending=False,
    ).to_string(index=False)
)


# ============================================================
# 4. PROCESS / VED REGION ERROR
# ============================================================

print()
print("=" * 70)
print("4. PROCESS REGION ERROR")
print("=" * 70)


region_df = pred.copy()


# ------------------------------------------------------------
# VED BINS
# ------------------------------------------------------------

region_df["VED_Region"] = pd.cut(
    region_df["VED_J_mm3"],
    bins=[
        -np.inf,
        50,
        75,
        100,
        150,
        np.inf,
    ],
    labels=[
        "<50",
        "50-75",
        "75-100",
        "100-150",
        ">150",
    ],
)


# ------------------------------------------------------------
# LASER POWER BINS
# ------------------------------------------------------------

region_df["Power_Region"] = pd.cut(
    region_df["Laser_Power_W"],
    bins=[
        -np.inf,
        150,
        250,
        350,
        np.inf,
    ],
    labels=[
        "<=150",
        "150-250",
        "250-350",
        ">350",
    ],
)


# ------------------------------------------------------------
# SCAN SPEED BINS
# ------------------------------------------------------------

region_df["ScanSpeed_Region"] = pd.cut(
    region_df["Scanning_Speed_mm_s"],
    bins=[
        -np.inf,
        600,
        1000,
        1400,
        np.inf,
    ],
    labels=[
        "<=600",
        "600-1000",
        "1000-1400",
        ">1400",
    ],
)


# ------------------------------------------------------------
# LAYER THICKNESS BINS
# ------------------------------------------------------------

region_df["Layer_Region"] = pd.cut(
    region_df["Layer_Thickness_um"],
    bins=[
        -np.inf,
        30,
        50,
        70,
        np.inf,
    ],
    labels=[
        "<=30",
        "30-50",
        "50-70",
        ">70",
    ],
)


# ------------------------------------------------------------
# CALCULATE REGION ERRORS
# ------------------------------------------------------------

region_rows = []


region_definitions = [
    ("VED_Region", "VED"),
    ("Power_Region", "Laser Power"),
    ("ScanSpeed_Region", "Scanning Speed"),
    ("Layer_Region", "Layer Thickness"),
]


for column, region_type in region_definitions:

    for region, group in region_df.groupby(
        column,
        observed=True,
    ):

        row = {
            "Region_Type": region_type,
            "Region": str(region),
            "Records": len(group),
        }

        for model_name, model_targets in MODEL_COLUMNS.items():

            short_name = (
                "RF"
                if model_name == "Random Forest V2"
                else
                "XGB"
                if model_name == "XGBoost V2"
                else
                "MLP"
            )

            for target in TARGETS:

                error = (
                    group[
                        model_targets[target]
                    ]
                    - group[target]
                )

                row[
                    f"{short_name}_{target}_MAE"
                ] = np.abs(error).mean()

                row[
                    f"{short_name}_{target}_Bias"
                ] = error.mean()

        region_rows.append(row)


region_error_df = pd.DataFrame(
    region_rows
)

region_error_df.to_csv(
    REGION_ERROR_FILE,
    index=False,
)


# ------------------------------------------------------------
# DISPLAY VED RESULTS
# ------------------------------------------------------------

print()
print("XGBoost error by VED region")
print("-" * 70)

ved_display = region_error_df[
    region_error_df["Region_Type"] == "VED"
][
    [
        "Region",
        "Records",
        "XGB_UTS_MPa_MAE",
        "XGB_YS_MPa_MAE",
        "XGB_Elongation_pct_MAE",
    ]
]

print(
    ved_display.to_string(index=False)
)


# ============================================================
# 5. MODEL ERROR COMPARISON
# ============================================================

print()
print("=" * 70)
print("5. MODEL ERROR COMPARISON")
print("=" * 70)


comparison_rows = []


for target in TARGETS:

    actual = pred[target]

    for model_name, model_targets in MODEL_COLUMNS.items():

        predicted = pred[
            model_targets[target]
        ]

        error = predicted - actual

        comparison_rows.append(
            {
                "Model": model_name,
                "Target": target,
                "MAE": np.abs(
                    error
                ).mean(),
                "RMSE": np.sqrt(
                    np.mean(
                        error ** 2
                    )
                ),
                "Mean_Signed_Error": error.mean(),
                "Median_Signed_Error": np.median(
                    error
                ),
            }
        )


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    MODEL_COMPARISON_FILE,
    index=False,
)


print(
    comparison_df.to_string(index=False)
)


# ============================================================
# 6. DIAGNOSTIC SUMMARY
# ============================================================

print()
print("=" * 70)
print("6. DIAGNOSTIC SUMMARY")
print("=" * 70)


summary_rows = []


for model_name, model_targets in MODEL_COLUMNS.items():

    for target in TARGETS:

        actual = pred[target]

        predicted = pred[
            model_targets[target]
        ]

        error = predicted - actual

        summary_rows.append(
            {
                "Model": model_name,
                "Target": target,
                "Records": len(pred),
                "MAE": np.abs(error).mean(),
                "RMSE": np.sqrt(
                    np.mean(
                        error ** 2
                    )
                ),
                "Mean_Bias": error.mean(),
                "Underprediction_Count": int(
                    (error < 0).sum()
                ),
                "Overprediction_Count": int(
                    (error > 0).sum()
                ),
                "Exact_Zero_Error_Count": int(
                    (error == 0).sum()
                ),
            }
        )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    DIAGNOSTIC_SUMMARY_FILE,
    index=False,
)


print(
    summary_df.to_string(index=False)
)


# ============================================================
# 7. KEY DIAGNOSTIC INDICATORS
# ============================================================

print()
print("=" * 70)
print("KEY DIAGNOSTIC INDICATORS")
print("=" * 70)


# XGBoost highest-error source

worst_xgb_uts_source = (
    source_error_df
    .sort_values(
        "XGB_UTS_MPa_MAE",
        ascending=False,
    )
    .iloc[0]
)

worst_xgb_ys_source = (
    source_error_df
    .sort_values(
        "XGB_YS_MPa_MAE",
        ascending=False,
    )
    .iloc[0]
)

worst_xgb_elong_source = (
    source_error_df
    .sort_values(
        "XGB_Elongation_pct_MAE",
        ascending=False,
    )
    .iloc[0]
)


print(
    "Highest XGBoost UTS source error:"
)

print(
    f"  {worst_xgb_uts_source['Source_ID']} "
    f"(MAE="
    f"{worst_xgb_uts_source['XGB_UTS_MPa_MAE']:.3f}"
    f" MPa)"
)


print(
    "Highest XGBoost YS source error:"
)

print(
    f"  {worst_xgb_ys_source['Source_ID']} "
    f"(MAE="
    f"{worst_xgb_ys_source['XGB_YS_MPa_MAE']:.3f}"
    f" MPa)"
)


print(
    "Highest XGBoost elongation source error:"
)

print(
    f"  {worst_xgb_elong_source['Source_ID']} "
    f"(MAE="
    f"{worst_xgb_elong_source['XGB_Elongation_pct_MAE']:.3f}"
    f" %)"
)


# ------------------------------------------------------------
# XGBOOST BIAS
# ------------------------------------------------------------

print()

xgb_bias = bias_df[
    bias_df["Model"] == "XGBoost V2"
]

for _, row in xgb_bias.iterrows():

    direction = (
        "underprediction"
        if row["Mean_Signed_Error"] < 0
        else
        "overprediction"
        if row["Mean_Signed_Error"] > 0
        else
        "no systematic bias"
    )

    print(
        f"XGBoost {row['Target']}: "
        f"{direction}, "
        f"mean signed error="
        f"{row['Mean_Signed_Error']:.3f}"
    )


# ============================================================
# FILES CREATED
# ============================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    SOURCE_ERROR_FILE
)

print(
    EXPERIMENT_ERROR_FILE
)

print(
    TARGET_BIAS_FILE
)

print(
    REGION_ERROR_FILE
)

print(
    MODEL_COMPARISON_FILE
)

print(
    DIAGNOSTIC_SUMMARY_FILE
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("STEP 9G COMPLETE")
print("=" * 70)

print(
    f"Prediction records analyzed : {len(pred)}"
)

print(
    f"Sources analyzed            : "
    f"{pred['Source_ID'].nunique()}"
)

print(
    "No model or dataset files were modified."
)