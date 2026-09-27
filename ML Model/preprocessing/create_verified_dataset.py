from pathlib import Path
import pandas as pd


# ============================================================
# PRINTOPT AI — STEP 9C
# CREATE VERIFIED / MODELING DATASET
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "ti64_lpbf_processed.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
EVALUATION_DIR = BASE_DIR / "evaluation"

VERIFIED_FILE = OUTPUT_DIR / "ti64_lpbf_verified.csv"
FLAGS_FILE = EVALUATION_DIR / "data_quality_flags.csv"


print("=" * 70)
print("PRINTOPT AI — STEP 9C")
print("CREATE VERIFIED / MODELING DATASET")
print("=" * 70)


# ============================================================
# 1. LOAD ORIGINAL PROCESSED DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Processed dataset not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nINPUT DATASET")
print("-" * 70)
print(f"Records : {len(df)}")
print(f"Columns : {len(df.columns)}")
print(f"Sources : {df['Source_ID'].nunique()}")


# ============================================================
# 2. PRESERVE ORIGINAL DATA
# ============================================================

verified = df.copy()


# ============================================================
# 3. INITIAL STATUS
# ============================================================

verified["Data_Status"] = "UNREVIEWED"
verified["Verification_Note"] = ""


# ============================================================
# 4. MARK ALL RECORDS AS CURRENTLY USABLE
# ============================================================
#
# Important:
# We are NOT deleting suspicious records.
#
# They remain available for traceability.
#
# "CURRENT" means the record remains in the modeling dataset,
# but may still carry a quality flag.
# ============================================================

verified["Data_Status"] = "CURRENT"


# ============================================================
# 5. FLAG YS > UTS
# ============================================================

ys_gt_uts = verified["YS_MPa"] > verified["UTS_MPa"]

verified.loc[ys_gt_uts, "Data_Status"] = "REVIEW_REQUIRED"

verified.loc[
    ys_gt_uts,
    "Verification_Note"
] = (
    "Yield strength exceeds UTS; original source verification required."
)


# ============================================================
# 6. EXPERIMENT 63 — POSSIBLE UTS/YS COLUMN SWAP
# ============================================================
#
# We do NOT change the values.
#
# We only record the possibility so the original measurement
# remains preserved.
# ============================================================

mask_63 = verified["Experiment_ID"] == 63

verified.loc[
    mask_63,
    "Data_Status"
] = "REVIEW_REQUIRED"

verified.loc[
    mask_63,
    "Verification_Note"
] = (
    "Possible UTS/YS column-order issue identified during "
    "source verification; values intentionally unchanged pending "
    "direct verification of the original source table."
)


# ============================================================
# 7. REF_24 — SOURCE DATASET MISMATCH
# ============================================================
#
# Four REF_24 records have YS > UTS.
#
# Additionally, the current dataset contains YS values up to
# 1700 MPa. These require direct source verification.
#
# Again: DO NOT modify values.
# ============================================================

ref24_problem = (
    (verified["Source_ID"] == "REF_24")
    & (verified["YS_MPa"] > verified["UTS_MPa"])
)

verified.loc[
    ref24_problem,
    "Data_Status"
] = "REVIEW_REQUIRED"

verified.loc[
    ref24_problem,
    "Verification_Note"
] = (
    "REF_24 record requires direct source verification; "
    "YS exceeds UTS and target value should not be altered "
    "without checking the original source."
)


# ============================================================
# 8. HIGH VED FLAG
# ============================================================

ved_threshold = verified["VED_J_mm3"].quantile(0.95)

high_ved = verified["VED_J_mm3"] > ved_threshold

for idx in verified.index[high_ved]:

    existing_note = verified.at[idx, "Verification_Note"]

    high_ved_note = (
        f"VED above 95th percentile ({ved_threshold:.3f} J/mm³); "
        "domain-extreme record retained."
    )

    if existing_note:
        verified.at[idx, "Verification_Note"] = (
            existing_note + " " + high_ved_note
        )
    else:
        verified.at[idx, "Verification_Note"] = high_ved_note


# ============================================================
# 9. CREATE QUALITY FLAG TABLE
# ============================================================

flag_records = []

for _, row in verified.iterrows():

    flags = []

    if row["YS_MPa"] > row["UTS_MPa"]:
        flags.append("YS_GT_UTS")

    if row["Experiment_ID"] == 63:
        flags.append("POSSIBLE_UTS_YS_SWAP")

    if (
        row["Source_ID"] == "REF_24"
        and row["YS_MPa"] > row["UTS_MPa"]
    ):
        flags.append("REF24_SOURCE_REVIEW")

    if row["VED_J_mm3"] > ved_threshold:
        flags.append("HIGH_VED")

    if flags:

        flag_records.append(
            {
                "Experiment_ID": row["Experiment_ID"],
                "Source_ID": row["Source_ID"],
                "Flags": "; ".join(flags),
                "UTS_MPa": row["UTS_MPa"],
                "YS_MPa": row["YS_MPa"],
                "Elongation_pct": row["Elongation_pct"],
                "VED_J_mm3": row["VED_J_mm3"],
                "Verification_Note": row["Verification_Note"],
            }
        )


flags_df = pd.DataFrame(flag_records)


# ============================================================
# 10. SAVE FILES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

verified.to_csv(
    VERIFIED_FILE,
    index=False
)

flags_df.to_csv(
    FLAGS_FILE,
    index=False
)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\nDATA STATUS")
print("-" * 70)

print(
    verified["Data_Status"]
    .value_counts()
    .to_string()
)

print("\nQUALITY FLAGS")
print("-" * 70)

print(f"Flagged records : {len(flags_df)}")
print(f"YS > UTS       : {ys_gt_uts.sum()}")
print(f"High VED       : {high_ved.sum()}")
print(
    f"Possible swap  : {mask_63.sum()}"
)

print("\nREVIEW REQUIRED RECORDS")
print("-" * 70)

review = verified[
    verified["Data_Status"] == "REVIEW_REQUIRED"
][
    [
        "Experiment_ID",
        "Source_ID",
        "UTS_MPa",
        "YS_MPa",
        "Elongation_pct",
        "VED_J_mm3",
        "Verification_Note",
    ]
]

print(review.to_string(index=False))


print("\nFILES CREATED")
print("-" * 70)

print(f"Verified dataset:")
print(VERIFIED_FILE)

print("\nQuality flags:")
print(FLAGS_FILE)


print("\nIMPORTANT")
print("-" * 70)

print(
    "The original processed dataset was NOT modified."
)

print(
    "No suspicious records were deleted."
)

print(
    "No UTS/YS values were automatically changed."
)

print(
    "All questionable records remain traceable."
)

print("\nSTEP 9C COMPLETE")
print("=" * 70)