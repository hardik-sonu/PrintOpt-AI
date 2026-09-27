from pathlib import Path
import pandas as pd


# ============================================================
# PRINTOPT AI — DATASET INSPECTION
# ============================================================

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "ti64_lpbf_dataset.xlsx"


def main():
    print("=" * 80)
    print("PRINTOPT AI — Ti-6Al-4V LPBF DATASET INSPECTION")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Check whether dataset exists
    # --------------------------------------------------------
    print("\n[1] Checking dataset path...")

    if not DATA_FILE.exists():
        print(f"ERROR: Dataset not found at:")
        print(DATA_FILE)
        return

    print(f"Dataset found:")
    print(DATA_FILE)

    # --------------------------------------------------------
    # 2. Load Excel file
    # --------------------------------------------------------
    print("\n[2] Loading Excel dataset...")

    df = pd.read_excel(DATA_FILE)

    print("Dataset loaded successfully.")

    # --------------------------------------------------------
    # 3. Basic dimensions
    # --------------------------------------------------------
    print("\n[3] Dataset dimensions")
    print("-" * 80)

    rows, columns = df.shape

    print(f"Rows    : {rows}")
    print(f"Columns : {columns}")

    # --------------------------------------------------------
    # 4. Column names
    # --------------------------------------------------------
    print("\n[4] Column names")
    print("-" * 80)

    for index, column in enumerate(df.columns, start=1):
        print(f"{index:2}. {column}")

    # --------------------------------------------------------
    # 5. Data types
    # --------------------------------------------------------
    print("\n[5] Data types")
    print("-" * 80)

    print(df.dtypes)

    # --------------------------------------------------------
    # 6. Missing values
    # --------------------------------------------------------
    print("\n[6] Missing values")
    print("-" * 80)

    missing = df.isnull().sum()

    missing_table = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Missing %": (missing.values / len(df) * 100).round(2)
    })

    print(missing_table.to_string(index=False))

    # --------------------------------------------------------
    # 7. Duplicate rows
    # --------------------------------------------------------
    print("\n[7] Duplicate rows")
    print("-" * 80)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")

    # --------------------------------------------------------
    # 8. First 10 rows
    # --------------------------------------------------------
    print("\n[8] First 10 rows")
    print("-" * 80)

    print(df.head(10).to_string(index=False))

    # --------------------------------------------------------
    # 9. Numerical statistics
    # --------------------------------------------------------
    print("\n[9] Numerical statistics")
    print("-" * 80)

    numerical_df = df.select_dtypes(include="number")

    if numerical_df.empty:
        print("No numerical columns found.")
    else:
        statistics = numerical_df.describe().T

        statistics["median"] = numerical_df.median()

        statistics = statistics[
            [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "median",
                "50%",
                "75%",
                "max",
            ]
        ]

        print(statistics.round(3).to_string())

    # --------------------------------------------------------
    # 10. Unique values for every column
    # --------------------------------------------------------
    print("\n[10] Unique values")
    print("-" * 80)

    for column in df.columns:
        unique_count = df[column].nunique(dropna=False)

        print(f"\n{column}")
        print(f"Unique values: {unique_count}")

        # Print values only when there aren't too many
        if unique_count <= 20:
            print(df[column].unique())

    # --------------------------------------------------------
    # 11. Potential constant columns
    # --------------------------------------------------------
    print("\n[11] Constant columns")
    print("-" * 80)

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]

    if constant_columns:
        print("Potential constant columns:")
        for column in constant_columns:
            print(f"- {column}")
    else:
        print("No constant columns found.")

    # --------------------------------------------------------
    # 12. Correlation matrix for numerical columns
    # --------------------------------------------------------
    print("\n[12] Numerical correlation matrix")
    print("-" * 80)

    if len(numerical_df.columns) >= 2:
        correlation = numerical_df.corr()

        print(correlation.round(3).to_string())
    else:
        print("Not enough numerical columns for correlation analysis.")

    # --------------------------------------------------------
    # 13. Important engineering columns check
    # --------------------------------------------------------
    print("\n[13] Expected engineering columns")
    print("-" * 80)

    expected_columns = [
        "Laser Power",
        "Scan Speed",
        "Layer Thickness",
        "Hatch Spacing",
        "Laser Spot",
        "Powder Size",
        "UTS",
        "Yield Strength",
        "EF",
    ]

    actual_columns_lower = {
        column.strip().lower(): column
        for column in df.columns
    }

    for expected in expected_columns:
        if expected.lower() in actual_columns_lower:
            actual = actual_columns_lower[expected.lower()]
            print(f"[FOUND] {actual}")
        else:
            print(f"[NOT FOUND] {expected}")

    # --------------------------------------------------------
    # 14. Final summary
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

    print(f"Rows: {rows}")
    print(f"Columns: {columns}")
    print(f"Duplicate rows: {duplicate_count}")
    print(f"Total missing cells: {df.isnull().sum().sum()}")

    print("\nNo preprocessing or model training was performed.")
    print("This step only inspected the original dataset.")


if __name__ == "__main__":
    main()