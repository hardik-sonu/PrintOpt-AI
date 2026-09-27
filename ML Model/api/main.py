from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import csv
from pathlib import Path
from collections import Counter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_v2.joblib"
)

PROCESS_MAP_DATASET_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "ti64_lpbf_processed.csv"
)


# ============================================================
# MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# PROCESS MAP DATASET
# ============================================================

PROCESS_MAP_COLUMNS = [
    "Experiment_ID",
    "Source_ID",
    "VED_J_mm3",
    "UTS_MPa",
    "YS_MPa",
    "Elongation_pct",
]


def load_process_map_dataset() -> pd.DataFrame:
    """
    Load the verified processed Ti-6Al-4V LPBF dataset used
    for the experimental Process Map.

    This is intentionally separate from train/test splits.
    The Process Map represents the available experimental data.
    """

    if not PROCESS_MAP_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Process Map dataset was not found at: "
            f"{PROCESS_MAP_DATASET_PATH}"
        )

    df = pd.read_csv(
        PROCESS_MAP_DATASET_PATH
    )

    missing_columns = [
        column
        for column in PROCESS_MAP_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Process Map dataset is missing required columns: "
            f"{missing_columns}"
        )

    return df[PROCESS_MAP_COLUMNS].copy()


# Load once when the API starts.
process_map_dataset = load_process_map_dataset()


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# OPTIMIZATION CONSTANTS
# ============================================================

# The current Optimization UI exposes four process variables.
#
# Powder size and laser spot are required model inputs but are not
# currently exposed as optimization variables in the frontend.
#
# Therefore they remain fixed during the optimization search.
#
# These values should later be connected to explicit user inputs
# when the Optimization UI is expanded to support them.

DEFAULT_POWDER_SIZE_UM = 35.0
DEFAULT_LASER_SPOT_UM = 81.0

OPTIMIZATION_CANDIDATES = 10000


# ============================================================
# MODEL EVALUATION DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

BASELINE_RESULTS_FILE = EVALUATION_DIR / "baseline_results_v2.csv"
FEATURE_IMPORTANCE_FILE = EVALUATION_DIR / "baseline_feature_importance_v2.csv"
TEST_PREDICTIONS_FILE = EVALUATION_DIR / "test_predictions.csv"


def read_csv_file(path: Path) -> list[dict]:
    """Read a CSV file and return rows as dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_float(value):
    """Convert a CSV value to float while preserving missing values."""
    if value is None or value == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_model_metrics():
    """Load baseline V2 evaluation metrics."""
    rows = read_csv_file(BASELINE_RESULTS_FILE)

    metrics = []

    for row in rows:
        metrics.append(
            {
                "model_name": row["Model"],
                "target": row["Target"],
                "r2": safe_float(row["R2"]),
                "mae": safe_float(row["MAE"]),
                "rmse": safe_float(row["RMSE"]),
            }
        )

    return metrics


def load_feature_importance():
    """Load baseline V2 feature importance."""
    rows = read_csv_file(FEATURE_IMPORTANCE_FILE)

    importance = []

    for row in rows:
        importance.append(
            {
                "model_name": row["Model"],
                "target": row["Target"],
                "feature": row["Feature"],
                "importance": safe_float(row["Importance"]),
            }
        )

    return importance


def load_test_predictions():
    """Load held-out test predictions."""
    rows = read_csv_file(TEST_PREDICTIONS_FILE)

    predictions = []

    for row in rows:
        predictions.append(
            {
                "experiment_id": row["Experiment_ID"],
                "source_id": row["Source_ID"],
                "laser_power_w": safe_float(row["Laser_Power_W"]),
                "scanning_speed_mm_s": safe_float(
                    row["Scanning_Speed_mm_s"]
                ),
                "hatch_distance_um": safe_float(
                    row["Hatch_Distance_um"]
                ),
                "layer_thickness_um": safe_float(
                    row["Layer_Thickness_um"]
                ),
                "ved_j_mm3": safe_float(row["VED_J_mm3"]),

                "uts_actual": safe_float(row["UTS_MPa"]),
                "ys_actual": safe_float(row["YS_MPa"]),
                "elongation_actual": safe_float(
                    row["Elongation_pct"]
                ),

                "rf_uts_pred": safe_float(row["RF_UTS_Pred"]),
                "rf_ys_pred": safe_float(row["RF_YS_Pred"]),
                "rf_elongation_pred": safe_float(
                    row["RF_Elongation_Pred"]
                ),

                "xgb_uts_pred": safe_float(row["XGB_UTS_Pred"]),
                "xgb_ys_pred": safe_float(row["XGB_YS_Pred"]),
                "xgb_elongation_pred": safe_float(
                    row["XGB_Elongation_Pred"]
                ),

                "ved_outside_training_range":
                    row["VED_Outside_Training_Range"].lower()
                    == "true",
            }
        )

    return predictions


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="PrintOpt AI ML API",
    description=(
        "AI-driven Ti-6Al-4V LPBF process-property "
        "prediction, optimization, and experimental "
        "process-map API"
    ),
    version="1.2.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST SCHEMAS
# ============================================================


class PredictionRequest(BaseModel):
    powder_size_um: float = Field(
        ...,
        gt=0,
        description="Powder particle size in micrometers",
    )

    laser_spot_um: float = Field(
        ...,
        gt=0,
        description="Laser spot size in micrometers",
    )

    laser_power_w: float = Field(
        ...,
        gt=0,
        description="Laser power in watts",
    )

    scanning_speed_mm_s: float = Field(
        ...,
        gt=0,
        description="Scanning speed in millimeters per second",
    )

    hatch_distance_um: float = Field(
        ...,
        gt=0,
        description="Hatch distance in micrometers",
    )

    layer_thickness_um: float = Field(
        ...,
        gt=0,
        description="Layer thickness in micrometers",
    )


class OptimizationConstraint(BaseModel):
    min: float = Field(
        ...,
        gt=0,
        description="Minimum allowed parameter value",
    )

    max: float = Field(
        ...,
        gt=0,
        description="Maximum allowed parameter value",
    )


class OptimizationConstraints(BaseModel):
    laser_power_w: OptimizationConstraint
    scan_speed_mm_s: OptimizationConstraint
    layer_thickness_um: OptimizationConstraint
    hatch_spacing_um: OptimizationConstraint


class OptimizationObjective(BaseModel):
    target: str = Field(
        ...,
        description="Optimization target",
    )


class OptimizationRequest(BaseModel):
    objective: OptimizationObjective
    constraints: OptimizationConstraints


# ============================================================
# RESPONSE HELPERS
# ============================================================


def calculate_ved(
    laser_power_w: float,
    scanning_speed_mm_s: float,
    hatch_distance_um: float,
    layer_thickness_um: float,
) -> float:
    """
    Volumetric Energy Density:

        VED = P / (v * h * t)

    Dataset units:
        P = W
        v = mm/s
        h = um
        t = um

    Hatch and layer thickness are converted to mm.
    """

    hatch_distance_mm = (
        hatch_distance_um / 1000.0
    )

    layer_thickness_mm = (
        layer_thickness_um / 1000.0
    )

    denominator = (
        scanning_speed_mm_s
        * hatch_distance_mm
        * layer_thickness_mm
    )

    if denominator <= 0:
        raise ValueError(
            "Scanning speed, hatch distance, and layer "
            "thickness must all be greater than zero."
        )

    ved = laser_power_w / denominator

    return float(ved)


def build_model_input(
    powder_size_um: float,
    laser_spot_um: float,
    laser_power_w: float,
    scanning_speed_mm_s: float,
    hatch_distance_um: float,
    layer_thickness_um: float,
) -> pd.DataFrame:
    """
    Build model input using the exact feature order expected
    by Random Forest v2.
    """

    ved = calculate_ved(
        laser_power_w=laser_power_w,
        scanning_speed_mm_s=scanning_speed_mm_s,
        hatch_distance_um=hatch_distance_um,
        layer_thickness_um=layer_thickness_um,
    )

    return pd.DataFrame(
        [
            {
                "Powder_Size_um": powder_size_um,
                "Laser_Spot_um": laser_spot_um,
                "Laser_Power_W": laser_power_w,
                "Scanning_Speed_mm_s": scanning_speed_mm_s,
                "Hatch_Distance_um": hatch_distance_um,
                "Layer_Thickness_um": layer_thickness_um,
                "VED_J_mm3": ved,
            }
        ],
        columns=FEATURES,
    )


def predict_with_model(
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and Random Forest tree-to-tree
    prediction standard deviations.

    The saved model is a MultiOutputRegressor containing
    three RandomForestRegressor models.

    The DataFrame is preserved so the model receives the same
    named feature structure used during training.
    """

    predictions = model.predict(features)[0]

    uncertainties = []

    for estimator in model.estimators_:
        tree_predictions = np.array(
            [
                tree.predict(features)[0]
                for tree in estimator.estimators_
            ],
            dtype=float,
        )

        uncertainties.append(
            float(
                np.std(tree_predictions)
            )
        )

    return (
        np.asarray(
            predictions,
            dtype=float,
        ),
        np.asarray(
            uncertainties,
            dtype=float,
        ),
    )


# ============================================================
# OPTIMIZATION HELPERS
# ============================================================


def validate_constraint(
    name: str,
    constraint: OptimizationConstraint,
) -> None:
    """
    Validate an optimization parameter range.
    """

    if not np.isfinite(
        constraint.min
    ):
        raise ValueError(
            f"{name} minimum must be a finite number."
        )

    if not np.isfinite(
        constraint.max
    ):
        raise ValueError(
            f"{name} maximum must be a finite number."
        )

    if constraint.min <= 0:
        raise ValueError(
            f"{name} minimum must be greater than zero."
        )

    if constraint.max <= 0:
        raise ValueError(
            f"{name} maximum must be greater than zero."
        )

    if constraint.min >= constraint.max:
        raise ValueError(
            f"{name} minimum must be lower than maximum."
        )


def normalize_values(
    values: np.ndarray,
) -> np.ndarray:
    """
    Min-max normalization used to place different mechanical
    properties onto a common 0-1 optimization scale.
    """

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum - minimum <= 1e-12:
        return np.zeros_like(
            values,
            dtype=float,
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def calculate_objective_scores(
    predictions: np.ndarray,
    target: str,
) -> np.ndarray:
    """
    Calculate optimization score for each candidate.

    Single-objective:
        maximize UTS
        maximize Yield Strength
        maximize Elongation

    Multi-objective:
        equal-weight combination of all three normalized targets.
    """

    uts = predictions[:, 0]

    yield_strength = predictions[:, 1]

    elongation = predictions[:, 2]

    uts_normalized = normalize_values(
        uts
    )

    yield_normalized = normalize_values(
        yield_strength
    )

    elongation_normalized = normalize_values(
        elongation
    )

    if target == "maximize_uts":
        return uts_normalized

    if target == "maximize_yield_strength":
        return yield_normalized

    if target == "maximize_elongation":
        return elongation_normalized

    if target == "multi_objective":
        return (
            (uts_normalized / 3.0)
            + (yield_normalized / 3.0)
            + (elongation_normalized / 3.0)
        )

    raise ValueError(
        "Unsupported optimization target. "
        "Use maximize_uts, maximize_yield_strength, "
        "maximize_elongation, or multi_objective."
    )


def generate_optimization_candidates(
    constraints: OptimizationConstraints,
    candidate_count: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate random process-parameter candidates inside the
    user-defined constraint bounds.

    Candidate column order:

        0 = Laser Power
        1 = Scan Speed
        2 = Hatch Spacing
        3 = Layer Thickness
    """

    rng = np.random.default_rng(
        seed
    )

    laser_power = rng.uniform(
        constraints.laser_power_w.min,
        constraints.laser_power_w.max,
        candidate_count,
    )

    scan_speed = rng.uniform(
        constraints.scan_speed_mm_s.min,
        constraints.scan_speed_mm_s.max,
        candidate_count,
    )

    hatch_spacing = rng.uniform(
        constraints.hatch_spacing_um.min,
        constraints.hatch_spacing_um.max,
        candidate_count,
    )

    layer_thickness = rng.uniform(
        constraints.layer_thickness_um.min,
        constraints.layer_thickness_um.max,
        candidate_count,
    )

    return np.column_stack(
        [
            laser_power,
            scan_speed,
            hatch_spacing,
            layer_thickness,
        ]
    )


def build_optimization_dataframe(
    candidates: np.ndarray,
) -> pd.DataFrame:
    """
    Convert optimization candidates into the exact seven-feature
    DataFrame expected by Random Forest v2.
    """

    laser_power = candidates[:, 0]

    scan_speed = candidates[:, 1]

    hatch_spacing = candidates[:, 2]

    layer_thickness = candidates[:, 3]

    ved = (
        laser_power
        / (
            scan_speed
            * (hatch_spacing / 1000.0)
            * (layer_thickness / 1000.0)
        )
    )

    return pd.DataFrame(
        {
            "Powder_Size_um": np.full(
                len(candidates),
                DEFAULT_POWDER_SIZE_UM,
            ),
            "Laser_Spot_um": np.full(
                len(candidates),
                DEFAULT_LASER_SPOT_UM,
            ),
            "Laser_Power_W": laser_power,
            "Scanning_Speed_mm_s": scan_speed,
            "Hatch_Distance_um": hatch_spacing,
            "Layer_Thickness_um": layer_thickness,
            "VED_J_mm3": ved,
        },
        columns=FEATURES,
    )


def calculate_selected_uncertainty(
    features: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate Random Forest tree-to-tree spread for the selected
    optimization candidate.

    This is model spread, not a calibrated probability of correctness.
    """

    uncertainties = []

    for estimator in model.estimators_:
        tree_predictions = np.array(
            [
                tree.predict(features)[0]
                for tree in estimator.estimators_
            ],
            dtype=float,
        )

        uncertainties.append(
            float(
                np.std(tree_predictions)
            )
        )

    return {
        "uts_mpa": round(
            uncertainties[0],
            2,
        ),
        "yield_strength_mpa": round(
            uncertainties[1],
            2,
        ),
        "elongation_pct": round(
            uncertainties[2],
            2,
        ),
    }


# ============================================================
# ROUTES
# ============================================================


@app.get("/")
def root():
    return {
        "name": "PrintOpt AI ML API",
        "status": "running",
        "model": "Random Forest v2",
        "material": "Ti-6Al-4V",
        "process": "LPBF",
        "capabilities": [
            "prediction",
            "optimization",
            "process_map",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "process_map_loaded": True,
        "process_map_records": int(
            len(process_map_dataset)
        ),
    }


# ============================================================
# PREDICTION
# ============================================================


@app.post("/predict")
def predict(
    request: PredictionRequest,
):

    # --------------------------------------------------------
    # Build model input
    # --------------------------------------------------------

    input_data = build_model_input(
        powder_size_um=request.powder_size_um,
        laser_spot_um=request.laser_spot_um,
        laser_power_w=request.laser_power_w,
        scanning_speed_mm_s=request.scanning_speed_mm_s,
        hatch_distance_um=request.hatch_distance_um,
        layer_thickness_um=request.layer_thickness_um,
    )

    ved = float(
        input_data[
            "VED_J_mm3"
        ].iloc[0]
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions, uncertainties = (
        predict_with_model(
            input_data
        )
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "input": {
            "powder_size_um": request.powder_size_um,
            "laser_spot_um": request.laser_spot_um,
            "laser_power_w": request.laser_power_w,
            "scanning_speed_mm_s": (
                request.scanning_speed_mm_s
            ),
            "hatch_distance_um": (
                request.hatch_distance_um
            ),
            "layer_thickness_um": (
                request.layer_thickness_um
            ),
            "ved_j_mm3": round(
                ved,
                4,
            ),
        },
        "prediction": {
            "uts_mpa": round(
                float(predictions[0]),
                2,
            ),
            "yield_strength_mpa": round(
                float(predictions[1]),
                2,
            ),
            "elongation_pct": round(
                float(predictions[2]),
                2,
            ),
        },
        "uncertainty": {
            "uts_mpa": round(
                float(uncertainties[0]),
                2,
            ),
            "yield_strength_mpa": round(
                float(uncertainties[1]),
                2,
            ),
            "elongation_pct": round(
                float(uncertainties[2]),
                2,
            ),
        },
        "model": {
            "name": "Random Forest v2",
            "material": "Ti-6Al-4V",
            "process": "LPBF",
            "features": FEATURES,
        },
    }


# ============================================================
# PROCESS MAP
# ============================================================


@app.get("/api/process-map")
def process_map():
    """
    Return the experimental data required by the Process Map.

    Source:
        data/processed/ti64_lpbf_processed.csv

    The dataset contains 173 experimental records and includes:
        - VED
        - UTS
        - Yield Strength
        - Elongation
        - Experiment ID
        - Source ID

    Null mechanical-property values are preserved as null so the
    frontend can omit those points from the corresponding plot.
    """

    df = process_map_dataset.copy()

    # --------------------------------------------------------
    # Ensure numeric fields are numeric
    # --------------------------------------------------------

    numeric_columns = [
        "VED_J_mm3",
        "UTS_MPa",
        "YS_MPa",
        "Elongation_pct",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Build response arrays
    # --------------------------------------------------------

    ved_values = []

    uts_values = []

    yield_strength_values = []

    elongation_values = []

    record_ids = []

    sources = []

    for _, row in df.iterrows():

        ved = row["VED_J_mm3"]

        uts = row["UTS_MPa"]

        ys = row["YS_MPa"]

        elongation = row[
            "Elongation_pct"
        ]

        experiment_id = row[
            "Experiment_ID"
        ]

        source_id = row[
            "Source_ID"
        ]

        # ----------------------------------------------------
        # Skip records without a valid VED.
        #
        # VED is the X-axis and therefore required for every
        # Process Map point.
        # ----------------------------------------------------

        if pd.isna(ved):
            continue

        ved_values.append(
            float(ved)
        )

        # ----------------------------------------------------
        # Preserve missing target values as None.
        # FastAPI serializes None as JSON null.
        # ----------------------------------------------------

        if pd.isna(uts):
            uts_values.append(None)
        else:
            uts_values.append(
                float(uts)
            )

        if pd.isna(ys):
            yield_strength_values.append(
                None
            )
        else:
            yield_strength_values.append(
                float(ys)
            )

        if pd.isna(elongation):
            elongation_values.append(
                None
            )
        else:
            elongation_values.append(
                float(elongation)
            )

        # ----------------------------------------------------
        # Dataset identifiers
        # ----------------------------------------------------

        if pd.isna(experiment_id):
            record_ids.append(
                f"Record {len(record_ids) + 1}"
            )
        else:
            record_ids.append(
                f"EXP-{int(experiment_id):03d}"
            )

        if pd.isna(source_id):
            sources.append(
                "Unknown source"
            )
        else:
            sources.append(
                str(source_id)
            )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "ved_values": ved_values,
        "uts_values": uts_values,
        "yield_strength_values": (
            yield_strength_values
        ),
        "elongation_values": (
            elongation_values
        ),
        "record_ids": record_ids,
        "sources": sources,
    }


@app.get("/api/dataset")
def get_dataset():
    dataset_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ti64_lpbf_processed.csv"
    )

    try:
        rows = read_csv_file(dataset_path)

        records = []

        for row in rows:
            records.append({
                "record_id": row["Experiment_ID"],
                "laser_power_w": safe_float(row["Laser_Power_W"]),
                "scan_speed_mm_s": safe_float(
                    row["Scanning_Speed_mm_s"]
                ),
                "layer_thickness_um": safe_float(
                    row["Layer_Thickness_um"]
                ),
                "hatch_spacing_um": safe_float(
                    row["Hatch_Distance_um"]
                ),
                "laser_spot_um": safe_float(
                    row["Laser_Spot_um"]
                ),
                "powder_size_um": safe_float(
                    row["Powder_Size_um"]
                ),
                "ved_j_mm3": safe_float(
                    row["VED_J_mm3"]
                ),
                "uts_mpa": safe_float(
                    row["UTS_MPa"]
                ),
                "yield_strength_mpa": safe_float(
                    row["YS_MPa"]
                ),
                "elongation_pct": safe_float(
                    row["Elongation_pct"]
                ),
                "source": row["Source_ID"],
            })

        return records

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


@app.get("/api/sources")
def get_sources():
    try:
        dataset_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "ti64_lpbf_processed.csv"
        )

        rows = read_csv_file(dataset_path)

        source_records = {}

        for row in rows:
            source_id = (
                row.get("Source_ID")
                or "UNKNOWN"
            ).strip()

            reference = (
                row.get("Reference")
                or "Unknown reference"
            ).strip()

            if source_id not in source_records:
                source_records[source_id] = {
                    "id": source_id,
                    "title": reference,
                    "authors": [],
                    "year": None,
                    "doi": None,
                    "source_type": "other",
                    "record_count": 0,
                    "journal": None,
                }

            source_records[source_id]["record_count"] += 1

        return list(source_records.values())

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


# ============================================================
# OPTIMIZATION
# ============================================================


@app.post("/api/optimize")
def optimize(
    request: OptimizationRequest,
):

    # --------------------------------------------------------
    # Validate objective
    # --------------------------------------------------------

    allowed_targets = {
        "maximize_uts",
        "maximize_yield_strength",
        "maximize_elongation",
        "multi_objective",
    }

    if request.objective.target not in allowed_targets:
        raise ValueError(
            "Unsupported optimization target."
        )

    # --------------------------------------------------------
    # Validate parameter constraints
    # --------------------------------------------------------

    validate_constraint(
        "Laser Power",
        request.constraints.laser_power_w,
    )

    validate_constraint(
        "Scan Speed",
        request.constraints.scan_speed_mm_s,
    )

    validate_constraint(
        "Layer Thickness",
        request.constraints.layer_thickness_um,
    )

    validate_constraint(
        "Hatch Spacing",
        request.constraints.hatch_spacing_um,
    )

    # --------------------------------------------------------
    # Generate candidate parameter combinations
    # --------------------------------------------------------

    candidates = (
        generate_optimization_candidates(
            constraints=request.constraints,
            candidate_count=(
                OPTIMIZATION_CANDIDATES
            ),
            seed=42,
        )
    )

    # --------------------------------------------------------
    # Build model input for every candidate
    # --------------------------------------------------------

    candidate_features = (
        build_optimization_dataframe(
            candidates
        )
    )

    # --------------------------------------------------------
    # Predict all candidates
    # --------------------------------------------------------

    predictions = model.predict(
        candidate_features
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if predictions.ndim != 2:
        raise ValueError(
            "Unexpected model output shape."
        )

    if predictions.shape[1] != 3:
        raise ValueError(
            "Random Forest v2 must return three outputs: "
            "UTS, Yield Strength, and Elongation."
        )

    # --------------------------------------------------------
    # Calculate objective scores
    # --------------------------------------------------------

    scores = calculate_objective_scores(
        predictions=predictions,
        target=request.objective.target,
    )

    # --------------------------------------------------------
    # Select best candidate
    # --------------------------------------------------------

    best_index = int(
        np.argmax(scores)
    )

    best_candidate = candidates[
        best_index
    ]

    best_prediction = predictions[
        best_index
    ]

    # --------------------------------------------------------
    # Extract recommended parameters
    # --------------------------------------------------------

    recommended_laser_power = float(
        best_candidate[0]
    )

    recommended_scan_speed = float(
        best_candidate[1]
    )

    recommended_hatch_spacing = float(
        best_candidate[2]
    )

    recommended_layer_thickness = float(
        best_candidate[3]
    )

    # --------------------------------------------------------
    # Calculate recommended VED
    # --------------------------------------------------------

    recommended_ved = calculate_ved(
        laser_power_w=(
            recommended_laser_power
        ),
        scanning_speed_mm_s=(
            recommended_scan_speed
        ),
        hatch_distance_um=(
            recommended_hatch_spacing
        ),
        layer_thickness_um=(
            recommended_layer_thickness
        ),
    )

    # --------------------------------------------------------
    # Calculate selected-candidate uncertainty
    # --------------------------------------------------------

    selected_features = (
        candidate_features.iloc[
            [best_index]
        ]
    )

    uncertainty = (
        calculate_selected_uncertainty(
            selected_features
        )
    )

    # --------------------------------------------------------
    # Search-space VED statistics
    # --------------------------------------------------------

    all_ved = candidate_features[
        "VED_J_mm3"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {
        "recommended_parameters": {
            "laser_power_w": round(
                recommended_laser_power,
                2,
            ),
            "scan_speed_mm_s": round(
                recommended_scan_speed,
                2,
            ),
            "layer_thickness_um": round(
                recommended_layer_thickness,
                2,
            ),
            "hatch_spacing_um": round(
                recommended_hatch_spacing,
                2,
            ),
        },

        "predicted_properties": {
            "uts_mpa": round(
                float(best_prediction[0]),
                2,
            ),
            "yield_strength_mpa": round(
                float(best_prediction[1]),
                2,
            ),
            "elongation_pct": round(
                float(best_prediction[2]),
                2,
            ),
        },

        "uncertainty": uncertainty,

        # RF tree spread is not calibrated probability.
        "confidence": None,

        "objective": {
            "target": request.objective.target,
            "score": round(
                float(
                    scores[best_index]
                ),
                6,
            ),
        },

        "optimization": {
            "method": (
                "bounded_random_search"
            ),
            "candidate_count": (
                OPTIMIZATION_CANDIDATES
            ),
            "random_seed": 42,
        },

        "fixed_parameters": {
            "powder_size_um": (
                DEFAULT_POWDER_SIZE_UM
            ),
            "laser_spot_um": (
                DEFAULT_LASER_SPOT_UM
            ),
        },

        "ved": {
            "recommended_j_mm3": round(
                recommended_ved,
                4,
            ),
            "search_min_j_mm3": round(
                float(
                    np.min(all_ved)
                ),
                4,
            ),
            "search_max_j_mm3": round(
                float(
                    np.max(all_ved)
                ),
                4,
            ),
        },

        "model": {
            "name": "Random Forest v2",
            "material": "Ti-6Al-4V",
            "process": "LPBF",
            "features": FEATURES,
        },
    }


# ============================================================
# MODEL EVALUATION ENDPOINTS
# ============================================================

@app.get("/api/model/status")
def get_model_status():
    """Return current model availability and evaluation status."""

    return {
        "is_connected": True,
        "models_available": [
            "random_forest_v2",
            "xgboost_v2",
            "mlp_v2",
        ],
        "active_model": "random_forest_v2",
        "model_name": "Random Forest V2",
        "material": "Ti-6Al-4V",
        "process": "Laser Powder Bed Fusion",
        "dataset_size": 173,
        "evaluation_samples": 28,
        "evaluation_type": "source-aware held-out evaluation",
    }


@app.get("/api/model/metrics")
def get_model_metrics():
    """Return baseline V2 model evaluation metrics."""

    try:
        return load_model_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )


@app.get("/api/model/diagnostics")
def get_model_diagnostics():
    """Return feature importance and held-out prediction data."""

    try:
        return {
            "feature_importance": load_feature_importance(),
            "predictions": load_test_predictions(),
        }

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )