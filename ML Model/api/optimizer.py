from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model feature contract
# ---------------------------------------------------------------------------
#
# Random Forest V2 was trained using these seven process/derived features.
#
# The optimizer exposes four of them to the user:
#   - Laser Power
#   - Scan Speed
#   - Hatch Distance
#   - Layer Thickness
#
# Powder size and laser spot remain fixed because they are not exposed as
# optimization constraints in the current frontend.
#
# ---------------------------------------------------------------------------

MODEL_FEATURES = [
    "Powder_Size_um",
    "Laser_Spot_um",
    "Laser_Power_W",
    "Scanning_Speed_mm_s",
    "Hatch_Distance_um",
    "Layer_Thickness_um",
    "VED_J_mm3",
]


# These are representative fixed values for the two parameters that are not
# currently exposed by the optimization UI.
#
# They are deliberately kept fixed instead of pretending that the optimizer
# has searched over them.
#
# These can later be replaced by explicit user constraints once the UI
# supports them.
DEFAULT_POWDER_SIZE_UM = 35.0
DEFAULT_LASER_SPOT_UM = 81.0


OptimizationTarget = Literal[
    "maximize_uts",
    "maximize_yield_strength",
    "maximize_elongation",
    "multi_objective",
]


def calculate_ved(
    laser_power_w: float,
    scan_speed_mm_s: float,
    hatch_spacing_um: float,
    layer_thickness_um: float,
) -> float:
    """
    Calculate volumetric energy density.

    VED = P / (v * h * t)

    P  = laser power [W = J/s]
    v  = scan speed [mm/s]
    h  = hatch spacing [mm]
    t  = layer thickness [mm]

    Returns:
        VED in J/mm^3
    """

    hatch_mm = hatch_spacing_um / 1000.0
    layer_mm = layer_thickness_um / 1000.0

    denominator = (
        scan_speed_mm_s
        * hatch_mm
        * layer_mm
    )

    if denominator <= 0:
        raise ValueError(
            "Scan speed, hatch spacing, and layer thickness "
            "must all be greater than zero."
        )

    return laser_power_w / denominator


def _build_feature_frame(
    candidates: np.ndarray,
) -> pd.DataFrame:
    """
    Convert optimizer candidates into the exact feature schema expected
    by the trained Random Forest model.

    candidates columns:
        0 = laser power
        1 = scan speed
        2 = hatch spacing
        3 = layer thickness
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
        columns=MODEL_FEATURES,
    )


def _predict(
    model: Any,
    candidates: np.ndarray,
) -> np.ndarray:
    """
    Predict all three mechanical-property targets.

    Expected model output:
        [UTS, YS, Elongation]
    """

    features = _build_feature_frame(candidates)

    predictions = model.predict(features)

    predictions = np.asarray(predictions)

    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    if predictions.shape[1] != 3:
        raise ValueError(
            "The loaded optimization model must return "
            "three targets: UTS, YS, and Elongation."
        )

    return predictions


def _normalize(values: np.ndarray) -> np.ndarray:
    """
    Min-max normalize an array.

    If all values are identical, return zeros rather than dividing by zero.
    """

    minimum = np.min(values)
    maximum = np.max(values)

    if maximum - minimum <= 1e-12:
        return np.zeros_like(values, dtype=float)

    return (values - minimum) / (maximum - minimum)


def _objective_scores(
    predictions: np.ndarray,
    target: OptimizationTarget,
) -> np.ndarray:
    """
    Convert predicted properties into an optimization score.

    For single objectives, the corresponding normalized target is maximized.

    For multi-objective optimization, UTS, YS, and elongation receive equal
    weights. This is deliberately transparent and can later be replaced by
    user-defined weights in the frontend.
    """

    uts = predictions[:, 0]
    ys = predictions[:, 1]
    elongation = predictions[:, 2]

    uts_norm = _normalize(uts)
    ys_norm = _normalize(ys)
    elongation_norm = _normalize(elongation)

    if target == "maximize_uts":
        return uts_norm

    if target == "maximize_yield_strength":
        return ys_norm

    if target == "maximize_elongation":
        return elongation_norm

    if target == "multi_objective":
        return (
            0.333333 * uts_norm
            + 0.333333 * ys_norm
            + 0.333334 * elongation_norm
        )

    raise ValueError(
        f"Unsupported optimization target: {target}"
    )


def optimize_process(
    model: Any,
    objective: OptimizationTarget,
    constraints: dict[str, dict[str, float]],
    *,
    candidate_count: int = 10000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Perform constrained model-based process optimization.

    The optimizer samples candidate process combinations only inside the
    user-provided parameter bounds.

    This is intentionally a bounded model-search optimizer rather than a
    physics simulator or a claim of experimentally validated optimum.
    """

    required_keys = [
        "laser_power_w",
        "scan_speed_mm_s",
        "layer_thickness_um",
        "hatch_spacing_um",
    ]

    for key in required_keys:
        if key not in constraints:
            raise ValueError(
                f"Missing optimization constraint: {key}"
            )

        minimum = float(constraints[key]["min"])
        maximum = float(constraints[key]["max"])

        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(
                f"Invalid constraint values for {key}."
            )

        if minimum >= maximum:
            raise ValueError(
                f"Minimum must be lower than maximum for {key}."
            )

    rng = np.random.default_rng(random_seed)

    # -----------------------------------------------------------------------
    # Generate bounded candidates
    # -----------------------------------------------------------------------

    laser_power = rng.uniform(
        constraints["laser_power_w"]["min"],
        constraints["laser_power_w"]["max"],
        candidate_count,
    )

    scan_speed = rng.uniform(
        constraints["scan_speed_mm_s"]["min"],
        constraints["scan_speed_mm_s"]["max"],
        candidate_count,
    )

    layer_thickness = rng.uniform(
        constraints["layer_thickness_um"]["min"],
        constraints["layer_thickness_um"]["max"],
        candidate_count,
    )

    hatch_spacing = rng.uniform(
        constraints["hatch_spacing_um"]["min"],
        constraints["hatch_spacing_um"]["max"],
        candidate_count,
    )

    candidates = np.column_stack(
        [
            laser_power,
            scan_speed,
            hatch_spacing,
            layer_thickness,
        ]
    )

    # -----------------------------------------------------------------------
    # Model inference
    # -----------------------------------------------------------------------

    predictions = _predict(
        model,
        candidates,
    )

    scores = _objective_scores(
        predictions,
        objective,
    )

    best_index = int(np.argmax(scores))

    best_parameters = candidates[best_index]
    best_prediction = predictions[best_index]

    best_laser_power = float(best_parameters[0])
    best_scan_speed = float(best_parameters[1])
    best_hatch_spacing = float(best_parameters[2])
    best_layer_thickness = float(best_parameters[3])

    best_ved = calculate_ved(
        best_laser_power,
        best_scan_speed,
        best_hatch_spacing,
        best_layer_thickness,
    )

    # -----------------------------------------------------------------------
    # Candidate-space summary
    # -----------------------------------------------------------------------

    candidate_ved = (
        candidates[:, 0]
        / (
            candidates[:, 1]
            * (candidates[:, 2] / 1000.0)
            * (candidates[:, 3] / 1000.0)
        )
    )

    # -----------------------------------------------------------------------
    # Optional RF tree-level variation
    # -----------------------------------------------------------------------
    #
    # This is NOT presented as statistical confidence.
    #
    # It is simply the spread of individual RF estimators around the selected
    # candidate. It can later be exposed as engineering uncertainty.
    #
    # -----------------------------------------------------------------------

    model_uncertainty = None

    estimators = getattr(model, "estimators_", None)

    if estimators is not None:
        selected_frame = _build_feature_frame(
            best_parameters.reshape(1, -1)
        )

        tree_predictions = np.column_stack(
            [
                estimator.predict(selected_frame)[0]
                for estimator in estimators
            ]
        )

        if tree_predictions.shape[1] == 3:
            model_uncertainty = {
                "uts_mpa": float(
                    np.std(tree_predictions[:, 0])
                ),
                "yield_strength_mpa": float(
                    np.std(tree_predictions[:, 1])
                ),
                "elongation_pct": float(
                    np.std(tree_predictions[:, 2])
                ),
            }

    # -----------------------------------------------------------------------
    # Return frontend-compatible result
    # -----------------------------------------------------------------------

    result: dict[str, Any] = {
        "recommended_parameters": {
            "laser_power_w": round(
                best_laser_power,
                3,
            ),
            "scan_speed_mm_s": round(
                best_scan_speed,
                3,
            ),
            "layer_thickness_um": round(
                best_layer_thickness,
                3,
            ),
            "hatch_spacing_um": round(
                best_hatch_spacing,
                3,
            ),
        },

        "predicted_properties": {
            "uts_mpa": round(
                float(best_prediction[0]),
                3,
            ),
            "yield_strength_mpa": round(
                float(best_prediction[1]),
                3,
            ),
            "elongation_pct": round(
                float(best_prediction[2]),
                3,
            ),
        },

        "objective": {
            "target": objective,
            "score": round(
                float(scores[best_index]),
                6,
            ),
        },

        "optimization": {
            "candidate_count": int(candidate_count),
            "search_method": "bounded_random_search",
            "random_seed": random_seed,
        },

        "fixed_parameters": {
            "powder_size_um": DEFAULT_POWDER_SIZE_UM,
            "laser_spot_um": DEFAULT_LASER_SPOT_UM,
        },

        "search_summary": {
            "ved_min_j_mm3": round(
                float(np.min(candidate_ved)),
                3,
            ),
            "ved_max_j_mm3": round(
                float(np.max(candidate_ved)),
                3,
            ),
            "recommended_ved_j_mm3": round(
                float(best_ved),
                3,
            ),
        },
    }

    if model_uncertainty is not None:
        result["uncertainty"] = model_uncertainty

    return result