"""
Inference test suite.

These tests validate the reusable inference layer and frozen model artifacts.
They do not load or evaluate the held-out test datasets.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import (  # noqa: E402
    AI4I_EARLY_WARNING_THRESHOLD,
    AI4I_PRIMARY_THRESHOLD,
    get_expected_features,
    load_models,
    predict_ai4i,
    predict_rt_iot2022,
)


EXPECTED_AI4I_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def test_final_model_artifacts_exist() -> None:
    """The two frozen V0.5 model artifacts must be present."""
    model_dir = PROJECT_ROOT / "models" / "final"

    assert (model_dir / "ai4i_xgboost_final.joblib").exists()
    assert (model_dir / "rt_iot2022_random_forest_final.joblib").exists()


def test_models_load() -> None:
    """Both frozen models must load successfully."""
    ai4i_model, rt_iot_model = load_models()

    assert ai4i_model is not None
    assert rt_iot_model is not None


def test_input_schemas_match_frozen_models() -> None:
    """Verify the public inference schemas come from the saved model artifacts."""
    schemas = get_expected_features()

    assert schemas["ai4i"] == EXPECTED_AI4I_FEATURES
    assert len(schemas["rt_iot2022"]) == 78


def test_ai4i_prediction_shape_and_range() -> None:
    """AI4I should return a bounded probability and consistent decisions."""
    sample = {
        "Type": "L",
        "Air temperature": 298.2,
        "Process temperature": 308.5,
        "Rotational speed": 1500.0,
        "Torque": 42.0,
        "Tool wear": 120.0,
    }

    result = predict_ai4i(sample)

    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["primary_threshold"] == AI4I_PRIMARY_THRESHOLD
    assert result["early_warning_threshold"] == AI4I_EARLY_WARNING_THRESHOLD

    assert result["primary_failure"] == (
        result["failure_probability"] >= AI4I_PRIMARY_THRESHOLD
    )
    assert result["early_warning"] == (
        result["failure_probability"] >= AI4I_EARLY_WARNING_THRESHOLD
    )

    assert result["primary_status"] in {"NORMAL", "HIGH RISK"}
    assert result["early_warning_status"] in {
        "NO EARLY WARNING",
        "EARLY WARNING",
    }


def test_ai4i_dataframe_input_matches_dict_interface() -> None:
    """The two supported input interfaces should produce the same prediction."""
    sample = {
        "Type": "M",
        "Air temperature": 300.0,
        "Process temperature": 310.0,
        "Rotational speed": 1400.0,
        "Torque": 50.0,
        "Tool wear": 80.0,
    }

    dict_result = predict_ai4i(sample)
    frame_result = predict_ai4i(pd.DataFrame([sample]))

    assert frame_result["failure_probability"] == pytest.approx(
        dict_result["failure_probability"]
    )
    assert frame_result["primary_failure"] == dict_result["primary_failure"]
    assert frame_result["early_warning"] == dict_result["early_warning"]


def test_ai4i_missing_feature_is_rejected() -> None:
    """Missing AI4I inputs must fail with a useful validation error."""
    sample = {
        "Type": "L",
        "Air temperature": 298.2,
        "Process temperature": 308.5,
        "Rotational speed": 1500.0,
        "Torque": 42.0,
        # Tool wear intentionally omitted
    }

    with pytest.raises(ValueError, match="missing required features"):
        predict_ai4i(sample)


def test_ai4i_extra_feature_is_rejected() -> None:
    """Unexpected AI4I inputs must fail rather than being silently ignored."""
    sample = {
        "Type": "L",
        "Air temperature": 298.2,
        "Process temperature": 308.5,
        "Rotational speed": 1500.0,
        "Torque": 42.0,
        "Tool wear": 120.0,
        "Machine failure": 0,
    }

    with pytest.raises(ValueError, match="unexpected features"):
        predict_ai4i(sample)


def test_rt_iot_prediction_shape_and_range() -> None:
    """RT-IoT2022 should accept its 78-feature numeric schema."""
    schema = get_expected_features()["rt_iot2022"]
    sample = pd.DataFrame([{feature: 0.0 for feature in schema}])

    result = predict_rt_iot2022(sample)

    assert len(schema) == 78
    assert 0.0 <= result["attack_probability"] <= 1.0
    assert result["attack_threshold"] == 0.50
    assert result["classification"] in {"NORMAL", "ATTACK"}
    assert result["risk_level"] in {
        "LOW RISK",
        "MEDIUM RISK",
        "HIGH RISK",
    }

    assert result["attack"] == (result["attack_probability"] >= 0.50)


def test_rt_iot_missing_feature_is_rejected() -> None:
    """Missing RT-IoT features must be rejected."""
    schema = get_expected_features()["rt_iot2022"]
    sample = {feature: 0.0 for feature in schema[:-1]}

    with pytest.raises(ValueError, match="missing required features"):
        predict_rt_iot2022(sample)


def test_rt_iot_extra_feature_is_rejected() -> None:
    """Unexpected RT-IoT features must be rejected."""
    schema = get_expected_features()["rt_iot2022"]
    sample = {feature: 0.0 for feature in schema}
    sample["unexpected_feature"] = 1.0

    with pytest.raises(ValueError, match="unexpected features"):
        predict_rt_iot2022(sample)
