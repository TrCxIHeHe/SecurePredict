"""
Reusable inference layer.

Keeps model loading/prediction logic separate from the Streamlit UI.
Uses the final frozen model artifacts produced in V0.5.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .validation import validate_ai4i_row, validate_rt_iot_row


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "final"

AI4I_MODEL_PATH = MODEL_DIR / "ai4i_xgboost_final.joblib"
RT_IOT_MODEL_PATH = MODEL_DIR / "rt_iot2022_random_forest_final.joblib"

AI4I_PRIMARY_THRESHOLD = 0.77
AI4I_EARLY_WARNING_THRESHOLD = 0.07


def _ensure_model(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}\n"
            "Make sure the final V0.5 models exist under models/final/."
        )


def _model_feature_names(model: Any) -> list[str]:
    """Return feature names expected by the fitted pipeline/model."""
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]

    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        for step in reversed(list(named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return [str(x) for x in step.feature_names_in_]

    raise AttributeError(
        "Could not determine feature names from the fitted model artifact."
    )


@lru_cache(maxsize=1)
def load_models() -> tuple[Any, Any]:
    """Load both final models once per Python process."""
    _ensure_model(AI4I_MODEL_PATH)
    _ensure_model(RT_IOT_MODEL_PATH)

    ai4i_model = joblib.load(AI4I_MODEL_PATH)
    rt_iot_model = joblib.load(RT_IOT_MODEL_PATH)
    return ai4i_model, rt_iot_model


def get_expected_features() -> dict[str, list[str]]:
    """Expose the exact input schema expected by the frozen models."""
    ai4i_model, rt_iot_model = load_models()
    return {
        "ai4i": _model_feature_names(ai4i_model),
        "rt_iot2022": _model_feature_names(rt_iot_model),
    }


def _to_single_row(data: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        if len(data) != 1:
            raise ValueError("Inference expects exactly one input row.")
        return data.copy()

    if isinstance(data, dict):
        return pd.DataFrame([data])

    raise TypeError("Input must be a dict or a one-row pandas DataFrame.")


def _validate_columns(
    row: pd.DataFrame, expected: list[str], system_name: str
) -> pd.DataFrame:
    actual = list(row.columns)
    expected_set = set(expected)
    actual_set = set(actual)

    missing = [c for c in expected if c not in actual_set]
    extra = [c for c in actual if c not in expected_set]

    if missing:
        raise ValueError(
            f"{system_name}: missing required features ({len(missing)}): "
            + ", ".join(missing)
        )

    if extra:
        raise ValueError(
            f"{system_name}: unexpected features ({len(extra)}): "
            + ", ".join(extra)
        )

    return row[expected]


def predict_ai4i(data: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
    """
    Predict machine failure risk.
    Primary decision threshold: 0.77
    Early-warning threshold: 0.07
    """
    ai4i_model, _ = load_models()
    expected = _model_feature_names(ai4i_model)
    row = _validate_columns(
        _to_single_row(data), expected, "AI4I"
    )
    validate_ai4i_row(row)

    failure_probability = float(ai4i_model.predict_proba(row)[0, 1])

    primary_failure = failure_probability >= AI4I_PRIMARY_THRESHOLD
    early_warning = failure_probability >= AI4I_EARLY_WARNING_THRESHOLD

    if primary_failure:
        primary_status = "HIGH RISK"
    else:
        primary_status = "NORMAL"

    if early_warning:
        early_warning_status = "EARLY WARNING"
    else:
        early_warning_status = "NO EARLY WARNING"

    return {
        "failure_probability": failure_probability,
        "primary_threshold": AI4I_PRIMARY_THRESHOLD,
        "primary_failure": bool(primary_failure),
        "primary_status": primary_status,
        "early_warning_threshold": AI4I_EARLY_WARNING_THRESHOLD,
        "early_warning": bool(early_warning),
        "early_warning_status": early_warning_status,
    }


def predict_rt_iot2022(
    data: dict[str, Any] | pd.DataFrame
) -> dict[str, Any]:
    """Predict whether a network-flow sample is an attack."""
    _, rt_iot_model = load_models()
    expected = _model_feature_names(rt_iot_model)
    row = _validate_columns(
        _to_single_row(data), expected, "RT-IoT2022"
    )
    validate_rt_iot_row(row)

    attack_probability = float(rt_iot_model.predict_proba(row)[0, 1])
    attack = attack_probability >= 0.50

    if attack_probability >= 0.80:
        risk_level = "HIGH RISK"
    elif attack_probability >= 0.50:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"

    return {
        "attack_probability": attack_probability,
        "attack_threshold": 0.50,
        "attack": bool(attack),
        "classification": "ATTACK" if attack else "NORMAL",
        "risk_level": risk_level,
    }


def _validated_ai4i_row(data):
    ai4i_model, _ = load_models()
    row = _validate_columns(
        _to_single_row(data), _model_feature_names(ai4i_model), "AI4I"
    )
    validate_ai4i_row(row)
    return row


def _validated_rt_row(data):
    _, rt_model = load_models()
    row = _validate_columns(
        _to_single_row(data), _model_feature_names(rt_model), "RT-IoT2022"
    )
    validate_rt_iot_row(row)
    return row


def explain_ai4i(data: dict[str, Any] | pd.DataFrame, top_k: int = 8) -> dict[str, Any]:
    """SHAP contributions (log-odds) for THIS machine reading."""
    from .explain import explain_row

    return explain_row("ai4i", _validated_ai4i_row(data), top_k)


def explain_rt_iot2022(data: dict[str, Any] | pd.DataFrame, top_k: int = 8) -> dict[str, Any]:
    """SHAP contributions (probability) for THIS network flow."""
    from .explain import explain_row

    return explain_row("rt_iot2022", _validated_rt_row(data), top_k)
