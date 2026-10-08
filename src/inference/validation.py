"""
Input validation for the inference layer (V1.0.1).

Order of checks:  schema -> type -> finite -> domain/range.
Every failure raises ValueError with a message naming the offending feature.
"""

from __future__ import annotations

import math
import numbers
from typing import Any

import pandas as pd

AI4I_TYPES = ("L", "M", "H")

# Hard physical/plausibility bounds (inclusive). Values outside are rejected.
# They match the dashboard form limits so UI and API agree.
AI4I_BOUNDS: dict[str, tuple[float, float]] = {
    "Air temperature": (270.0, 330.0),       # Kelvin
    "Process temperature": (280.0, 340.0),   # Kelvin
    "Rotational speed": (500.0, 3000.0),     # rpm
    "Torque": (0.0, 100.0),                  # Nm
    "Tool wear": (0.0, 300.0),               # minutes
}

# RT-IoT2022 flow features: no per-feature physical range is claimed, but values
# must be real finite numbers of sane magnitude.
RT_ABS_LIMIT = 1e12


def _as_number(value: Any, name: str, system: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            f"{system}: '{name}' must be a number, got {type(value).__name__} ({value!r})"
        )
    value = float(value)
    if math.isnan(value):
        raise ValueError(f"{system}: '{name}' is NaN")
    if math.isinf(value):
        raise ValueError(f"{system}: '{name}' is infinite")
    return value


def validate_ai4i_row(row: pd.DataFrame) -> None:
    """Validate a single-row AI4I frame whose columns already match the schema."""
    r = row.iloc[0]
    machine_type = r["Type"]
    if not isinstance(machine_type, str) or machine_type not in AI4I_TYPES:
        raise ValueError(
            f"AI4I: 'Type' must be one of {list(AI4I_TYPES)}, got {machine_type!r}"
        )
    for name, (lo, hi) in AI4I_BOUNDS.items():
        value = _as_number(r[name], name, "AI4I")
        if not lo <= value <= hi:
            raise ValueError(
                f"AI4I: '{name}'={value} is outside the accepted range [{lo}, {hi}]"
            )


def validate_rt_iot_row(row: pd.DataFrame) -> None:
    """Validate a single-row RT-IoT2022 frame (78 numeric flow features)."""
    r = row.iloc[0]
    for name in row.columns:
        value = _as_number(r[name], name, "RT-IoT2022")
        if abs(value) > RT_ABS_LIMIT:
            raise ValueError(
                f"RT-IoT2022: '{name}'={value} exceeds the sane magnitude limit {RT_ABS_LIMIT:g}"
            )


# ---------------------------------------------------------------------------
# Vectorised variants for batch inference (same rules, whole frame at once).
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402


def _numeric_block(frame: pd.DataFrame, cols: list[str], system: str) -> np.ndarray:
    for c in cols:
        if frame[c].dtype == bool or not pd.api.types.is_numeric_dtype(frame[c]):
            raise ValueError(f"{system}: column '{c}' must be numeric")
    arr = frame[cols].to_numpy(dtype=float)
    bad = ~np.isfinite(arr)
    if bad.any():
        r, c = np.argwhere(bad)[0]
        raise ValueError(f"{system}: row {r}: '{cols[c]}' is NaN or infinite")
    return arr


def validate_ai4i_frame(frame: pd.DataFrame) -> None:
    bad_type = ~frame["Type"].isin(AI4I_TYPES)
    if bad_type.any():
        i = int(np.argmax(bad_type.to_numpy()))
        raise ValueError(f"AI4I: row {i}: 'Type' must be one of {list(AI4I_TYPES)}, got {frame['Type'].iloc[i]!r}")
    cols = list(AI4I_BOUNDS)
    arr = _numeric_block(frame, cols, "AI4I")
    for j, name in enumerate(cols):
        lo, hi = AI4I_BOUNDS[name]
        out = (arr[:, j] < lo) | (arr[:, j] > hi)
        if out.any():
            i = int(np.argmax(out))
            raise ValueError(f"AI4I: row {i}: '{name}'={arr[i, j]} is outside the accepted range [{lo}, {hi}]")


def validate_rt_iot_frame(frame: pd.DataFrame) -> None:
    arr = _numeric_block(frame, list(frame.columns), "RT-IoT2022")
    big = np.abs(arr) > RT_ABS_LIMIT
    if big.any():
        r, c = np.argwhere(big)[0]
        raise ValueError(f"RT-IoT2022: row {r}: '{frame.columns[c]}' exceeds the sane magnitude limit {RT_ABS_LIMIT:g}")
