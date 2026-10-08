"""V1.0.1 - value-level input validation (NaN, Inf, bad Type, out of range)."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from inference import get_expected_features, predict_ai4i, predict_rt_iot2022  # noqa: E402

GOOD = {
    "Type": "L",
    "Air temperature": 298.2,
    "Process temperature": 308.5,
    "Rotational speed": 1500.0,
    "Torque": 42.0,
    "Tool wear": 120.0,
}


def _with(**changes):
    row = dict(GOOD)
    row.update({k.replace("_", " "): v for k, v in changes.items()})
    return row


def test_valid_row_still_works():
    assert 0.0 <= predict_ai4i(GOOD)["failure_probability"] <= 1.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_ai4i_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="NaN|infinite"):
        predict_ai4i(_with(Air_temperature=bad))


@pytest.mark.parametrize("bad", ["X", "l", "", None, 3])
def test_ai4i_rejects_invalid_type(bad):
    with pytest.raises(ValueError, match="'Type'"):
        predict_ai4i(_with(Type=bad))


@pytest.mark.parametrize(
    "field,value",
    [
        ("Air_temperature", -5000),
        ("Process_temperature", 10_000),
        ("Rotational_speed", -1),
        ("Torque", 1e6),
        ("Tool_wear", -10),
    ],
)
def test_ai4i_rejects_out_of_range(field, value):
    with pytest.raises(ValueError, match="outside the accepted range"):
        predict_ai4i(_with(**{field: value}))


@pytest.mark.parametrize("bad", ["298.2", True, None])
def test_ai4i_rejects_non_numeric(bad):
    with pytest.raises(ValueError, match="must be a number"):
        predict_ai4i(_with(Torque=bad))


def test_rt_rejects_nan_inf_and_text():
    schema = get_expected_features()["rt_iot2022"]
    for bad, pat in [(float("nan"), "NaN"), (float("inf"), "infinite"), ("abc", "must be a number")]:
        row = {f: 0.0 for f in schema}
        row[schema[3]] = bad
        with pytest.raises(ValueError, match=pat):
            predict_rt_iot2022(row)


def test_rt_rejects_absurd_magnitude():
    schema = get_expected_features()["rt_iot2022"]
    row = {f: 0.0 for f in schema}
    row[schema[0]] = 1e15
    with pytest.raises(ValueError, match="magnitude"):
        predict_rt_iot2022(row)
