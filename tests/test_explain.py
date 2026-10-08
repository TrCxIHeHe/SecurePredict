"""V1.0.2 - per-input SHAP explanations."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from inference import (  # noqa: E402
    explain_ai4i,
    explain_rt_iot2022,
    get_expected_features,
)

SAMPLE = {
    "Type": "L",
    "Air temperature": 298.2,
    "Process temperature": 308.5,
    "Rotational speed": 1500.0,
    "Torque": 42.0,
    "Tool wear": 120.0,
}


def test_ai4i_explanation_structure():
    out = explain_ai4i(SAMPLE, top_k=5)
    assert out["output_space"] == "log-odds"
    assert len(out["contributions"]) == 5
    for c in out["contributions"]:
        assert {"feature", "value", "shap"} <= set(c)
    mags = [abs(c["shap"]) for c in out["contributions"]]
    assert mags == sorted(mags, reverse=True)


def test_ai4i_explanation_is_input_specific():
    high = dict(SAMPLE, Torque=70.0, **{"Tool wear": 240.0})
    a = {c["feature"]: c["shap"] for c in explain_ai4i(SAMPLE, 8)["contributions"]}
    b = {c["feature"]: c["shap"] for c in explain_ai4i(high, 8)["contributions"]}
    assert a != b


def test_ai4i_explanation_rejects_bad_input():
    with pytest.raises(ValueError):
        explain_ai4i(dict(SAMPLE, Type="X"))


def test_rt_explanation_structure():
    schema = get_expected_features()["rt_iot2022"]
    out = explain_rt_iot2022({f: 0.0 for f in schema}, top_k=6)
    assert out["output_space"] == "probability"
    assert len(out["contributions"]) == 6
