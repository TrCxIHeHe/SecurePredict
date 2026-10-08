"""
Per-input SHAP explanations (V1.0.2).

Explains the SAME frozen pipelines used for prediction, for the ONE row the
user supplied. Contributions are in the model's native output space:
  - AI4I (XGBoost):       log-odds of failure
  - RT-IoT2022 (RF):      probability of attack
Positive value = pushes toward failure / attack.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import shap


@lru_cache(maxsize=2)
def _explainer(kind: str):
    from .predictors import load_models

    ai4i_model, rt_model = load_models()
    pipeline = ai4i_model if kind == "ai4i" else rt_model
    return shap.TreeExplainer(pipeline.named_steps["model"]), pipeline


def _clean(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


def _transform(pipeline, row: pd.DataFrame):
    pre = pipeline[:-1]  # every step except the final estimator
    X = pre.transform(row)
    if hasattr(X, "toarray"):
        X = X.toarray()
    try:
        names = [_clean(n) for n in pre.get_feature_names_out()]
    except Exception:
        names = list(row.columns)
    return np.asarray(X, dtype=float), names


def explain_row(kind: str, row: pd.DataFrame, top_k: int = 8) -> dict[str, Any]:
    """`row` must already be validated and schema-ordered (see predictors)."""
    explainer, pipeline = _explainer(kind)
    X, names = _transform(pipeline, row)
    values = explainer.shap_values(X)
    if isinstance(values, list):          # older shap: list per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:                  # (n, features, classes)
        values = values[:, :, 1]
    values = values[0]

    base = np.atleast_1d(explainer.expected_value)
    base_value = float(base[-1])

    order = np.argsort(-np.abs(values))[:top_k]
    contributions = [
        {
            "feature": names[i],
            "value": float(X[0, i]),
            "shap": float(values[i]),
        }
        for i in order
    ]
    return {
        "output_space": "log-odds" if kind == "ai4i" else "probability",
        "base_value": base_value,
        "contributions": contributions,
    }


def _shap_matrix(kind: str, frame: pd.DataFrame):
    explainer, pipeline = _explainer(kind)
    X, names = _transform(pipeline, frame)
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values, names


def explain_batch(kind: str, frame: pd.DataFrame, top_k: int = 3) -> list[list[list]]:
    """For each row: top_k [feature, shap] pairs ordered by |shap|."""
    values, names = _shap_matrix(kind, frame)
    out = []
    for row in values:
        idx = np.argsort(-np.abs(row))[:top_k]
        out.append([[names[i], float(row[i])] for i in idx])
    return out
