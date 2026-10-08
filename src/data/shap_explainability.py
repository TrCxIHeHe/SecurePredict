"""
SecurePredict V0.6 - SHAP Explainability

Explains the FROZEN final models using validation data only.

Models:
    AI4I:
        models/final/ai4i_xgboost_final.joblib

    RT-IoT2022:
        models/final/rt_iot2022_random_forest_final.joblib

"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models" / "final"

RANDOM_STATE = 42
MAX_EXPLAIN_ROWS_AI4I = 800
MAX_EXPLAIN_ROWS_RT = 800


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def save_bar_plot(
    names: list[str],
    values: np.ndarray,
    title: str,
    filename: str,
    top_n: int = 15,
) -> None:
    values = np.asarray(values)

    order = np.argsort(values)[::-1][:top_n]
    selected_names = [str(names[i]) for i in order][::-1]
    selected_values = values[order][::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(
        selected_names,
        selected_values,
    )
    plt.title(title)
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.savefig(
        FIGURES / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()


def make_ai4i_data():
    df = pd.read_csv(
        PROCESSED / "ai4i_2020_modeling.csv"
    )

    validation = df[
        df["split"] == "validation"
    ].copy()

    features = [
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    X = validation[features].copy()

    return validation, X


def make_rt_data():
    df = pd.read_csv(
        PROCESSED / "rt_iot2022_modeling.csv"
    )

    validation = df[
        df["split"] == "validation"
    ].copy()

    excluded = {
        "row_id",
        "duplicate_group_id",
        "duplicate_group_size",
        "split",
        "Attack_type",
        "binary_target",
        "bwd_URG_flag_count",
        "proto",
        "service",
        "id.orig_p",
        "id.resp_p",
    }

    features = [
        c for c in validation.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(
            validation[c]
        )
    ]

    X = validation[features].copy()

    return validation, X


def explain_ai4i(report_lines: list[str]) -> None:
    report_lines.extend([
        "",
        "=" * 90,
        "AI4I 2020 - SHAP EXPLAINABILITY",
        "=" * 90,
    ])

    validation, X = make_ai4i_data()

    if len(X) > MAX_EXPLAIN_ROWS_AI4I:
        X_sample = X.sample(
            n=MAX_EXPLAIN_ROWS_AI4I,
            random_state=RANDOM_STATE,
        )
    else:
        X_sample = X.copy()

    model_path = (
        MODELS / "ai4i_xgboost_final.joblib"
    )

    pipeline = joblib.load(model_path)

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]
    estimator = pipeline.named_steps[
        "model"
    ]

    transformed = preprocessor.transform(
        X_sample
    )

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    # Dense conversion is safe for this tiny transformed matrix.
    if hasattr(transformed, "toarray"):
        transformed_for_shap = transformed.toarray()
    else:
        transformed_for_shap = np.asarray(
            transformed
        )

    explainer = shap.TreeExplainer(
        estimator
    )

    shap_result = explainer(
        transformed_for_shap
    )

    shap_values = shap_result.values

    # Binary tree models may expose (n, features) or (n, features, classes).
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance.to_csv(
        TABLES / "14_ai4i_shap_global_importance.csv",
        index=False,
    )

    save_bar_plot(
        importance["feature"].tolist(),
        importance["mean_abs_shap"].to_numpy(),
        "AI4I Global SHAP Feature Importance",
        "ai4i_shap_global_importance.png",
        top_n=15,
    )

    # Select a high-risk validation example.
    probabilities = pipeline.predict_proba(
        validation[
            [
                "Type",
                "Air temperature",
                "Process temperature",
                "Rotational speed",
                "Torque",
                "Tool wear",
            ]
        ]
    )[:, 1]

    high_risk_position = int(
        np.argmax(probabilities)
    )

    local_row = validation.iloc[
        high_risk_position
    ]

    local_X = X.iloc[
        high_risk_position:
        high_risk_position + 1
    ]

    local_transformed = preprocessor.transform(
        local_X
    )

    if hasattr(local_transformed, "toarray"):
        local_transformed = (
            local_transformed.toarray()
        )

    local_shap = explainer(
        local_transformed
    ).values

    if local_shap.ndim == 3:
        local_shap = local_shap[:, :, 1]

    local_values = local_shap[0]

    local_importance = pd.DataFrame({
        "feature": feature_names,
        "shap_value": local_values,
        "abs_shap_value": np.abs(local_values),
    }).sort_values(
        "abs_shap_value",
        ascending=False,
    )

    local_importance.to_csv(
        TABLES / "14_ai4i_shap_local_example.csv",
        index=False,
    )

    top_local = local_importance.head(8)

    plt.figure(figsize=(9, 6))
    colors = [
        "positive" if value >= 0 else "negative"
        for value in top_local["shap_value"]
    ]
    # Matplotlib default colors are intentionally used;
    # no explicit color styling is necessary for the report.
    plt.barh(
        top_local["feature"].iloc[::-1],
        top_local["shap_value"].iloc[::-1],
    )
    plt.axvline(
        0,
        linewidth=1,
    )
    plt.title(
        "AI4I Local SHAP Explanation - Highest Risk Validation Example"
    )
    plt.xlabel("SHAP contribution")
    plt.tight_layout()
    plt.savefig(
        FIGURES / "ai4i_shap_local_example.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    prediction_probability = float(
        probabilities[high_risk_position]
    )

    report_lines.extend([
        f"Validation rows explained: {len(X_sample):,}",
        f"Selected local example probability: {prediction_probability:.6f}",
        f"Selected local example actual failure: "
        f"{int(local_row['Machine failure'])}",
        "",
        "Top global features:",
    ])

    for _, row in importance.head(10).iterrows():
        report_lines.append(
            f"  {row['feature']}: "
            f"{row['mean_abs_shap']:.6f}"
        )

    report_lines.extend([
        "",
        "Top local contributors:",
    ])

    for _, row in top_local.iterrows():
        direction = (
            "toward failure"
            if row["shap_value"] > 0
            else "away from failure"
        )
        report_lines.append(
            f"  {row['feature']}: "
            f"{row['shap_value']:.6f} "
            f"({direction})"
        )


def explain_rt_iot(report_lines: list[str]) -> None:
    report_lines.extend([
        "",
        "=" * 90,
        "RT-IoT2022 - SHAP EXPLAINABILITY",
        "=" * 90,
    ])

    validation, X = make_rt_data()

    if len(X) > MAX_EXPLAIN_ROWS_RT:
        X_sample = X.sample(
            n=MAX_EXPLAIN_ROWS_RT,
            random_state=RANDOM_STATE,
        )
    else:
        X_sample = X.copy()

    model_path = (
        MODELS
        / "rt_iot2022_random_forest_final.joblib"
    )

    pipeline = joblib.load(model_path)

    imputer = pipeline.named_steps[
        "imputer"
    ]
    estimator = pipeline.named_steps[
        "model"
    ]

    transformed = imputer.transform(
        X_sample
    )

    feature_names = list(X.columns)

    explainer = shap.TreeExplainer(
        estimator
    )

    shap_result = explainer(
        transformed
    )

    shap_values = shap_result.values

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(
        shap_values
    ).mean(axis=0)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
    }).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    importance.to_csv(
        TABLES / "15_rt_iot2022_shap_global_importance.csv",
        index=False,
    )

    save_bar_plot(
        importance["feature"].tolist(),
        importance["mean_abs_shap"].to_numpy(),
        "RT-IoT2022 Global SHAP Feature Importance",
        "rt_iot2022_shap_global_importance.png",
        top_n=15,
    )

    probabilities = pipeline.predict_proba(
        validation[
            feature_names
        ]
    )[:, 1]

    high_risk_position = int(
        np.argmax(probabilities)
    )

    local_X = X.iloc[
        high_risk_position:
        high_risk_position + 1
    ]

    local_transformed = imputer.transform(
        local_X
    )

    local_shap = explainer(
        local_transformed
    ).values

    if local_shap.ndim == 3:
        local_shap = local_shap[:, :, 1]

    local_values = local_shap[0]

    local_importance = pd.DataFrame({
        "feature": feature_names,
        "shap_value": local_values,
        "abs_shap_value": np.abs(local_values),
    }).sort_values(
        "abs_shap_value",
        ascending=False,
    )

    local_importance.to_csv(
        TABLES / "15_rt_iot2022_shap_local_example.csv",
        index=False,
    )

    top_local = local_importance.head(10)

    plt.figure(figsize=(9, 6))
    plt.barh(
        top_local["feature"].iloc[::-1],
        top_local["shap_value"].iloc[::-1],
    )
    plt.axvline(
        0,
        linewidth=1,
    )
    plt.title(
        "RT-IoT2022 Local SHAP Explanation - Highest Risk Validation Example"
    )
    plt.xlabel("SHAP contribution")
    plt.tight_layout()
    plt.savefig(
        FIGURES / "rt_iot2022_shap_local_example.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    prediction_probability = float(
        probabilities[high_risk_position]
    )

    report_lines.extend([
        f"Validation rows explained: {len(X_sample):,}",
        f"Selected local example attack probability: "
        f"{prediction_probability:.6f}",
        f"Selected local example actual binary target: "
        f"{validation.iloc[high_risk_position]['binary_target']}",
        "",
        "Top global features:",
    ])

    for _, row in importance.head(10).iterrows():
        report_lines.append(
            f"  {row['feature']}: "
            f"{row['mean_abs_shap']:.6f}"
        )

    report_lines.extend([
        "",
        "Top local contributors:",
    ])

    for _, row in top_local.iterrows():
        direction = (
            "toward attack"
            if row["shap_value"] > 0
            else "away from attack"
        )
        report_lines.append(
            f"  {row['feature']}: "
            f"{row['shap_value']:.6f} "
            f"({direction})"
        )


def main() -> None:
    ensure_dirs()

    print("SECUREPREDICT V0.6 - SHAP EXPLAINABILITY")
    print("=" * 90)
    print("Using validation data only; test set remains untouched.")
    print("")

    report_lines = [
        "SECUREPREDICT V0.6 - SHAP EXPLAINABILITY",
        "=" * 90,
        "Global and local explanations use validation data only.",
        "The final test set is not used for explanations.",
    ]

    print("Explaining AI4I final XGBoost...")
    explain_ai4i(report_lines)
    print("AI4I explanations complete.")

    print("Explaining RT-IoT2022 final Random Forest...")
    explain_rt_iot(report_lines)
    print("RT-IoT2022 explanations complete.")

    report_lines.extend([
        "",
        "=" * 90,
        "SHAP EXPLAINABILITY COMPLETE",
        "=" * 90,
        f"Tables: {TABLES}",
        f"Figures: {FIGURES}",
    ])

    report_path = (
        REPORTS / "14_shap_explainability.txt"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("=" * 90)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 90)
    print(f"Report: {report_path}")
    print(f"Tables: {TABLES}")
    print(f"Figures: {FIGURES}")


if __name__ == "__main__":
    main()
