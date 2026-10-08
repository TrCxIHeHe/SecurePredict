"""
Evaluate Tuned AI4I XGBoost + Threshold Study
Script:
1. Loads the tuned XGBoost selected by V0.4.
2. Evaluates it on the reserved AI4I validation split.
3. Compares it with the untuned XGBoost baseline.
4. Sweeps probability thresholds to quantify the precision/recall trade-off.
5. DOES NOT touch the final test set.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"

THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.01),
    2,
)


def evaluate_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "predicted_positive_rate": float(
            predictions.mean()
        ),
    }


def evaluate_model(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    label: str,
) -> tuple[dict, np.ndarray]:
    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    default = evaluate_threshold(
        y_val,
        probabilities,
        0.50,
    )

    default["model"] = label
    default["roc_auc"] = roc_auc_score(
        y_val,
        probabilities,
    )
    default["pr_auc"] = average_precision_score(
        y_val,
        probabilities,
    )

    return default, probabilities


def main() -> None:
    print(
        "SECUREPREDICT V0.4.1 - "
        "TUNED AI4I VALIDATION + THRESHOLD STUDY"
    )
    print("=" * 90)

    df = pd.read_csv(
        PROCESSED / "ai4i_2020_modeling.csv"
    )

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()

    feature_columns = [
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    X_val = val[feature_columns]
    y_val = val["Machine failure"].astype(int).to_numpy()

    baseline_path = (
        MODELS
        / "xgboost_baseline"
        / "ai4i_xgboost_baseline.joblib"
    )

    tuned_path = (
        MODELS
        / "tuned"
        / "ai4i_xgboost_tuned.joblib"
    )

    baseline = joblib.load(baseline_path)
    tuned = joblib.load(tuned_path)

    baseline_metrics, baseline_probs = evaluate_model(
        baseline,
        X_val,
        y_val,
        "XGBoost_Baseline",
    )

    tuned_metrics, tuned_probs = evaluate_model(
        tuned,
        X_val,
        y_val,
        "XGBoost_Tuned",
    )

    comparison = pd.DataFrame(
        [baseline_metrics, tuned_metrics]
    )

    comparison.to_csv(
        TABLES / "11_ai4i_xgboost_validation_comparison.csv",
        index=False,
    )

    print("")
    print("Validation comparison at threshold 0.50")
    print("-" * 90)

    for _, row in comparison.iterrows():
        print(
            f"{row['model']}: "
            f"F1={row['f1']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"ROC-AUC={row['roc_auc']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}"
        )

    # Threshold sweep on tuned model only.
    rows = []

    for threshold in THRESHOLDS:
        metrics = evaluate_threshold(
            y_val,
            tuned_probs,
            float(threshold),
        )

        rows.append(metrics)

    threshold_df = pd.DataFrame(rows)

    threshold_df.to_csv(
        TABLES / "11_ai4i_tuned_threshold_sweep.csv",
        index=False,
    )

    best_f1 = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    # Best recall among thresholds achieving at least 0.30 precision.
    precision_constrained = threshold_df[
        threshold_df["precision"] >= 0.30
    ]

    if not precision_constrained.empty:
        best_recall_at_precision = (
            precision_constrained
            .sort_values(
                ["recall", "f1"],
                ascending=False,
            )
            .iloc[0]
        )
    else:
        best_recall_at_precision = None

    # Highest recall threshold meeting F1 >= 0.50.
    f1_constrained = threshold_df[
        threshold_df["f1"] >= 0.50
    ]

    if not f1_constrained.empty:
        best_recall_at_f1 = (
            f1_constrained
            .sort_values(
                ["recall", "f1"],
                ascending=False,
            )
            .iloc[0]
        )
    else:
        best_recall_at_f1 = None

    report_lines = [
        "SECUREPREDICT V0.4.1 - AI4I VALIDATION + THRESHOLD STUDY",
        "=" * 90,
        "Validation split only.",
        "Final test split remains untouched.",
        "",
        "DEFAULT THRESHOLD = 0.50",
        "-" * 50,
    ]

    for _, row in comparison.iterrows():
        report_lines.extend([
            f"{row['model']}:",
            f"  Precision: {row['precision']:.6f}",
            f"  Recall: {row['recall']:.6f}",
            f"  F1: {row['f1']:.6f}",
            f"  ROC-AUC: {row['roc_auc']:.6f}",
            f"  PR-AUC: {row['pr_auc']:.6f}",
            "",
        ])

    report_lines.extend([
        "=" * 90,
        "TUNED MODEL THRESHOLD STUDY",
        "=" * 90,
        "",
        "Best F1 threshold:",
        f"  threshold = {best_f1['threshold']:.2f}",
        f"  precision = {best_f1['precision']:.6f}",
        f"  recall = {best_f1['recall']:.6f}",
        f"  F1 = {best_f1['f1']:.6f}",
        "",
    ])

    if best_recall_at_precision is not None:
        report_lines.extend([
            "Highest recall with precision >= 0.30:",
            f"  threshold = {best_recall_at_precision['threshold']:.2f}",
            f"  precision = {best_recall_at_precision['precision']:.6f}",
            f"  recall = {best_recall_at_precision['recall']:.6f}",
            f"  F1 = {best_recall_at_precision['f1']:.6f}",
            "",
        ])

    if best_recall_at_f1 is not None:
        report_lines.extend([
            "Highest recall with F1 >= 0.50:",
            f"  threshold = {best_recall_at_f1['threshold']:.2f}",
            f"  precision = {best_recall_at_f1['precision']:.6f}",
            f"  recall = {best_recall_at_f1['recall']:.6f}",
            f"  F1 = {best_recall_at_f1['f1']:.6f}",
            "",
        ])

    report_lines.extend([
        "=" * 90,
        "OUTPUTS",
        "=" * 90,
        f"Comparison: {TABLES / '11_ai4i_xgboost_validation_comparison.csv'}",
        f"Threshold sweep: {TABLES / '11_ai4i_tuned_threshold_sweep.csv'}",
        "",
        "NEXT STEP:",
        "Review the threshold trade-off, freeze the operating threshold,",
        "then perform one final untouched test-set evaluation.",
    ])

    report_path = (
        REPORTS / "11_ai4i_threshold_study.txt"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print(
        f"Best F1 threshold: "
        f"{best_f1['threshold']:.2f} "
        f"(F1={best_f1['f1']:.4f}, "
        f"Recall={best_f1['recall']:.4f}, "
        f"Precision={best_f1['precision']:.4f})"
    )

    if best_recall_at_precision is not None:
        print(
            f"Best recall @ precision>=0.30: "
            f"{best_recall_at_precision['threshold']:.2f} "
            f"(Recall={best_recall_at_precision['recall']:.4f})"
        )

    print("")
    print("=" * 90)
    print("THRESHOLD STUDY COMPLETE")
    print("=" * 90)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
