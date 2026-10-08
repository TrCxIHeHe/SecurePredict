"""
Final Untouched AI4I Test Evaluation
Model-selection state before this script:
    - Hyperparameters selected using TRAIN-only cross-validation.
    - Operating threshold selected using the reserved VALIDATION split.
    - Frozen primary operating threshold: 0.77 (best validation F1).
    - Early-warning threshold from validation: 0.07 (secondary analysis only).
This script:
    1. Combines TRAIN + VALIDATION for final model fitting using the already
       selected hyperparameters.
    2. Reconstructs the tuned XGBoost pipeline with the frozen parameters.
    3. Evaluates exactly once on the untouched TEST set.
    4. Reports performance at the frozen primary threshold (0.77).
    5. Also reports probability-ranking metrics (ROC-AUC and PR-AUC).
    6. Does NOT tune anything using TEST.
No test-driven model selection is performed.
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"
MODELS = ROOT / "models" / "final"

RANDOM_STATE = 42
PRIMARY_THRESHOLD = 0.77
EARLY_WARNING_THRESHOLD = 0.07


FEATURE_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def make_preprocessor() -> ColumnTransformer:
    numeric = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        ),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=True,
                        ),
                    ),
                ]),
                ["Type"],
            ),
        ],
        remainder="drop",
    )


def build_final_model(
    scale_pos_weight: float,
) -> Pipeline:
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        min_child_weight=4,
        subsample=1.0,
        colsample_bytree=0.9,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
    )

    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", classifier),
    ])


def evaluate_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

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
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        PROCESSED / "ai4i_2020_modeling.csv"
    )

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    # Final fit uses only data that was permitted for model fitting after
    # hyperparameter and threshold decisions had already been made.
    fit_df = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    X_fit = fit_df[FEATURE_COLUMNS]
    y_fit = fit_df["Machine failure"].astype(int)

    X_test = test[FEATURE_COLUMNS]
    y_test = test["Machine failure"].astype(int).to_numpy()

    negative = int((y_fit == 0).sum())
    positive = int((y_fit == 1).sum())
    scale_pos_weight = negative / positive

    model = build_final_model(
        scale_pos_weight=scale_pos_weight,
    )

    print(
        "SECUREPREDICT V0.5 - FINAL AI4I TEST EVALUATION"
    )
    print("=" * 90)
    print(
        "Model selection and threshold selection are complete."
    )
    print(
        f"Frozen primary threshold: {PRIMARY_THRESHOLD:.2f}"
    )
    print(
        f"Secondary early-warning threshold: "
        f"{EARLY_WARNING_THRESHOLD:.2f}"
    )
    print("")
    print(
        f"Final fitting rows (train + validation): "
        f"{len(fit_df):,}"
    )
    print(
        f"Untouched test rows: {len(test):,}"
    )

    print("")
    print("Fitting final model on TRAIN + VALIDATION...")
    model.fit(X_fit, y_fit)

    model_path = (
        MODELS / "ai4i_xgboost_final.joblib"
    )
    joblib.dump(model, model_path)

    print("Final model fitted.")
    print(
        "Evaluating ONCE on the untouched TEST set..."
    )

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    ranking_metrics = {
        "roc_auc": roc_auc_score(
            y_test,
            test_probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            test_probabilities,
        ),
    }

    primary = evaluate_at_threshold(
        y_test,
        test_probabilities,
        PRIMARY_THRESHOLD,
    )

    early_warning = evaluate_at_threshold(
        y_test,
        test_probabilities,
        EARLY_WARNING_THRESHOLD,
    )

    results = pd.DataFrame([
        {
            "operating_mode": "balanced_primary",
            **primary,
            **ranking_metrics,
        },
        {
            "operating_mode": "early_warning_secondary",
            **early_warning,
            **ranking_metrics,
        },
    ])

    results_path = (
        TABLES / "12_ai4i_final_test_results.csv"
    )
    results.to_csv(
        results_path,
        index=False,
    )

    probability_path = (
        TABLES / "12_ai4i_test_predictions.csv"
    )

    test_predictions = pd.DataFrame({
        "row_id": test["row_id"].to_numpy(),
        "actual_failure": y_test,
        "predicted_failure_probability": test_probabilities,
        "predicted_failure_primary": (
            test_probabilities >= PRIMARY_THRESHOLD
        ).astype(int),
        "predicted_failure_early_warning": (
            test_probabilities >= EARLY_WARNING_THRESHOLD
        ).astype(int),
    })

    test_predictions.to_csv(
        probability_path,
        index=False,
    )

    metadata = {
        "model": "XGBoost",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.08,
        "min_child_weight": 4,
        "subsample": 1.0,
        "colsample_bytree": 0.9,
        "gamma": 0.5,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
        "primary_threshold": PRIMARY_THRESHOLD,
        "early_warning_threshold": EARLY_WARNING_THRESHOLD,
        "fit_rows": len(fit_df),
        "test_rows": len(test),
        "test_failure_count": int(y_test.sum()),
        "model_path": str(model_path),
        "selection_note": (
            "Hyperparameters selected using TRAIN-only CV; "
            "primary threshold selected on VALIDATION."
        ),
    }

    metadata_path = (
        TABLES / "12_ai4i_final_test_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = [
        "SECUREPREDICT V0.5 - FINAL AI4I TEST EVALUATION",
        "=" * 90,
        "FINAL MODEL: Tuned XGBoost",
        "FIT DATA: TRAIN + VALIDATION",
        "EVALUATION DATA: UNTOUCHED TEST",
        "",
        f"Final fitting rows: {len(fit_df):,}",
        f"Test rows: {len(test):,}",
        f"Test failures: {int(y_test.sum()):,}",
        f"scale_pos_weight: {scale_pos_weight:.6f}",
        "",
        "FROZEN PRIMARY OPERATING MODE",
        "-" * 50,
        f"Threshold: {PRIMARY_THRESHOLD:.2f}",
        f"Accuracy: {primary['accuracy']:.6f}",
        f"Balanced Accuracy: {primary['balanced_accuracy']:.6f}",
        f"Precision: {primary['precision']:.6f}",
        f"Recall: {primary['recall']:.6f}",
        f"F1: {primary['f1']:.6f}",
        f"ROC-AUC: {ranking_metrics['roc_auc']:.6f}",
        f"PR-AUC: {ranking_metrics['pr_auc']:.6f}",
        (
            "Confusion matrix: "
            f"TN={primary['tn']} "
            f"FP={primary['fp']} "
            f"FN={primary['fn']} "
            f"TP={primary['tp']}"
        ),
        "",
        "SECONDARY EARLY-WARNING MODE",
        "-" * 50,
        f"Threshold: {EARLY_WARNING_THRESHOLD:.2f}",
        f"Accuracy: {early_warning['accuracy']:.6f}",
        f"Balanced Accuracy: {early_warning['balanced_accuracy']:.6f}",
        f"Precision: {early_warning['precision']:.6f}",
        f"Recall: {early_warning['recall']:.6f}",
        f"F1: {early_warning['f1']:.6f}",
        (
            "Confusion matrix: "
            f"TN={early_warning['tn']} "
            f"FP={early_warning['fp']} "
            f"FN={early_warning['fn']} "
            f"TP={early_warning['tp']}"
        ),
        "",
        "OUTPUTS",
        "-" * 50,
        f"Final model: {model_path}",
        f"Results: {results_path}",
        f"Predictions: {probability_path}",
        f"Metadata: {metadata_path}",
        "",
        "IMPORTANT:",
        "This test evaluation was performed after model and threshold",
        "selection and is the first use of the untouched test split.",
    ]

    report_path = (
        REPORTS / "12_ai4i_final_test.txt"
    )

    report_path.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("")
    print("=" * 90)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 90)
    print(
        f"Primary threshold {PRIMARY_THRESHOLD:.2f}: "
        f"F1={primary['f1']:.4f}, "
        f"Recall={primary['recall']:.4f}, "
        f"Precision={primary['precision']:.4f}, "
        f"ROC-AUC={ranking_metrics['roc_auc']:.4f}, "
        f"PR-AUC={ranking_metrics['pr_auc']:.4f}"
    )
    print(
        f"Early-warning threshold {EARLY_WARNING_THRESHOLD:.2f}: "
        f"F1={early_warning['f1']:.4f}, "
        f"Recall={early_warning['recall']:.4f}, "
        f"Precision={early_warning['precision']:.4f}"
    )
    print("")
    print(f"Report: {report_path}")
    print(f"Results: {results_path}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
