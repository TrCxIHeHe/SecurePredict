"""
Final RT-IoT2022 Test Evaluation
Final security-model decision:
    Random Forest, numeric-flow-only feature set.
Why numeric-only?
    The V0.2 controlled ablation showed essentially unchanged validation
    performance after removing proto/service and port features:
        FULL          F1 ~= 0.9994
        NUMERIC_ONLY  F1 ~= 0.9993

For parsimony and a cleaner inference contract, the final candidate uses
numeric network-flow features only.

Model:
    RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=42,
    )

    - Train + validation are used for the final fit.
    - Test is untouched until this script.
    - No test-driven tuning is performed.
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"
MODELS = ROOT / "models" / "final"

RANDOM_STATE = 42

METADATA_COLUMNS = {
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

PRIMARY_THRESHOLD = 0.50


def select_numeric_features(df: pd.DataFrame) -> list[str]:
    candidates = [
        c for c in df.columns
        if c not in METADATA_COLUMNS
    ]

    return [
        c
        for c in candidates
        if pd.api.types.is_numeric_dtype(df[c])
    ]


def build_model() -> Pipeline:
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def evaluate(
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
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
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
        PROCESSED / "rt_iot2022_modeling.csv"
    )

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    fit_df = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    numeric_features = select_numeric_features(fit_df)

    X_fit = fit_df[numeric_features]
    X_test = test[numeric_features]

    y_fit = (
        fit_df["binary_target"]
        .map({"Normal": 0, "Attack": 1})
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test["binary_target"]
        .map({"Normal": 0, "Attack": 1})
        .astype(int)
        .to_numpy()
    )

    model = build_model()

    print(
        "SECUREPREDICT V0.5 - FINAL RT-IoT2022 TEST EVALUATION"
    )
    print("=" * 90)
    print(
        "Final candidate: Random Forest, numeric-flow-only"
    )
    print(
        f"Numeric features: {len(numeric_features)}"
    )
    print(
        f"Final fitting rows (train + validation): "
        f"{len(fit_df):,}"
    )
    print(
        f"Untouched test rows: {len(test):,}"
    )
    print(
        f"Test attack count: {int(y_test.sum()):,}"
    )
    print("")

    print(
        "Fitting final security model on TRAIN + VALIDATION..."
    )
    model.fit(X_fit, y_fit)

    model_path = (
        MODELS / "rt_iot2022_random_forest_final.joblib"
    )
    joblib.dump(model, model_path)

    print(
        "Final model fitted."
    )
    print(
        "Evaluating ONCE on the untouched TEST set..."
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = evaluate(
        y_test,
        probabilities,
        PRIMARY_THRESHOLD,
    )

    results = pd.DataFrame([
        {
            "operating_mode": "binary_attack_detection",
            **metrics,
            "n_numeric_features": len(numeric_features),
        }
    ])

    results_path = (
        TABLES / "13_rt_iot2022_final_test_results.csv"
    )
    results.to_csv(
        results_path,
        index=False,
    )

    predictions = pd.DataFrame({
        "row_id": test["row_id"].to_numpy(),
        "actual_binary_target": y_test,
        "predicted_attack_probability": probabilities,
        "predicted_attack": (
            probabilities >= PRIMARY_THRESHOLD
        ).astype(int),
    })

    predictions_path = (
        TABLES / "13_rt_iot2022_test_predictions.csv"
    )
    predictions.to_csv(
        predictions_path,
        index=False,
    )

    metadata = {
        "model": "RandomForestClassifier",
        "n_estimators": 200,
        "class_weight": "balanced_subsample",
        "min_samples_leaf": 2,
        "random_state": RANDOM_STATE,
        "feature_policy": "numeric_flow_only",
        "removed_constant_feature": "bwd_URG_flag_count",
        "removed_categorical_features": [
            "proto",
            "service",
        ],
        "removed_port_features": [
            "id.orig_p",
            "id.resp_p",
        ],
        "n_numeric_features": len(numeric_features),
        "fit_rows": len(fit_df),
        "test_rows": len(test),
        "threshold": PRIMARY_THRESHOLD,
        "model_path": str(model_path),
        "selection_note": (
            "Model family and feature set selected from TRAIN/VALIDATION "
            "experiments; test used only for final evaluation."
        ),
    }

    metadata_path = (
        TABLES / "13_rt_iot2022_final_test_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = [
        "SECUREPREDICT V0.5 - FINAL RT-IoT2022 TEST EVALUATION",
        "=" * 90,
        "FINAL MODEL: Random Forest (numeric-flow-only)",
        "FIT DATA: TRAIN + VALIDATION",
        "EVALUATION DATA: UNTOUCHED TEST",
        "",
        f"Numeric features: {len(numeric_features)}",
        f"Final fitting rows: {len(fit_df):,}",
        f"Test rows: {len(test):,}",
        f"Test attack rows: {int(y_test.sum()):,}",
        "",
        "FEATURE POLICY",
        "-" * 50,
        "Removed constant feature: bwd_URG_flag_count",
        "Removed categorical features: proto, service",
        "Removed port features: id.orig_p, id.resp_p",
        "",
        "FINAL TEST PERFORMANCE",
        "-" * 50,
        f"Threshold: {PRIMARY_THRESHOLD:.2f}",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Balanced Accuracy: {metrics['balanced_accuracy']:.6f}",
        f"Precision: {metrics['precision']:.6f}",
        f"Recall: {metrics['recall']:.6f}",
        f"F1: {metrics['f1']:.6f}",
        f"ROC-AUC: {metrics['roc_auc']:.6f}",
        f"PR-AUC: {metrics['pr_auc']:.6f}",
        (
            "Confusion matrix: "
            f"TN={metrics['tn']} "
            f"FP={metrics['fp']} "
            f"FN={metrics['fn']} "
            f"TP={metrics['tp']}"
        ),
        "",
        "OUTPUTS",
        "-" * 50,
        f"Final model: {model_path}",
        f"Results: {results_path}",
        f"Predictions: {predictions_path}",
        f"Metadata: {metadata_path}",
        "",
        "IMPORTANT:",
        "This test evaluation was performed after model/feature selection.",
        "The test split was not used for tuning or model selection.",
    ]

    report_path = (
        REPORTS / "13_rt_iot2022_final_test.txt"
    )
    report_path.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("")
    print("=" * 90)
    print("FINAL RT-IoT2022 TEST EVALUATION COMPLETE")
    print("=" * 90)
    print(
        f"F1={metrics['f1']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}, "
        f"PR-AUC={metrics['pr_auc']:.4f}"
    )
    print("")
    print(f"Report: {report_path}")
    print(f"Results: {results_path}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
