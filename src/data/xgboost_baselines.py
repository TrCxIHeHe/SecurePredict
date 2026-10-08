"""
XGBoost Baseline Experiments
Purpose:
    Compare gradient boosting against the established Logistic Regression
    and Random Forest baselines without hyperparameter tuning.
Datasets:
    AI4I 2020 -> Machine failure
    RT-IoT2022 -> Normal vs Attack
Evaluation:
    Train on TRAIN split.
    Evaluate on VALIDATION split.
    Final TEST split remains untouched.
"""
from __future__ import annotations
from pathlib import Path
import time
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"
MODELS = ROOT / "models" / "xgboost_baseline"

RANDOM_STATE = 42


def ensure_dirs() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dataset_name: str,
    model_name: str,
) -> dict:
    start = time.perf_counter()

    pipeline.fit(X_train, y_train)

    fit_seconds = time.perf_counter() - start

    predictions = pipeline.predict(X_val)
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    )

    return {
        "dataset": dataset_name,
        "model": model_name,
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "precision": precision_score(
            y_val,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_val,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_val,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_val,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_val,
            probabilities,
        ),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "fit_seconds": fit_seconds,
        "n_train": len(y_train),
        "n_validation": len(y_val),
    }


def make_preprocessor(
    X: pd.DataFrame,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    categorical = [
        c for c in ["Type", "proto", "service"]
        if c in X.columns
    ]

    numeric = [
        c for c in X.columns
        if c not in categorical
    ]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    # The flag is kept for transparent reuse/experimentation.
    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(numeric_steps),
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
                categorical,
            ),
        ],
        remainder="drop",
    )


def run_ai4i(report_lines: list[str]) -> dict:
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

    X_train = train[feature_columns]
    X_val = val[feature_columns]

    y_train = train["Machine failure"].astype(int)
    y_val = val["Machine failure"].astype(int)

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.10,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
    )

    preprocessor = make_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])

    metrics = evaluate_model(
        pipeline,
        X_train,
        y_train,
        X_val,
        y_val,
        "AI4I 2020",
        "XGBoost_Baseline",
    )

    joblib.dump(
        pipeline,
        MODELS / "ai4i_xgboost_baseline.joblib",
    )

    report_lines.extend([
        "",
        "=" * 90,
        "AI4I 2020 - XGBOOST BASELINE",
        "=" * 90,
        f"Training rows: {len(train):,}",
        f"Validation rows: {len(val):,}",
        f"Positive failures: {positive:,}",
        f"Negative normal rows: {negative:,}",
        f"scale_pos_weight: {scale_pos_weight:.6f}",
        "Hyperparameter tuning: NO",
        "",
    ])

    append_metrics(report_lines, metrics)

    return metrics


def run_rt_iot(report_lines: list[str]) -> dict:
    df = pd.read_csv(
        PROCESSED / "rt_iot2022_modeling.csv"
    )

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()

    excluded = {
        "row_id",
        "duplicate_group_id",
        "duplicate_group_size",
        "split",
        "Attack_type",
        "binary_target",
        "bwd_URG_flag_count",
    }

    feature_columns = [
        c for c in train.columns
        if c not in excluded
    ]

    X_train = train[feature_columns].copy()
    X_val = val[feature_columns].copy()

    y_train = (
        train["binary_target"]
        .map({"Normal": 0, "Attack": 1})
        .astype(int)
    )

    y_val = (
        val["binary_target"]
        .map({"Normal": 0, "Attack": 1})
        .astype(int)
    )

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.10,
        subsample=0.80,
        colsample_bytree=0.80,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=1.0,
    )

    preprocessor = make_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier),
    ])

    metrics = evaluate_model(
        pipeline,
        X_train,
        y_train,
        X_val,
        y_val,
        "RT-IoT2022",
        "XGBoost_Baseline",
    )

    joblib.dump(
        pipeline,
        MODELS / "rt_iot2022_xgboost_baseline.joblib",
    )

    report_lines.extend([
        "",
        "=" * 90,
        "RT-IoT2022 - XGBOOST BINARY INTRUSION BASELINE",
        "=" * 90,
        f"Training rows: {len(train):,}",
        f"Validation rows: {len(val):,}",
        "Target: Normal=0, Attack=1",
        "Dropped constant feature: bwd_URG_flag_count",
        "scale_pos_weight: 1.0",
        "Hyperparameter tuning: NO",
        "",
    ])

    append_metrics(report_lines, metrics)

    return metrics


def append_metrics(
    report_lines: list[str],
    metrics: dict,
) -> None:
    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "fit_seconds",
    ]:
        report_lines.append(
            f"{key}: {metrics[key]:.6f}"
        )

    report_lines.append(
        "confusion_matrix: "
        f"TN={metrics['tn']} "
        f"FP={metrics['fp']} "
        f"FN={metrics['fn']} "
        f"TP={metrics['tp']}"
    )


def main() -> None:
    ensure_dirs()

    print("SECUREPREDICT V0.3 - XGBOOST BASELINES")
    print("=" * 90)

    report_lines = [
        "SECUREPREDICT V0.3 - XGBOOST BASELINE EXPERIMENTS",
        "=" * 90,
        "Validation-only model comparison.",
        "Final test sets remain untouched.",
        "No hyperparameter tuning in this experiment.",
        "",
    ]

    print("Running AI4I XGBoost...")
    ai4i_metrics = run_ai4i(report_lines)

    print(
        "AI4I XGBoost: "
        f"F1={ai4i_metrics['f1']:.4f}, "
        f"Recall={ai4i_metrics['recall']:.4f}, "
        f"ROC-AUC={ai4i_metrics['roc_auc']:.4f}, "
        f"PR-AUC={ai4i_metrics['pr_auc']:.4f}"
    )

    print("")
    print("Running RT-IoT2022 XGBoost...")
    rt_metrics = run_rt_iot(report_lines)

    print(
        "RT-IoT2022 XGBoost: "
        f"F1={rt_metrics['f1']:.4f}, "
        f"Recall={rt_metrics['recall']:.4f}, "
        f"ROC-AUC={rt_metrics['roc_auc']:.4f}, "
        f"PR-AUC={rt_metrics['pr_auc']:.4f}"
    )

    results = pd.DataFrame(
        [ai4i_metrics, rt_metrics]
    )

    results.to_csv(
        TABLES / "09_xgboost_baseline_results.csv",
        index=False,
    )

    report_lines.extend([
        "",
        "=" * 90,
        "XGBOOST BASELINE EXPERIMENT COMPLETE",
        "=" * 90,
        f"Metrics: {TABLES / '09_xgboost_baseline_results.csv'}",
        f"Models: {MODELS}",
    ])

    report_path = REPORTS / "09_xgboost_baselines.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("=" * 90)
    print("XGBOOST BASELINE COMPLETE")
    print("=" * 90)
    print(f"Report:  {report_path}")
    print(
        f"Metrics: {TABLES / '09_xgboost_baseline_results.csv'}"
    )

if __name__ == "__main__":
    main()
