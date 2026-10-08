"""
SecurePredict V0.2 - RT-IoT2022 Feature Ablation / Shortcut Audit
Feature sets:
    1. FULL
    2. NO_CATEGORICAL    -> remove proto, service
    3. NO_PORTS          -> remove id.orig_p, id.resp_p
    4. NUMERIC_ONLY      -> numeric flow features only
Constant feature bwd_URG_flag_count is excluded in every experiment.
IMPORTANT:
    - Validation set only for comparison.
    - Final test set remains untouched.
    - Same model hyperparameters and random seed for all experiments.
"""

from __future__ import annotations

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models" / "ablation"

RANDOM_STATE = 42

TARGET = "binary_target"
DROP_ALWAYS = {
    "row_id",
    "duplicate_group_id",
    "duplicate_group_size",
    "split",
    "Attack_type",
    "binary_target",
    "bwd_URG_flag_count",
}

ABLATIONS = {
    "FULL": set(),
    "NO_CATEGORICAL": {"proto", "service"},
    "NO_PORTS": {"id.orig_p", "id.resp_p"},
    "NUMERIC_ONLY": {"proto", "service"},
}


def ensure_dirs() -> None:
    for path in (REPORTS, TABLES, FIGURES, MODELS):
        path.mkdir(parents=True, exist_ok=True)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = [
        c for c in ["proto", "service"]
        if c in X.columns
    ]

    numeric = [
        c for c in X.columns
        if c not in categorical
    ]

    transformers = [
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            numeric,
        )
    ]

    if categorical:
        transformers.append(
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
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict, np.ndarray]:
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

    metrics = {
        "accuracy": accuracy_score(
            y_val, predictions
        ),
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
    }

    return metrics, cm


def main() -> None:
    ensure_dirs()

    df = pd.read_csv(
        PROCESSED / "rt_iot2022_modeling.csv"
    )

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()

    candidate_features = [
        c for c in train.columns
        if c not in DROP_ALWAYS
    ]

    y_train = (
        train[TARGET]
        .map({"Normal": 0, "Attack": 1})
        .astype(int)
    )

    y_val = (
        val[TARGET]
        .map({"Normal": 0, "Attack": 1})
        .astype(int)
    )

    report_lines = [
        "SECUREPREDICT V0.2 - RT-IoT2022 FEATURE ABLATION",
        "=" * 90,
        "Controlled feature-family ablation using the validation split.",
        "Test set remains untouched.",
        "",
        f"Training rows: {len(train):,}",
        f"Validation rows: {len(val):,}",
        "",
    ]

    all_results = []

    for name, additional_drop in ABLATIONS.items():
        excluded = set(DROP_ALWAYS) | set(additional_drop)

        features = [
            c for c in candidate_features
            if c not in additional_drop
        ]

        X_train = train[features].copy()
        X_val = val[features].copy()

        classifier = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        preprocessor = make_preprocessor(X_train)

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", classifier),
        ])

        metrics, cm = evaluate(
            pipeline,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        result = {
            "feature_set": name,
            "n_features_before_encoding": len(features),
            **metrics,
        }
        all_results.append(result)

        joblib.dump(
            pipeline,
            MODELS / f"rt_iot2022_rf_{name.lower()}.joblib",
        )

        np.savetxt(
            FIGURES / (
                f"rt_iot2022_{name.lower()}_confusion_matrix.csv"
            ),
            cm,
            delimiter=",",
            fmt="%d",
        )

        report_lines.extend([
            name,
            "-" * 50,
            f"Features before encoding: {len(features)}",
            f"Accuracy: {metrics['accuracy']:.6f}",
            f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
            f"Precision: {metrics['precision']:.6f}",
            f"Recall: {metrics['recall']:.6f}",
            f"F1: {metrics['f1']:.6f}",
            f"ROC-AUC: {metrics['roc_auc']:.6f}",
            f"PR-AUC: {metrics['pr_auc']:.6f}",
            (
                "Confusion matrix: "
                f"TN={metrics['tn']} FP={metrics['fp']} "
                f"FN={metrics['fn']} TP={metrics['tp']}"
            ),
            f"Fit time (sec): {metrics['fit_seconds']:.3f}",
            "",
        ])

        print(
            f"{name}: "
            f"F1={metrics['f1']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}"
        )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        TABLES / "08_rt_iot_feature_ablation_results.csv",
        index=False,
    )

    # Simple impact table relative to FULL.
    full = results_df.loc[
        results_df["feature_set"] == "FULL"
    ].iloc[0]

    impact_rows = []

    for _, row in results_df.iterrows():
        impact_rows.append({
            "feature_set": row["feature_set"],
            "delta_f1_vs_full": row["f1"] - full["f1"],
            "delta_recall_vs_full": (
                row["recall"] - full["recall"]
            ),
            "delta_roc_auc_vs_full": (
                row["roc_auc"] - full["roc_auc"]
            ),
            "delta_pr_auc_vs_full": (
                row["pr_auc"] - full["pr_auc"]
            ),
        })

    impact_df = pd.DataFrame(impact_rows)

    impact_df.to_csv(
        TABLES / "08_rt_iot_feature_ablation_impact.csv",
        index=False,
    )

    report_lines.extend([
        "=" * 90,
        "INTERPRETATION GUIDE",
        "=" * 90,
        "FULL is the reference.",
        "NO_CATEGORICAL tests whether proto/service are driving the performance.",
        "NO_PORTS tests whether endpoint port information is driving the performance.",
        "NUMERIC_ONLY tests a purely numeric flow-feature representation.",
        "",
        "No test-set result is used in this experiment.",
    ])

    report_path = REPORTS / "08_rt_iot_feature_ablation.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("")
    print("=" * 90)
    print("FEATURE ABLATION COMPLETE")
    print("=" * 90)
    print(f"Report: {report_path}")
    print(
        "Results:",
        TABLES / "08_rt_iot_feature_ablation_results.csv",
    )
    print(
        "Impact:",
        TABLES / "08_rt_iot_feature_ablation_impact.csv",
    )


if __name__ == "__main__":
    main()
