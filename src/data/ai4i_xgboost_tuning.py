"""
AI4I XGBoost Hyperparameter Tuning
Purpose:
    Tune the AI4I predictive-maintenance XGBoost model using the TRAINING
    split only, with stratified cross-validation.
Selection metric:
    Average Precision (PR-AUC)
"""
from __future__ import annotations

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"
MODELS = ROOT / "models" / "tuned"

RANDOM_STATE = 42


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = ["Type"]
    numeric = [c for c in X.columns if c not in categorical]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
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


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        PROCESSED / "ai4i_2020_modeling.csv"
    )

    train = df[df["split"] == "train"].copy()

    feature_columns = [
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    X_train = train[feature_columns]
    y_train = train["Machine failure"].astype(int)

    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    scale_pos_weight = negative / positive

    preprocessor = make_preprocessor(X_train)

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", base_model),
    ])

    param_distributions = {
        "model__n_estimators": [
            200, 300, 400, 500, 650
        ],
        "model__max_depth": [
            2, 3, 4, 5, 6, 8
        ],
        "model__learning_rate": [
            0.02, 0.04, 0.06, 0.08, 0.10, 0.15
        ],
        "model__min_child_weight": [
            1, 2, 4, 6, 10
        ],
        "model__subsample": [
            0.70, 0.80, 0.90, 1.00
        ],
        "model__colsample_bytree": [
            0.70, 0.80, 0.90, 1.00
        ],
        "model__gamma": [
            0.0, 0.1, 0.25, 0.5
        ],
        "model__reg_lambda": [
            1.0, 2.0, 5.0, 10.0
        ],
        "model__reg_alpha": [
            0.0, 0.01, 0.1, 0.5
        ],
    }

    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        return_train_score=True,
        refit=True,
    )

    print("SECUREPREDICT V0.4 - AI4I XGBOOST TUNING")
    print("=" * 90)
    print(f"Training rows: {len(train):,}")
    print(f"Positive failures: {positive:,}")
    print(f"Negative normal rows: {negative:,}")
    print(f"scale_pos_weight: {scale_pos_weight:.6f}")
    print("CV folds: 4")
    print("Random search iterations: 30")
    print("Primary scoring metric: Average Precision (PR-AUC)")
    print("")
    print("Starting search...")

    start = time.perf_counter()

    search.fit(X_train, y_train)

    elapsed = time.perf_counter() - start

    print("")
    print("=" * 90)
    print("TUNING COMPLETE")
    print("=" * 90)

    print(f"Elapsed time: {elapsed / 60:.2f} minutes")
    print(
        f"Best mean CV PR-AUC: "
        f"{search.best_score_:.6f}"
    )

    print("\nBest parameters:")
    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")

    # Save full CV history for research/reporting.
    cv_results = pd.DataFrame(search.cv_results_)

    cv_results = cv_results.sort_values(
        "rank_test_score"
    )

    cv_results.to_csv(
        TABLES / "10_ai4i_xgboost_tuning_results.csv",
        index=False,
    )

    # Compact top-results table.
    top_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_train_score",
        "param_model__n_estimators",
        "param_model__max_depth",
        "param_model__learning_rate",
        "param_model__min_child_weight",
        "param_model__subsample",
        "param_model__colsample_bytree",
        "param_model__gamma",
        "param_model__reg_lambda",
        "param_model__reg_alpha",
    ]

    top_results = cv_results[
        [
            c for c in top_columns
            if c in cv_results.columns
        ]
    ].head(10)

    top_results.to_csv(
        TABLES / "10_ai4i_xgboost_top10.csv",
        index=False,
    )
    model_path = (
        MODELS / "ai4i_xgboost_tuned.joblib"
    )
    joblib.dump(
        search.best_estimator_,
        model_path,
    )

    metadata = {
        "best_cv_pr_auc": float(search.best_score_),
        "elapsed_seconds": float(elapsed),
        "cv_folds": 4,
        "n_iter": 30,
        "random_state": RANDOM_STATE,
        "scale_pos_weight": scale_pos_weight,
        "best_params": {
            key: (
                int(value)
                if isinstance(value, np.integer)
                else float(value)
                if isinstance(value, np.floating)
                else value
            )
            for key, value in search.best_params_.items()
        },
        "model_path": str(model_path),
    }

    metadata_path = (
        TABLES / "10_ai4i_xgboost_tuning_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    report = [
        "SECUREPREDICT V0.4 - AI4I XGBOOST HYPERPARAMETER TUNING",
        "=" * 90,
        "Training split only.",
        "Validation and test sets were not used during tuning.",
        "",
        f"Training rows: {len(train):,}",
        f"Positive failures: {positive:,}",
        f"Negative normal rows: {negative:,}",
        f"scale_pos_weight: {scale_pos_weight:.6f}",
        "CV folds: 4",
        "Randomized search iterations: 30",
        "Selection metric: Average Precision (PR-AUC)",
        f"Elapsed time (minutes): {elapsed / 60:.2f}",
        f"Best mean CV PR-AUC: {search.best_score_:.6f}",
        "",
        "Best parameters:",
    ]

    report.extend(
        [
            f"  {key}: {value}"
            for key, value in search.best_params_.items()
        ]
    )

    report.extend([
        "",
        f"Saved model: {model_path}",
        f"Full CV results: {TABLES / '10_ai4i_xgboost_tuning_results.csv'}",
        f"Top-10 results: {TABLES / '10_ai4i_xgboost_top10.csv'}",
        f"Metadata: {metadata_path}",
        "",
        "NEXT STEP:",
        "Evaluate the tuned estimator on the reserved validation split,",
        "then optimize the operating probability threshold before final",
        "untouched test-set evaluation.",
    ])

    report_path = (
        REPORTS / "10_ai4i_xgboost_tuning.txt"
    )

    report_path.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(f"\nReport: {report_path}")
    print(
        f"Full CV results: "
        f"{TABLES / '10_ai4i_xgboost_tuning_results.csv'}"
    )


if __name__ == "__main__":
    main()
