from __future__ import annotations

"""SecurePredict V0.1 - baseline ML experiments.

Fits on TRAIN only and evaluates on VALIDATION only. TEST remains untouched.
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models" / "baseline"
RANDOM_STATE = 42


def dirs() -> None:
    for p in [TABLES, FIGURES, MODELS]:
        p.mkdir(parents=True, exist_ok=True)


def metrics_row(dataset, model_name, y_true, y_pred, y_score, fit_seconds):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    return {
        "dataset": dataset,
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_attack_or_failure": precision_score(y_true, y_pred, zero_division=0),
        "recall_attack_or_failure": recall_score(y_true, y_pred, zero_division=0),
        "f1_attack_or_failure": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "normal_or_nonfailure_recall": tn / (tn + fp) if (tn + fp) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fit_seconds": fit_seconds,
    }


def ai4i_preprocessor():
    categorical = ["Type"]
    numeric = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), categorical),
    ])


def rt_preprocessor():
    numeric = None  # filled from X columns below
    return numeric


def run_dataset(name, train, val, X_cols, y_col, preprocessor_factory, models, drop_cols=None):
    drop_cols = drop_cols or []
    X_train = train[X_cols].drop(columns=drop_cols, errors="ignore")
    X_val = val[X_cols].drop(columns=drop_cols, errors="ignore")
    y_train = train[y_col]
    y_val = val[y_col]

    results = []
    for model_name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor_factory(X_train)),
            ("model", estimator),
        ])
        start = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        pred = pipe.predict(X_val)
        if hasattr(pipe, "predict_proba"):
            score = pipe.predict_proba(X_val)[:, 1]
        else:
            score = pipe.decision_function(X_val)
        row = metrics_row(name, model_name, y_val, pred, score, fit_seconds)
        results.append(row)
        joblib.dump(pipe, MODELS / f"{name.lower().replace(' ', '_')}_{model_name}.joblib")
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        np.savetxt(FIGURES / f"{name.lower().replace(' ', '_')}_{model_name}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
        print(f"  {model_name}: F1={row['f1_attack_or_failure']:.4f}, Recall={row['recall_attack_or_failure']:.4f}, ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}")
    return results


def main():
    dirs()
    report = [
        "SECUREPREDICT V0.1 - BASELINE ML EXPERIMENTS",
        "=" * 90,
        "Fit: TRAIN only | Evaluation: VALIDATION only | TEST: untouched",
        "",
    ]
    all_results = []

    # AI4I
    ai = pd.read_csv(PROCESSED / "ai4i_2020_modeling.csv")
    ai_train = ai[ai.split == "train"].copy()
    ai_val = ai[ai.split == "validation"].copy()
    ai_cols = ["Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
    ai_models = {
        "Dummy_MostFrequent": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression_Balanced": LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE),
        "RandomForest_Balanced": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
    }
    print("AI4I 2020 baselines")
    ai_results = run_dataset("AI4I_2020", ai_train, ai_val, ai_cols, "Machine failure", lambda X: ai4i_preprocessor(), ai_models)
    all_results.extend(ai_results)

    # RT-IoT2022
    rt = pd.read_csv(PROCESSED / "rt_iot2022_modeling.csv")
    rt_train = rt[rt.split == "train"].copy()
    rt_val = rt[rt.split == "validation"].copy()
    excluded = {"row_id", "duplicate_group_id", "duplicate_group_size", "split", "Attack_type", "binary_target"}
    rt_cols = [c for c in rt.columns if c not in excluded]
    rt_target = "binary_target"
    rt_train[rt_target] = rt_train[rt_target].map({"Normal": 0, "Attack": 1}).astype(int)
    rt_val[rt_target] = rt_val[rt_target].map({"Normal": 0, "Attack": 1}).astype(int)
    # Constant feature found by V0.1.1 audit. Near-constant features remain for baseline/ablation.
    rt_drop = ["bwd_URG_flag_count"]

    def rt_pre(X):
        cats = [c for c in ["proto", "service"] if c in X.columns]
        nums = [c for c in X.columns if c not in cats]
        return ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), nums),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), cats),
        ])

    rt_models = {
        "Dummy_MostFrequent": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression_Balanced": LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE),
        "RandomForest_Balanced": RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
    }
    print("\nRT-IoT2022 binary baselines")
    rt_results = run_dataset("RT_IoT2022", rt_train, rt_val, rt_cols, rt_target, rt_pre, rt_models, drop_cols=rt_drop)
    all_results.extend(rt_results)

    results = pd.DataFrame(all_results)
    results.to_csv(TABLES / "07_baseline_model_results.csv", index=False)

    report.append("AI4I 2020")
    report.append("-" * 90)
    for _, r in results[results.dataset == "AI4I_2020"].iterrows():
        report.append(f"{r.model}: Accuracy={r.accuracy:.4f}, BalancedAcc={r.balanced_accuracy:.4f}, Precision={r.precision_attack_or_failure:.4f}, Recall={r.recall_attack_or_failure:.4f}, F1={r.f1_attack_or_failure:.4f}, ROC-AUC={r.roc_auc:.4f}, PR-AUC={r.pr_auc:.4f}")
    report.append("")
    report.append("RT-IoT2022")
    report.append("-" * 90)
    for _, r in results[results.dataset == "RT_IoT2022"].iterrows():
        report.append(f"{r.model}: Accuracy={r.accuracy:.4f}, BalancedAcc={r.balanced_accuracy:.4f}, Precision={r.precision_attack_or_failure:.4f}, Recall={r.recall_attack_or_failure:.4f}, F1={r.f1_attack_or_failure:.4f}, ROC-AUC={r.roc_auc:.4f}, PR-AUC={r.pr_auc:.4f}, NormalRecall={r.normal_or_nonfailure_recall:.4f}")
    report.extend(["", "Test set was not used in this experiment.", ""])
    (REPORTS / "07_baseline_models.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nBASELINE EXPERIMENT COMPLETE")
    print(f"Metrics: {TABLES / '07_baseline_model_results.csv'}")
    print(f"Report:  {REPORTS / '07_baseline_models.txt'}")


if __name__ == "__main__":
    main()
