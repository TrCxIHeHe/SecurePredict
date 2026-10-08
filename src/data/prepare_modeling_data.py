#Modeling Data Preparation

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"

RANDOM_STATE = 42

AI4I_INPUT_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

AI4I_TARGET = "Machine failure"

RT_TARGET = "Attack_type"

# Normal traffic classes in RT-IoT2022.
RT_NORMAL_CLASSES = {
    "MQTT_Publish",
    "Thing_Speak",
    "Wipro_bulb",
}


def ensure_directories() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def select_best_group_fold(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    target_fraction: float,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the StratifiedGroupKFold fold closest to the desired size
    and class distribution.
    """

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    classes = sorted(y.unique())

    overall_distribution = (
        y.value_counts(normalize=True)
        .reindex(classes, fill_value=0)
    )

    best_score = float("inf")
    best_train_idx = None
    best_eval_idx = None

    for fold_id, (train_idx, eval_idx) in enumerate(
        splitter.split(X, y, groups)
    ):
        eval_fraction = len(eval_idx) / len(y)

        eval_distribution = (
            y.iloc[eval_idx]
            .value_counts(normalize=True)
            .reindex(classes, fill_value=0)
        )

        size_error = abs(eval_fraction - target_fraction)
        distribution_error = float(
            np.abs(
                eval_distribution.values
                - overall_distribution.values
            ).sum()
        )

        score = size_error + distribution_error

        print(
            f"  Candidate fold {fold_id}: "
            f"size={eval_fraction:.4f}, "
            f"distribution_error={distribution_error:.4f}, "
            f"score={score:.4f}"
        )

        if score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_eval_idx = eval_idx

    assert best_train_idx is not None
    assert best_eval_idx is not None

    return best_train_idx, best_eval_idx


def save_split_manifest(
    row_ids: pd.Series,
    split_labels: pd.Series,
    path: Path,
) -> None:
    manifest = pd.DataFrame(
        {
            "row_id": row_ids,
            "split": split_labels,
        }
    )

    manifest.to_csv(path, index=False)


def prepare_ai4i() -> None:
    print("\n" + "=" * 90)
    print("AI4I 2020 - MODELING DATA PREPARATION")
    print("=" * 90)

    features = pd.read_csv(
        RAW / "ai4i_2020" / "features.csv"
    )

    targets = pd.read_csv(
        RAW / "ai4i_2020" / "targets.csv"
    )

    target = targets[AI4I_TARGET].copy()

    modeling_df = features[AI4I_INPUT_COLUMNS].copy()
    modeling_df[AI4I_TARGET] = target.values
    modeling_df.insert(
        0,
        "row_id",
        np.arange(len(modeling_df)),
    )

    # Create stratified train/test split: 80/20.
    train_val, test = train_test_split(
        modeling_df,
        test_size=0.20,
        stratify=modeling_df[AI4I_TARGET],
        random_state=RANDOM_STATE,
    )

    # Split remaining 80% into 64% train + 16% validation.
    train, validation = train_test_split(
        train_val,
        test_size=0.20,
        stratify=train_val[AI4I_TARGET],
        random_state=RANDOM_STATE,
    )

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    prepared = pd.concat(
        [train, validation, test],
        axis=0,
        ignore_index=True,
    )

    prepared.to_csv(
        PROCESSED / "ai4i_2020_modeling.csv",
        index=False,
    )

    save_split_manifest(
        prepared["row_id"],
        prepared["split"],
        TABLES / "ai4i_split_manifest.csv",
    )

    print(f"Original rows: {len(modeling_df):,}")
    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")

    print("\nClass distribution:")
    print(
        prepared.groupby(["split", AI4I_TARGET])
        .size()
        .unstack(fill_value=0)
    )

    print(
        "\nSaved:",
        PROCESSED / "ai4i_2020_modeling.csv",
    )


def prepare_rt_iot() -> None:
    print("\n" + "=" * 90)
    print("RT-IoT2022 - MODELING DATA PREPARATION")
    print("=" * 90)

    features = pd.read_csv(
        RAW / "rt_iot2022" / "features.csv"
    )

    targets = pd.read_csv(
        RAW / "rt_iot2022" / "targets.csv"
    )

    df = pd.concat(
        [
            features.reset_index(drop=True),
            targets.reset_index(drop=True),
        ],
        axis=1,
    )

    feature_columns = list(features.columns)

    # Identify feature groups that contain conflicting labels.
    label_nunique = (
        df.groupby(feature_columns, dropna=False)[RT_TARGET]
        .transform("nunique")
    )

    conflict_mask = label_nunique > 1

    conflicting_rows = int(conflict_mask.sum())

    print(
        f"Rows in conflicting duplicate groups: "
        f"{conflicting_rows:,}"
    )

    # Remove only ambiguous groups.
    clean = df.loc[~conflict_mask].copy()
    clean.reset_index(drop=True, inplace=True)

    # Assign duplicate-group IDs.
    clean["duplicate_group_id"] = (
        clean.groupby(
            feature_columns,
            dropna=False,
            sort=False,
        ).ngroup()
    )

    clean["duplicate_group_size"] = (
        clean.groupby("duplicate_group_id")[
            "duplicate_group_id"
        ].transform("size")
    )

    # Binary intrusion target.
    clean["binary_target"] = np.where(
        clean[RT_TARGET].isin(RT_NORMAL_CLASSES),
        "Normal",
        "Attack",
    )

    clean.insert(
        0,
        "row_id",
        np.arange(len(clean)),
    )

    print(f"Original rows: {len(df):,}")
    print(f"Cleaned rows: {len(clean):,}")

    print("\nBinary target distribution:")
    print(clean["binary_target"].value_counts())

    # Group-aware 80/20 test split.
    train_val_idx, test_idx = select_best_group_fold(
        clean[feature_columns],
        clean["binary_target"],
        clean["duplicate_group_id"],
        target_fraction=0.20,
    )

    train_val = clean.iloc[train_val_idx].copy()
    test = clean.iloc[test_idx].copy()

    # Group-aware 80/20 split of train_val:
    # final proportions ≈ 64/16/20.
    train_idx, validation_idx = select_best_group_fold(
        train_val[feature_columns],
        train_val["binary_target"],
        train_val["duplicate_group_id"],
        target_fraction=0.20,
    )

    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[validation_idx].copy()

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    prepared = pd.concat(
        [train, validation, test],
        axis=0,
        ignore_index=True,
    )

    prepared.to_csv(
        PROCESSED / "rt_iot2022_modeling.csv",
        index=False,
    )

    save_split_manifest(
        prepared["row_id"],
        prepared["split"],
        TABLES / "rt_iot2022_split_manifest.csv",
    )

    print(f"\nTraining rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")

    print("\nBinary class distribution:")
    print(
        prepared.groupby(["split", "binary_target"])
        .size()
        .unstack(fill_value=0)
    )

    # Verify no duplicate group crosses splits.
    split_counts = (
        prepared.groupby("duplicate_group_id")["split"]
        .nunique()
    )

    crossing_groups = int((split_counts > 1).sum())

    print(
        f"\nDuplicate groups crossing split boundaries: "
        f"{crossing_groups}"
    )

    if crossing_groups != 0:
        raise RuntimeError(
            "Leakage check failed: duplicate groups cross splits."
        )

    # Verify no conflicting groups survived.
    remaining_conflicts = (
        prepared.groupby(feature_columns, dropna=False)[RT_TARGET]
        .nunique()
    )

    remaining_conflicts = int(
        (remaining_conflicts > 1).sum()
    )

    print(
        f"Conflicting duplicate groups remaining: "
        f"{remaining_conflicts}"
    )

    if remaining_conflicts != 0:
        raise RuntimeError(
            "Conflicting duplicate groups remain after cleaning."
        )

    print(
        "\nSaved:",
        PROCESSED / "rt_iot2022_modeling.csv",
    )


def write_report() -> None:
    report_path = (
        REPORTS / "04_data_cleaning_and_split_policy.txt"
    )

    report_text = """
SECUREPREDICT V0.1
DATA CLEANING AND SPLIT POLICY
================================

AI4I 2020
---------
Target:
    Machine failure

Input features:
    Type
    Air temperature
    Process temperature
    Rotational speed
    Torque
    Tool wear

Excluded from model inputs:
    TWF
    HDF
    PWF
    OSF
    RNF

Reason:
    These columns represent failure-mode components associated
    with the Machine failure target and are therefore excluded
    from the primary predictive model to avoid target-derived
    leakage.

Split:
    64% train
    16% validation
    20% test

Stratification:
    Yes


RT-IoT2022
----------
Primary task:
    Binary intrusion detection

Normal classes:
    MQTT_Publish
    Thing_Speak
    Wipro_bulb

Attack classes:
    All remaining Attack_type categories

Duplicate policy:
    Same-label duplicate feature groups are retained.

Conflicting duplicate policy:
    Feature groups associated with multiple target labels are
    removed from the modeling dataset.

Split:
    Approximately 64% train
    Approximately 16% validation
    Approximately 20% test

Split methodology:
    StratifiedGroupKFold

Leakage protection:
    Exact duplicate feature groups are assigned to a single
    split and cannot cross train/validation/test boundaries.


PROJECT PRINCIPLE
-----------------
The raw datasets are never modified.

All cleaning and modeling transformations are performed on
derived files under data/processed/.
"""

    report_path.write_text(
        report_text.strip() + "\n",
        encoding="utf-8",
    )

    print("\nSaved report:", report_path)


def main() -> None:
    ensure_directories()
    prepare_ai4i()
    prepare_rt_iot()
    write_report()

    print("\n" + "=" * 90)
    print("MODELING DATA PREPARATION COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()