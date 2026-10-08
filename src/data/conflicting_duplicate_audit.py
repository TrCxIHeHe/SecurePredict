#Detailed audit of conflicting duplicate feature groups.

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports" / "v0_1"
TABLES = REPORTS / "tables"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    feature_path = RAW / "rt_iot2022" / "features.csv"
    target_path = RAW / "rt_iot2022" / "targets.csv"

    features = pd.read_csv(feature_path)
    targets = pd.read_csv(target_path)

    target_col = "Attack_type"

    df = pd.concat(
        [
            features.reset_index(drop=True),
            targets.reset_index(drop=True),
        ],
        axis=1,
    )

    feature_columns = list(features.columns)

    # Find feature groups that occur more than once.
    duplicate_mask = features.duplicated(keep=False)

    duplicate_df = df.loc[duplicate_mask].copy()

    grouped_targets = (
        duplicate_df
        .groupby(feature_columns, dropna=False)[target_col]
        .agg(["nunique", "size", list])
        .reset_index()
    )

    conflicting = grouped_targets[grouped_targets["nunique"] > 1].copy()

    print("=" * 100)
    print("RT-IoT2022 - CONFLICTING DUPLICATE AUDIT")
    print("=" * 100)

    print(f"Total rows: {len(df):,}")
    print(f"Duplicate feature rows: {int(duplicate_mask.sum()):,}")
    print(f"Conflicting duplicate groups: {len(conflicting):,}")

    if conflicting.empty:
        print("\nNo conflicting duplicate groups found.")
        return

    # Save compact summary containing only target information and group size.
    summary = conflicting[["nunique", "size", "list"]].copy()
    summary = summary.rename(
        columns={
            "nunique": "unique_labels",
            "size": "rows_in_group",
            "list": "labels",
        }
    )

    summary["labels"] = summary["labels"].apply(
        lambda labels: ", ".join(map(str, sorted(set(labels))))
    )

    summary.insert(0, "group_id", range(1, len(summary) + 1))

    output_csv = TABLES / "rt_iot2022_conflicting_duplicate_groups.csv"
    summary.to_csv(output_csv, index=False)

    print("\nConflicting groups:")
    print(summary.to_string(index=False))

    affected_rows = int(summary["rows_in_group"].sum())

    print("\n" + "-" * 100)
    print(f"Total rows belonging to conflicting groups: {affected_rows:,}")
    print(f"Saved summary: {output_csv}")

    print("\nInterpretation:")
    print(
        "These groups contain identical feature vectors associated with multiple "
        "target labels. They require explicit handling before a leakage-safe "
        "train/validation/test split is finalized."
    )


if __name__ == "__main__":
    main()