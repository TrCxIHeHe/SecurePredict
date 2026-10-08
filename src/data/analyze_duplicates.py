#Audit duplicate feature rows for potential validation/test leakage.
from __future__ import annotations
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

def audit_dataset(name: str, target_col: str) -> None:
    base = RAW / name
    features = pd.read_csv(base / "features.csv")
    targets = pd.read_csv(base / "targets.csv")
    df = pd.concat(
        [features.reset_index(drop=True), targets.reset_index(drop=True)],
        axis=1,
    )
    duplicate_mask = features.duplicated(keep=False)
    duplicate_rows = df.loc[duplicate_mask].copy()
    print("=" * 90)
    print(f"{name} - DUPLICATE / LEAKAGE AUDIT")
    print("=" * 90)
    total_duplicates = int(features.duplicated().sum())
    print(f"Total rows: {len(features):,}")
    print(f"Exact duplicate feature rows: {total_duplicates:,}")
    if total_duplicates == 0:
        print("No duplicate feature rows found.")
        return
    grouped = (
        duplicate_rows.groupby(
            list(features.columns),
            dropna=False,
            sort=False,
        )[target_col]
        .nunique()
    )
    conflicting_groups = int((grouped > 1).sum())
    same_label_groups = int((grouped == 1).sum())
    print(f"Duplicate groups with same label: {same_label_groups:,}")
    print(f"Duplicate groups with conflicting labels: {conflicting_groups:,}")
    if conflicting_groups > 0:
        print("\nHIGH-RISK FINDING:")
        print(
            "Identical feature vectors appear with different target labels. "
            "A naive random train/test split could leak information."
        )
    else:
        print(
            "\nNo conflicting labels found among exact duplicate feature groups."
        )
    duplicate_group_sizes = (
        features.loc[duplicate_mask]
        .groupby(list(features.columns), dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    print("\nLargest duplicate groups:")
    print(duplicate_group_sizes.head(10).to_string())
def main() -> None:
    audit_dataset("ai4i_2020", "Machine failure")
    print()
    audit_dataset("rt_iot2022", "Attack_type")

if __name__ == "__main__":
    main()