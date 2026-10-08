#Data_Audit
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def inspect(name: str) -> None:
    base = RAW / name
    features = pd.read_csv(base / "features.csv")
    targets = pd.read_csv(base / "targets.csv")
    print("=" * 80)
    print(name)
    print("=" * 80)
    print("Features:", features.shape)
    print("Targets :", targets.shape)
    print("\nFeature columns:")
    for col in features.columns:
        print(f"  - {col}")
    print("\nTarget columns:")
    for col in targets.columns:
        print(f"  - {col}")
    print("\nMissing values:")
    missing = features.isna().sum()
    print(missing[missing.gt(0)].sort_values(ascending=False).to_string() if missing.any() else "  None")
    print("\nDuplicate feature rows:", int(features.duplicated().sum()))
    print("\nTarget distributions:")
    for col in targets.columns:
        print(f"\n{col}:")
        print(targets[col].value_counts(dropna=False).head(20).to_string())


def main() -> None:
    inspect("ai4i_2020")
    inspect("rt_iot2022")


if __name__ == "__main__":
    main()
