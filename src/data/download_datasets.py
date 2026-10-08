#Dataset download
from __future__ import annotations

from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def save_uci_dataset(uci_id: int, name: str) -> None:
    print(f"Fetching UCI dataset {uci_id}: {name}")
    dataset = fetch_ucirepo(id=uci_id)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    out_dir = RAW / name
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "features.csv", index=False)
    targets.to_csv(out_dir / "targets.csv", index=False)
    print(f"  features: {features.shape}")
    print(f"  targets : {targets.shape}")
    print(f"  saved   : {out_dir}")


def main() -> None:
    save_uci_dataset(601, "ai4i_2020")
    save_uci_dataset(942, "rt_iot2022")
    print("\nDatasets downloaded successfully.")


if __name__ == "__main__":
    main()
