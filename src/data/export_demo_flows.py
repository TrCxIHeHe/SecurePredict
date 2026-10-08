"""
Export REAL benchmark flows (up to 40 per Attack_type, from the VALIDATION split) that the
dashboard's digital twin replays as network traffic.

    python src/data/export_demo_flows.py

Output: reports/v0_1/tables/demo_sample_flows.csv  (Attack_type + the 78 model features)
The test split is not used.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "rt_iot2022_modeling.csv"
MODEL = ROOT / "models" / "final" / "rt_iot2022_random_forest_final.joblib"
OUT = ROOT / "reports" / "v0_1" / "tables" / "demo_sample_flows.csv"


def pick_samples(df: pd.DataFrame, features: list[str], per_class: int = 40, seed: int = 42) -> pd.DataFrame:
    val = df[df["split"] == "validation"]
    parts = []
    for _, g in val.groupby("Attack_type"):
        parts.append(g.sample(min(per_class, len(g)), random_state=seed))
    return pd.concat(parts)[["Attack_type", *features]].reset_index(drop=True)


def main() -> None:
    features = [str(c) for c in joblib.load(MODEL).feature_names_in_]
    out = pick_samples(pd.read_csv(PROCESSED), features)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} flows -> {OUT}")
    print(out["Attack_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
