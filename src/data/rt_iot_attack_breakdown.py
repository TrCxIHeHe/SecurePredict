"""
V1.0.3 - RT-IoT2022 per-attack-type breakdown of the FROZEN binary detector.

For each ORIGINAL Attack_type in the TEST split: what fraction was classified
Attack? (Normal classes: what fraction was correctly classified Normal?)

Needs data/processed/rt_iot2022_modeling.csv (created by prepare_modeling_data.py)
and reports/v0_1/tables/13_rt_iot2022_test_predictions.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "rt_iot2022_modeling.csv"
TABLES = ROOT / "reports" / "v0_1" / "tables"
REPORTS = ROOT / "reports" / "v0_1"
NORMAL = {"MQTT_Publish", "Thing_Speak", "Wipro_bulb"}


def build_breakdown(modeling: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    labels = modeling.loc[modeling["split"] == "test", ["row_id", "Attack_type"]]
    df = preds[["row_id", "predicted_attack"]].merge(labels, on="row_id", how="left")
    if df["Attack_type"].isna().any():
        raise RuntimeError("Some prediction row_ids were not found in the test split.")
    df["is_normal_class"] = df["Attack_type"].isin(NORMAL)
    df["correct"] = (df["predicted_attack"] == (~df["is_normal_class"]).astype(int))

    out = (
        df.groupby(["Attack_type", "is_normal_class"])
        .agg(samples=("correct", "size"), correct=("correct", "sum"))
        .reset_index()
    )
    out["correct_rate"] = out["correct"] / out["samples"]
    out["miss_rate"] = 1 - out["correct_rate"]
    out["kind"] = out["is_normal_class"].map({True: "Normal (want: Normal)", False: "Attack (want: Attack)"})
    out["share_of_test_%"] = 100 * out["samples"] / out["samples"].sum()
    return out.sort_values("samples", ascending=False)[
        ["Attack_type", "kind", "samples", "share_of_test_%", "correct", "correct_rate", "miss_rate"]
    ]


def main() -> None:
    modeling = pd.read_csv(PROCESSED)
    preds = pd.read_csv(TABLES / "13_rt_iot2022_test_predictions.csv")
    out = build_breakdown(modeling, preds)
    out.to_csv(TABLES / "17_rt_iot2022_attack_type_breakdown.csv", index=False)

    lines = [
        "SECUREPREDICT V1.0.3 - RT-IoT2022 PER-ATTACK-TYPE BREAKDOWN (FROZEN MODEL, TEST SPLIT)",
        "=" * 100,
        out.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "Read: correct_rate = fraction of that original class the binary detector labelled correctly.",
        "Small classes have wide uncertainty; see 'samples'.",
    ]
    (REPORTS / "17_rt_iot2022_attack_type_breakdown.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
