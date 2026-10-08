"""
Per-failure-mode recall of the FROZEN AI4I model on the TEST split.

Why: the overall recall (0.76) hides which failure modes the model actually catches. The
digital-twin 'cooling fault' probe produced no model reaction; this script checks, on the real
test data, whether heat-dissipation failures (HDF) are in fact the weak spot.

Needs data/raw/ai4i_2020/{features,targets}.csv and
reports/v0_1/tables/12_ai4i_test_predictions.csv. Thresholds are the frozen 0.77 / 0.07.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "ai4i_2020"
TABLES = ROOT / "reports" / "v0_1" / "tables"
REPORTS = ROOT / "reports" / "v0_1"
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
NAMES = {"TWF": "tool wear failure", "HDF": "heat dissipation failure", "PWF": "power failure",
         "OSF": "overstrain failure", "RNF": "random failure"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mode_recall(modes: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """modes: index = row_id, columns include MODES (0/1). preds: row_id + frozen-threshold decisions."""
    df = preds.merge(modes[[m for m in MODES if m in modes.columns]], left_on="row_id", right_index=True, how="left")
    out = []
    for m in [m for m in MODES if m in df.columns]:
        sub = df[df[m] == 1]
        n = len(sub)
        for label, col in (("primary_0.77", "predicted_failure_primary"), ("early_warning_0.07", "predicted_failure_early_warning")):
            k = int(sub[col].sum())
            lo, hi = wilson(k, n)
            out.append({"mode": m, "meaning": NAMES[m], "operating_mode": label,
                        "test_failures": n, "detected": k, "recall": k / n if n else float("nan"),
                        "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


def main() -> None:
    feats = pd.read_csv(RAW / "features.csv")
    targs = pd.read_csv(RAW / "targets.csv")
    both = pd.concat([feats, targs], axis=1)
    present = [m for m in MODES if m in both.columns]
    if not present:
        raise SystemExit("Failure-mode columns (TWF, HDF, PWF, OSF, RNF) not found in the raw files.")
    modes = both[present]
    modes.index.name = "row_id"
    res = mode_recall(modes, pd.read_csv(TABLES / "12_ai4i_test_predictions.csv"))
    res.to_csv(TABLES / "19_ai4i_failure_mode_recall.csv", index=False)
    lines = ["SECUREPREDICT V1.2 - AI4I RECALL BY FAILURE MODE (FROZEN MODEL, TEST SPLIT)", "=" * 100,
             res.to_string(index=False, float_format=lambda x: f"{x:.3f}"), "",
             "A failure can carry more than one mode label. Small counts give wide intervals.",
             "Modes are used ONLY for this analysis, never as model inputs."]
    (REPORTS / "19_ai4i_failure_mode_recall.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
