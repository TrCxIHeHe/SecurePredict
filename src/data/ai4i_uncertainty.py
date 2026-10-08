"""
V1.0.3 - AI4I uncertainty analysis.

Bootstraps the FROZEN test predictions (threshold 0.77 is NOT changed) to give
95% intervals. Writes reports/v0_1/16_ai4i_uncertainty.txt and a CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "reports" / "v0_1" / "tables"
REPORTS = ROOT / "reports" / "v0_1"
THRESHOLDS = {"primary_0.77": 0.77, "early_warning_0.07": 0.07}
N_BOOT, SEED = 5000, 42


def main() -> None:
    p = pd.read_csv(TABLES / "12_ai4i_test_predictions.csv")
    y = p["actual_failure"].to_numpy()
    s = p["predicted_failure_probability"].to_numpy()
    rng = np.random.default_rng(SEED)

    rows, lines = [], [
        "SECUREPREDICT V1.0.3 - AI4I UNCERTAINTY (FROZEN THRESHOLDS, NO RE-TUNING)",
        "=" * 80,
        f"Test rows: {len(y):,} | failures: {int(y.sum())} | bootstrap resamples: {N_BOOT}",
        "",
    ]
    for mode, th in THRESHOLDS.items():
        pred = (s >= th).astype(int)
        point = {
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        }
        boots = {k: [] for k in point}
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() == 0:
                continue
            bp = (s[i] >= th).astype(int)
            boots["precision"].append(precision_score(y[i], bp, zero_division=0))
            boots["recall"].append(recall_score(y[i], bp, zero_division=0))
            boots["f1"].append(f1_score(y[i], bp, zero_division=0))
        lines.append(f"{mode}")
        for k, v in point.items():
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            rows.append({"mode": mode, "metric": k, "point": v, "ci_low": lo, "ci_high": hi})
            lines.append(f"  {k:<10} {v:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
        lines.append("")

    boots_auc, boots_ap = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots_auc.append(roc_auc_score(y[i], s[i]))
            boots_ap.append(average_precision_score(y[i], s[i]))
    for name, pt, b in [("roc_auc", roc_auc_score(y, s), boots_auc), ("pr_auc", average_precision_score(y, s), boots_ap)]:
        lo, hi = np.percentile(b, [2.5, 97.5])
        rows.append({"mode": "ranking", "metric": name, "point": pt, "ci_low": lo, "ci_high": hi})
        lines.append(f"{name:<10} {pt:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    lines += ["", "Note: intervals reflect test-sample size only (68 failures); thresholds stay frozen."]
    pd.DataFrame(rows).to_csv(TABLES / "16_ai4i_uncertainty.csv", index=False)
    (REPORTS / "16_ai4i_uncertainty.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
