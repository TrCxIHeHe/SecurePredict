"""
V1.0.4 - regenerate EXPERIMENT_PROVENANCE.md (artifact-level audit trail).
Run from the repo root:  python src/data/make_provenance.py
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "reports" / "v0_1" / "tables"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for pkg in ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "streamlit", "joblib"]:
        try:
            from importlib.metadata import version
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = "not installed"
    return out


def main() -> None:
    ai = json.loads((TABLES / "12_ai4i_final_test_metadata.json").read_text())
    rt = json.loads((TABLES / "13_rt_iot2022_final_test_metadata.json").read_text())
    models = [
        ROOT / "models" / "final" / "ai4i_xgboost_final.joblib",
        ROOT / "models" / "final" / "rt_iot2022_random_forest_final.joblib",
    ]
    L = [
        "# Experiment Provenance",
        "",
        f"Generated {date.today().isoformat()} by `src/data/make_provenance.py`.",
        "",
        "**Scope of this document.** Git history for v0.1-v0.9 was squashed into one commit, so",
        "it cannot show the order of tuning, threshold selection and test evaluation. The order is",
        "documented by the numbered reports below, each of which states which split it used.",
        "Milestone commits and tags from v1.0 onward provide real chronological provenance.",
        "",
        "## Protocol",
        "| Stage | Report | Data used | Test touched? |",
        "|---|---|---|---|",
        "| Cleaning and split policy | 04_data_cleaning_and_split_policy.txt | full data | no |",
        "| Feature quality | 06_feature_quality_audit.txt | train | no |",
        "| Baselines | 07_baseline_models.txt | fit train, eval validation | no |",
        "| RT-IoT feature ablation | 08_rt_iot_feature_ablation.txt | fit train, eval validation | no |",
        "| XGBoost baselines | 09_xgboost_baselines.txt | fit train, eval validation | no |",
        "| AI4I tuning | 10_ai4i_xgboost_tuning.txt | train only (4-fold CV) | no |",
        "| AI4I threshold study | 11_ai4i_threshold_study.txt | validation | no |",
        "| AI4I final test | 12_ai4i_final_test.txt | fit train+val, eval test once | first use |",
        "| RT-IoT final test | 13_rt_iot2022_final_test.txt | fit train+val, eval test once | first use |",
        "| SHAP | 14_shap_explainability.txt | validation | no |",
        "",
        "## Frozen decisions",
        f"- AI4I primary threshold: {ai['primary_threshold']}; early-warning threshold: {ai['early_warning_threshold']}",
        f"- AI4I fit rows: {ai['fit_rows']:,}; test rows: {ai['test_rows']:,}; test failures: {ai['test_failure_count']}",
        f"- RT-IoT2022 threshold: {rt['threshold']}; fit rows: {rt['fit_rows']:,}; test rows: {rt['test_rows']:,}; numeric features: {rt['n_numeric_features']}",
        "",
        "## Final model artifacts (SHA-256)",
        "| File | Size (bytes) | SHA-256 |",
        "|---|---|---|",
    ]
    for m in models:
        L.append(f"| `{m.relative_to(ROOT).as_posix()}` | {m.stat().st_size:,} | `{sha256(m)}` |")
    L += [
        "",
        "## Environment used to generate this file",
        "Pickles were originally created with the versions pinned in `requirements.txt`.",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    L += [f"| {k} | {v} |" for k, v in versions().items()]
    (ROOT / "EXPERIMENT_PROVENANCE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("Wrote EXPERIMENT_PROVENANCE.md")


if __name__ == "__main__":
    main()
