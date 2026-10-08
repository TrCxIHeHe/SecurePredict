"""
V1.1 - RT-IoT2022 leave-one-attack-family-out (OOD / generalization) study.

Question: can the binary detector flag an attack FAMILY it never saw in training?

Protocol (the V0.5 benchmark and frozen model are NOT touched):
  * Development pool = rows with split in {train, validation}. The test split is
    never used for fitting.
  * For each family F: fit a FRESH model (same pipeline and hyperparameters as the
    final model) on the development pool with every row of F REMOVED.
  * Unseen-attack rows  = ALL rows of F (any split; F was never trained on).
    Negative rows       = Normal rows of the TEST split (never trained on).
  * Threshold fixed at 0.50. Nothing is tuned on the unseen family.
  * Reference ("seen") = a model trained on the full development pool, evaluated on
    the TEST-split rows of F plus the same test Normal rows.
  * Repeated over several random seeds (mean +/- std).

Usage (repo root, after prepare_modeling_data.py):
    python src/data/rt_iot_ood_study.py
    python src/data/rt_iot_ood_study.py --seeds 42 43 --families ARP_poisioning
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "rt_iot2022_modeling.csv"
FINAL_MODEL = ROOT / "models" / "final" / "rt_iot2022_random_forest_final.joblib"
TABLES = ROOT / "reports" / "v0_1" / "tables"
REPORTS = ROOT / "reports" / "v0_1"

FAMILIES: dict[str, list[str]] = {
    "ARP_poisioning": ["ARP_poisioning"],
    "NMAP_scans": [
        "NMAP_UDP_SCAN", "NMAP_XMAS_TREE_SCAN", "NMAP_OS_DETECTION",
        "NMAP_TCP_scan", "NMAP_FIN_SCAN",
    ],
    "DDOS_Slowloris": ["DDOS_Slowloris"],
    "Metasploit_SSH": ["Metasploit_Brute_Force_SSH"],
    "DOS_SYN_Hping": ["DOS_SYN_Hping"],
}
THRESHOLD = 0.5


def make_model(seed: int, n_estimators: int = 200) -> Pipeline:
    """Same recipe as the frozen final model (imputer + balanced RF)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced_subsample",
            min_samples_leaf=2, n_jobs=-1, random_state=seed)),
    ])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _scores(model: Pipeline, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(df[features])[:, 1]


def _eval(pos: pd.DataFrame, neg: pd.DataFrame, model: Pipeline, features: list[str]) -> dict:
    sp, sn = _scores(model, pos, features), _scores(model, neg, features)
    y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
    s = np.r_[sp, sn]
    pred = (s >= THRESHOLD).astype(int)
    first = ~pos["duplicate_group_id"].duplicated().to_numpy()   # one row per unique flow pattern
    out = {
        "n_rows": len(pos),
        "n_unique_patterns": int(first.sum()),
        "recall_rows": float((sp >= THRESHOLD).mean()),
        "recall_patterns": float((sp[first] >= THRESHOLD).mean()),
        "median_attack_score": float(np.median(sp)),
        "fpr_normal": float((sn >= THRESHOLD).mean()),
        "n_normal_eval": len(neg),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "eval_attack_share": float(len(sp) / len(s)),
    }
    out["roc_auc"] = float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")
    out["pr_auc"] = float(average_precision_score(y, s)) if y.sum() > 0 else float("nan")
    return out


def run_study(
    df: pd.DataFrame,
    features: list[str],
    families: dict[str, list[str]],
    seeds: list[int],
    n_estimators: int = 200,
    log=print,
) -> pd.DataFrame:
    dev = df[df["split"].isin(["train", "validation"])]
    test = df[df["split"] == "test"]
    test_normal = test[test["binary_target"] == "Normal"]
    rows = []

    refs = {}
    for seed in seeds:
        log(f"[seed {seed}] reference model on full development pool ...")
        refs[seed] = make_model(seed, n_estimators).fit(
            dev[features], (dev["binary_target"] == "Attack").astype(int))

    for fam, classes in families.items():
        all_fam = df[df["Attack_type"].isin(classes)]
        test_fam = test[test["Attack_type"].isin(classes)]
        train_df = dev[~dev["Attack_type"].isin(classes)]
        if all_fam.empty:
            log(f"  skip {fam}: no rows")
            continue
        for seed in seeds:
            log(f"[seed {seed}] hold out {fam}: train rows {len(train_df):,}, unseen rows {len(all_fam):,}")
            model = make_model(seed, n_estimators).fit(
                train_df[features], (train_df["binary_target"] == "Attack").astype(int))
            res = _eval(all_fam, test_normal, model, features)
            ref = _eval(test_fam, test_normal, refs[seed], features) if len(test_fam) else {}
            rows.append({
                "family": fam, "seed": seed, "classes": "+".join(classes),
                "train_rows": len(train_df),
                **{f"ood_{k}": v for k, v in res.items()},
                **{f"seen_{k}": v for k, v in ref.items()},
            })
    return pd.DataFrame(rows)


def summarise(raw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for fam, g in raw.groupby("family", sort=False):
        k = int(round(g["ood_recall_rows"].iloc[0] * g["ood_n_rows"].iloc[0]))
        lo, hi = wilson(k, int(g["ood_n_rows"].iloc[0]))
        out.append({
            "family": fam,
            "classes": g["classes"].iloc[0],
            "unseen_rows": int(g["ood_n_rows"].iloc[0]),
            "unseen_patterns": int(g["ood_n_unique_patterns"].iloc[0]),
            "ood_recall_rows": g["ood_recall_rows"].mean(),
            "ood_recall_std": g["ood_recall_rows"].std(ddof=0),
            "ood_recall_ci_lo_seed0": lo,
            "ood_recall_ci_hi_seed0": hi,
            "ood_recall_patterns": g["ood_recall_patterns"].mean(),
            "ood_median_attack_score": g["ood_median_attack_score"].mean(),
            "ood_fpr_normal": g["ood_fpr_normal"].mean(),
            "ood_roc_auc": g["ood_roc_auc"].mean(),
            "ood_f1": g["ood_f1"].mean(),
            "seen_recall_rows": g.get("seen_recall_rows", pd.Series([np.nan])).mean(),
            "seen_f1": g.get("seen_f1", pd.Series([np.nan])).mean(),
            "seeds": len(g),
        })
    s = pd.DataFrame(out)
    s["recall_drop_vs_seen"] = s["seen_recall_rows"] - s["ood_recall_rows"]
    return s


def write_report(summary: pd.DataFrame, path: Path) -> str:
    cols = ["family", "unseen_rows", "unseen_patterns", "ood_recall_rows", "ood_recall_std",
            "ood_recall_patterns", "ood_fpr_normal", "ood_roc_auc", "ood_f1",
            "seen_recall_rows", "recall_drop_vs_seen"]
    lines = [
        "SECUREPREDICT V1.1 - RT-IoT2022 LEAVE-ONE-ATTACK-FAMILY-OUT (OOD) STUDY",
        "=" * 110,
        "Fresh model per family, family REMOVED from training; threshold fixed at 0.50; frozen V0.5 model untouched.",
        "ood_*  : family unseen in training (all its rows) + test-split Normal rows as negatives.",
        "seen_* : reference model trained on everything, scored on that family's TEST rows + same negatives.",
        "recall_patterns counts each unique flow pattern once (guards against duplicate-heavy floods).",
        "ood_f1 / ood_roc_auc depend on the attack/normal mix of the evaluation set; compare recall first.",
        "",
        summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "Interpretation guide: a large recall_drop_vs_seen means the detector relies on having seen that",
        "attack family; a small drop means the numeric flow features generalise across families on this dataset.",
        "These are laboratory-testbed results on one public benchmark, not production guarantees.",
    ]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=PROCESSED)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--n-estimators", type=int, default=200)
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    if FINAL_MODEL.exists():
        import joblib
        features = [str(c) for c in joblib.load(FINAL_MODEL).feature_names_in_]
    else:
        raise SystemExit(f"Final model not found: {FINAL_MODEL}")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"Processed data is missing model features: {missing[:5]} ...")

    fams = {k: FAMILIES[k] for k in a.families}
    raw = run_study(df, features, fams, a.seeds, a.n_estimators)
    summary = summarise(raw)
    TABLES.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TABLES / "18_rt_iot2022_ood_runs.csv", index=False)
    summary.to_csv(TABLES / "18_rt_iot2022_ood_results.csv", index=False)
    print(write_report(summary, REPORTS / "18_rt_iot2022_ood_study.txt"))


if __name__ == "__main__":
    main()
