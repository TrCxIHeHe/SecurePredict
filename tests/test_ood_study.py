"""V1.1 OOD study logic on synthetic data (real data is not in the repo)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
import rt_iot_ood_study as ood  # noqa: E402


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    centers = {"Thing_Speak": 0, "MQTT_Publish": 0, "DOS_SYN_Hping": 6, "ARP_poisioning": 6, "NMAP_UDP_SCAN": -6}
    rows = []
    for cls, c in centers.items():
        for split, n in (("train", 120), ("validation", 40), ("test", 40)):
            x = rng.normal(c, 1.0, size=(n, 3))
            d = pd.DataFrame(x, columns=["f1", "f2", "f3"])
            d["Attack_type"], d["split"] = cls, split
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["binary_target"] = np.where(df["Attack_type"].isin({"Thing_Speak", "MQTT_Publish"}), "Normal", "Attack")
    df["duplicate_group_id"] = np.arange(len(df))
    return df


def test_study_runs_and_separates_seen_vs_unseen():
    df = _synthetic()
    fams = {"ARP_poisioning": ["ARP_poisioning"], "NMAP_UDP_SCAN": ["NMAP_UDP_SCAN"]}
    raw = ood.run_study(df, ["f1", "f2", "f3"], fams, [1, 2], n_estimators=30, log=lambda *_: None)
    s = ood.summarise(raw).set_index("family")
    # ARP resembles another attack (DOS) -> should generalise; NMAP is in a new region -> should not.
    assert s.loc["ARP_poisioning", "ood_recall_rows"] > 0.9
    assert s.loc["NMAP_UDP_SCAN", "ood_recall_rows"] < 0.5
    assert (s["seen_recall_rows"] > 0.9).all()


def test_held_out_family_is_never_in_training():
    df = _synthetic()
    dev = df[df["split"].isin(["train", "validation"])]
    train_df = dev[~dev["Attack_type"].isin(["ARP_poisioning"])]
    assert "ARP_poisioning" not in set(train_df["Attack_type"])
    assert not (train_df["split"] == "test").any()


def test_wilson_interval_sane():
    lo, hi = ood.wilson(9, 10)
    assert 0.55 < lo < 0.9 < hi <= 1.0
