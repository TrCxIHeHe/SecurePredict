import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
import ai4i_failure_mode_recall as fm  # noqa: E402
import export_demo_flows as ex  # noqa: E402


def test_export_uses_validation_split_only_and_caps_per_class():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((300, 3)), columns=["a", "b", "c"])
    df["Attack_type"] = ["X", "Y", "Z"] * 100
    df["split"] = ["validation"] * 150 + ["test"] * 150
    out = ex.pick_samples(df, ["a", "b", "c"], per_class=10)
    assert out["Attack_type"].value_counts().max() == 10 and len(out) == 30
    assert set(out.columns) == {"Attack_type", "a", "b", "c"}
    # none of the exported rows may come from the test split
    test_vals = set(map(tuple, df[df.split == "test"][["a", "b", "c"]].round(12).to_numpy()))
    assert not any(tuple(r) in test_vals for r in out[["a", "b", "c"]].round(12).to_numpy())


def test_mode_recall_counts_and_ci():
    modes = pd.DataFrame({"HDF": [1, 1, 1, 1, 0, 0], "PWF": [0, 0, 0, 0, 1, 1]})
    preds = pd.DataFrame({"row_id": range(6), "predicted_failure_primary": [0, 0, 0, 0, 1, 1],
                          "predicted_failure_early_warning": [0, 0, 1, 0, 1, 1]})
    r = fm.mode_recall(modes, preds).set_index(["mode", "operating_mode"])
    assert r.loc[("HDF", "primary_0.77"), "recall"] == 0.0
    assert r.loc[("HDF", "early_warning_0.07"), "detected"] == 1
    assert r.loc[("PWF", "primary_0.77"), "recall"] == 1.0
    assert 0 <= r.loc[("HDF", "primary_0.77"), "ci_high"] < 0.6
