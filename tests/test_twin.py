"""Digital twin: fusion rules, simulator, timeline (model-computed outputs), HTML, page links."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dashboard.twin import build_twin_html  # noqa: E402
from src.inference import (  # noqa: E402
    AI4I_EARLY_WARNING_THRESHOLD as EARLY,
    AI4I_PRIMARY_THRESHOLD as PRIMARY,
    get_expected_features,
    predict_ai4i,
    predict_ai4i_batch,
    predict_rt_iot2022,
    predict_rt_iot2022_batch,
)
from src.inference.fusion import fuse  # noqa: E402
from src.twin.simulator import AI4I_FEATURES, SCENARIOS, simulate_telemetry  # noqa: E402
from src.twin.timeline import build_timeline  # noqa: E402

APP = str(ROOT / "src" / "dashboard" / "app.py")
FLOWS_FILE = ROOT / "reports" / "v0_1" / "tables" / "demo_sample_flows.csv"


@pytest.fixture(scope="module")
def pool():
    """SYNTHETIC flows for plumbing tests only (real benchmark flows are not in the repo)."""
    feats = get_expected_features()["rt_iot2022"]
    rng = np.random.default_rng(1)
    X = np.zeros((2500, 78))
    for i in range(len(X)):
        k = rng.integers(1, 12)
        idx = rng.choice(78, k, replace=False)
        X[i, idx] = rng.choice([0.001, 0.5, 1, 10, 100, 1e3, 1e5], k) * rng.random(k)
    X = pd.DataFrame(X, columns=feats)
    p = predict_rt_iot2022_batch(X)["attack_probability"].to_numpy()
    lo, hi = X[p < 0.42].head(40).copy(), X[p > 0.58].head(40).copy()
    assert len(lo) > 5 and len(hi) > 5
    lo["Attack_type"], hi["Attack_type"] = "Thing_Speak", "ARP_poisioning"
    return pd.concat([lo, hi], ignore_index=True)


PARAMS = dict(scenario="Cyber-physical attack", target=2, n_ticks=60, seed=7, start=20, duration=25,
              family="ARP_poisioning", intensity=0.8)


# ---------------- fusion ----------------
@pytest.mark.parametrize("status,net,expected", [
    ("NORMAL", False, "OK"), ("EARLY WARNING", False, "WARNING"), ("HIGH RISK", False, "ALARM"),
    ("NORMAL", True, "NETWORK ALERT"), ("EARLY WARNING", True, "CRITICAL"), ("HIGH RISK", True, "CRITICAL"),
])
def test_fusion_rules(status, net, expected):
    assert fuse(status, net) == expected


# ---------------- batch == single ----------------
def test_batch_matches_single_ai4i():
    tel = simulate_telemetry(4, 1)
    b = predict_ai4i_batch(tel[AI4I_FEATURES])
    singles = [predict_ai4i(tel.iloc[i][AI4I_FEATURES].to_dict())["failure_probability"] for i in range(len(tel))]
    assert np.allclose(b["failure_probability"], singles)


def test_batch_matches_single_rt(pool):
    feats = get_expected_features()["rt_iot2022"]
    sub = pool[feats].head(5)
    b = predict_rt_iot2022_batch(sub)
    singles = [predict_rt_iot2022(sub.iloc[[i]])["attack_probability"] for i in range(5)]
    assert np.allclose(b["attack_probability"], singles)


def test_batch_validation_rejects_bad_rows():
    tel = simulate_telemetry(2, 1)[AI4I_FEATURES].copy()
    tel.loc[3, "Torque"] = float("nan")
    with pytest.raises(ValueError, match="row 3"):
        predict_ai4i_batch(tel)


# ---------------- simulator ----------------
def test_simulator_is_deterministic_and_within_bounds():
    a, b = simulate_telemetry(30, 5, "overload", 1), simulate_telemetry(30, 5, "overload", 1)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 30 * 6
    predict_ai4i_batch(a[AI4I_FEATURES])  # raises if anything is out of range


def test_overload_only_affects_target_inside_window():
    t = simulate_telemetry(60, 3, "overload", target=2, start=20, duration=20)
    inside = t[(t.machine == 2) & t.tick.between(28, 38)]["Torque"].mean()
    outside = t[(t.machine == 2) & (t.tick < 15)]["Torque"].mean()
    other = t[(t.machine == 4) & t.tick.between(28, 38)]["Torque"].mean()
    assert inside > outside + 15 and abs(other - outside) < 6


# ---------------- timeline: outputs come from the models ----------------
def test_timeline_statuses_match_single_row_inference(pool):
    payload, flows, tel = build_timeline(PARAMS, pool)
    for t, m in [(0, 0), (30, 2), (45, 2), (59, 5)]:
        f = payload["frames"][t][m]
        res = predict_ai4i(tel.iloc[t * 6 + m][AI4I_FEATURES].to_dict())
        assert f["p"] == pytest.approx(res["failure_probability"], abs=1e-5)
        if abs(f["p"] - PRIMARY) > 1e-3 and abs(f["p"] - EARLY) > 1e-3:
            assert (f["st"] == "HIGH RISK") == res["primary_failure"]
        rt = predict_rt_iot2022(flows.iloc[[t * 6 + m]][get_expected_features()["rt_iot2022"]])
        assert f["np"] == pytest.approx(rt["attack_probability"], abs=1e-5)


def test_critical_requires_both_signals(pool):
    payload, _, _ = build_timeline(PARAMS, pool)
    saw_critical = False
    for fr in payload["frames"]:
        for f in fr:
            if f["fu"] == "CRITICAL":
                saw_critical = True
                assert f["cf"] and f["st"] in ("EARLY WARNING", "HIGH RISK")
            if f["fu"] == "NETWORK ALERT":
                assert f["cf"] and f["st"] == "NORMAL"
    assert saw_critical, "cyber-physical scenario should produce a CRITICAL moment"


def test_normal_shift_is_quiet_and_events_are_consistent(pool):
    payload, _, _ = build_timeline(dict(PARAMS, scenario="Normal shift"), pool)
    assert all(f["fu"] == "OK" for fr in payload["frames"] for f in fr)
    assert payload["events"] == []
    p2, _, _ = build_timeline(PARAMS, pool)
    for e in p2["events"]:
        assert p2["frames"][e["t"]][e["m"]]["fu"] == e["lvl"]


def test_machine_only_mode_without_flows():
    payload, flows, _ = build_timeline(dict(PARAMS, scenario="Overload event"), None)
    assert flows is None and payload["netAvailable"] is False
    assert "np" not in payload["frames"][0][0]
    assert payload["kpis"]["peak_failure_probability"] > PRIMARY


def test_kpis_are_internally_consistent(pool):
    k = build_timeline(PARAMS, pool)[0]["kpis"]
    assert k["attack_flows_flagged"] <= k["attack_flows"]
    assert k["benign_flows_flagged"] <= k["benign_flows"]
    assert k["attack_flows"] + k["benign_flows"] == 60 * 6


# ---------------- html ----------------
def test_html_embeds_payload_and_escapes_script_end():
    payload = {"machines": ["</script>X"], "frames": [], "events": [], "levels": [], "thresholds": {}}
    html = build_twin_html(payload)
    assert "__PAYLOAD__" not in html and html.count("</script>") == 2
    data = re.search(r"const D = (.*?);\n", html, re.S).group(1).replace("<\\/", "</")
    assert json.loads(data)["machines"][0] == "</script>X"


# ---------------- streamlit page + cross-page links ----------------
def _open_twin():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.sidebar.radio[0].set_value("Digital Twin").run()
    return at


def test_twin_page_machine_only_runs():
    if FLOWS_FILE.exists():
        pytest.skip("real flows present; covered by the with-flows test")
    at = _open_twin()
    assert not at.exception
    assert any("What the models did in this run" in s.value for s in at.subheader)


def test_twin_links_open_machine_health_and_network_pages(pool):
    created = not FLOWS_FILE.exists()
    if created:
        pool.to_csv(FLOWS_FILE, index=False)
    try:
        at = _open_twin()
        assert not at.exception
        # machine health link
        next(b for b in at.button if b.key == "to_mh").click().run()
        assert not at.exception
        assert at.session_state["page"] == "Machine Health"
        assert any(s.value == "Prediction" for s in at.subheader)          # auto-ran with prefilled reading
        # network link
        at.sidebar.radio[0].set_value("Digital Twin").run()
        next(b for b in at.button if b.key == "to_ns").click().run()
        assert not at.exception
        assert at.session_state["page"] == "Network Security"
        assert any("Prediction" == s.value for s in at.subheader)
    finally:
        if created and FLOWS_FILE.exists():
            FLOWS_FILE.unlink()
