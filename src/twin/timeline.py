"""
Builds the full twin timeline: simulate -> score with the frozen models -> explain with SHAP
-> fuse -> events and KPIs. Every status/probability/alert in the output is computed here by
the models; nothing is hard-coded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.inference import (
    AI4I_EARLY_WARNING_THRESHOLD,
    AI4I_PRIMARY_THRESHOLD,
    explain_ai4i_batch,
    explain_rt_iot2022_batch,
    get_expected_features,
    predict_ai4i_batch,
    predict_rt_iot2022_batch,
)
from src.inference.fusion import LEVELS, explain_fusion, fuse, severity
from src.twin.simulator import (
    AI4I_FEATURES,
    NORMAL_CLASSES,
    SCENARIOS,
    build_flow_stream,
    simulate_telemetry,
)

CONFIRM_WINDOW = 3     # look at the last 3 flows of an asset ...
CONFIRM_COUNT = 2      # ... and call a network attack when >= 2 were flagged (debounce)
TICK_MINUTES = 2       # simulated minutes per tick (display only)
MAX_NET_SHAP = 24


def _status(primary: bool, early: bool) -> str:
    return "HIGH RISK" if primary else ("EARLY WARNING" if early else "NORMAL")


def build_timeline(params: dict[str, Any], pool: pd.DataFrame | None) -> tuple[dict, pd.DataFrame | None, pd.DataFrame]:
    """
    params: scenario, target (int), n_ticks, seed, start, duration, family (str|None), intensity
    pool  : real benchmark flows (Attack_type + 78 features) or None (machine-only mode)
    Returns (payload_for_ui, flows_df_or_None, telemetry_df). flows/telemetry are row-aligned
    (tick-major, machine-minor) for linking back to the Machine Health / Network Security pages.
    """
    sc = SCENARIOS[params["scenario"]]
    n_ticks, seed = int(params["n_ticks"]), int(params["seed"])
    target, start, duration = int(params["target"]), int(params["start"]), int(params["duration"])
    n_machines = 6

    tel = simulate_telemetry(n_ticks, seed, sc["fault"], target, start, duration, n_machines)
    ai = predict_ai4i_batch(tel[AI4I_FEATURES])
    ai_shap = explain_ai4i_batch(tel[AI4I_FEATURES], top_k=3)

    net_on = pool is not None
    flows, rt, rt_shap_map = None, None, {}
    if net_on:
        feats = get_expected_features()["rt_iot2022"]
        flows = build_flow_stream(
            pool, feats, n_ticks, n_machines, seed,
            attack=sc["cyber"], family=params.get("family"), target=target,
            start=start, duration=duration, intensity=float(params.get("intensity", 0.8)),
        )
        rt = predict_rt_iot2022_batch(flows[feats])
        flagged = np.flatnonzero(rt["attack"].to_numpy())
        if len(flagged):
            pick = flagged[np.linspace(0, len(flagged) - 1, min(MAX_NET_SHAP, len(flagged))).astype(int)]
            shap_rows = explain_rt_iot2022_batch(flows.iloc[pick][feats], top_k=3)
            rt_shap_map = {int(i): s for i, s in zip(pick, shap_rows)}

    p = ai["failure_probability"].to_numpy()
    status = [_status(a, b) for a, b in zip(ai["primary_failure"], ai["early_warning"])]
    net_p = rt["attack_probability"].to_numpy() if net_on else None
    net_flag = rt["attack"].to_numpy() if net_on else np.zeros(len(tel), bool)

    frames, level_grid = [], np.zeros((n_ticks, n_machines), int)
    for t in range(n_ticks):
        fr = []
        for m in range(n_machines):
            i = t * n_machines + m
            lo = max(0, t - CONFIRM_WINDOW + 1)
            idx = [tt * n_machines + m for tt in range(lo, t + 1)]
            k = int(net_flag[idx].sum())
            confirmed = net_on and k >= CONFIRM_COUNT
            level = fuse(status[i], confirmed)
            level_grid[t, m] = severity(level)
            row = tel.iloc[i]
            item = {
                "ty": row["Type"], "air": round(float(row["Air temperature"]), 2),
                "pr": round(float(row["Process temperature"]), 2), "rpm": round(float(row["Rotational speed"]), 1),
                "tq": round(float(row["Torque"]), 2), "wr": round(float(row["Tool wear"]), 1),
                "p": round(float(p[i]), 5), "st": status[i],
                "sh": [[n, round(v, 3)] for n, v in ai_shap[i]],
                "fu": level, "wy": explain_fusion(level, status[i], float(p[i]), k, len(idx)),
            }
            if net_on:
                item.update({
                    "np": round(float(net_p[i]), 5), "nf": bool(net_flag[i]),
                    "nt": str(flows.iloc[i]["true_class"]), "ta": bool(flows.iloc[i]["true_attack"]),
                    "cf": bool(confirmed), "ns": [[n, round(v, 4)] for n, v in rt_shap_map.get(i, [])],
                })
            fr.append(item)
        frames.append(fr)

    names = [f"CNC-{m + 1}" for m in range(n_machines)]
    events = _events(frames, names, n_ticks, n_machines, net_on)
    kpis = _kpis(frames, names, sc, params, n_ticks, n_machines, net_on, status, p, flows, net_flag)

    payload = {
        "machines": names,
        "frames": frames,
        "events": events,
        "kpis": kpis,
        "levels": LEVELS,
        "netAvailable": net_on,
        "tickMinutes": TICK_MINUTES,
        "thresholds": {"early": AI4I_EARLY_WARNING_THRESHOLD, "primary": AI4I_PRIMARY_THRESHOLD, "net": 0.5},
        "scenario": params["scenario"],
        "window": {"start": start, "end": start + duration, "target": target,
                   "fault": sc["fault"], "cyber": bool(sc["cyber"] and net_on)},
    }
    return payload, flows, tel


def _events(frames, names, n_ticks, n_machines, net_on):
    ev, last = [], [0] * n_machines
    for t in range(n_ticks):
        for m in range(n_machines):
            f = frames[t][m]
            lvl = LEVELS.index(f["fu"])
            if lvl != last[m]:
                if lvl == 0:
                    text = f"{names[m]} returned to OK"
                else:
                    text = f"{names[m]} {f['fu']}: {f['wy']}"
                    if f["fu"] in ("ALARM", "CRITICAL") and f["sh"]:
                        text += f" Top driver: {f['sh'][0][0]}."
                ev.append({"t": t, "m": m, "lvl": f["fu"], "text": text})
            last[m] = lvl
    return ev


def _kpis(frames, names, sc, params, n_ticks, n_machines, net_on, status, p, flows, net_flag):
    k: dict[str, Any] = {
        "peak_failure_probability": float(np.max(p)),
        "machines_with_alarm": int(len({m for t in range(n_ticks) for m in range(n_machines)
                                         if frames[t][m]["st"] == "HIGH RISK"})),
    }
    start, target = int(params["start"]), int(params["target"])
    if sc["fault"]:
        first = next((t for t in range(start, n_ticks) if frames[t][target]["st"] != "NORMAL"), None)
        k["fault_start_tick"] = start
        k["first_model_reaction_tick"] = first
        k["fault_lead_time_ticks"] = None if first is None else first - start
    if net_on:
        ta = flows["true_attack"].to_numpy()
        k["attack_flows"] = int(ta.sum())
        k["attack_flows_flagged"] = int((net_flag & ta).sum())
        k["benign_flows"] = int((~ta).sum())
        k["benign_flows_flagged"] = int((net_flag & ~ta).sum())
        if sc["cyber"]:
            first = next((t for t in range(start, n_ticks) if frames[t][target]["cf"]), None)
            k["attack_start_tick"] = start
            k["first_confirmed_alert_tick"] = first
            k["detection_delay_ticks"] = None if first is None else first - start
        k["critical_ticks"] = int(sum(1 for t in range(n_ticks) for m in range(n_machines)
                                       if frames[t][m]["fu"] == "CRITICAL"))
    return k
