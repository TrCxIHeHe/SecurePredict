"""
Digital-twin simulator: generates machine telemetry and a network-flow stream.

Everything here is a SIMULATION INPUT (scenario physics, random drift, which flows are
injected where). No model outputs are produced or hard-coded here: statuses, probabilities
and alerts all come from the frozen models in src/inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AI4I_FEATURES = ["Type", "Air temperature", "Process temperature", "Rotational speed", "Torque", "Tool wear"]
MACHINE_TYPES = ["L", "M", "L", "H", "L", "M"]
NORMAL_CLASSES = {"MQTT_Publish", "Thing_Speak", "Wipro_bulb"}

SCENARIOS = {
    "Normal shift":              {"fault": None,       "cyber": False},
    "Tool wear-out":             {"fault": "wear",     "cyber": False},
    "Overload event":            {"fault": "overload", "cyber": False},
    "Cooling fault (blind-spot probe)": {"fault": "cooling", "cyber": False},
    "Cyber attack":              {"fault": None,       "cyber": True},
    "Cyber-physical attack":     {"fault": "overload", "cyber": True},
}


def simulate_telemetry(
    n_ticks: int,
    seed: int,
    fault: str | None = None,
    target: int = 0,
    start: int = 40,
    duration: int = 40,
    n_machines: int = 6,
) -> pd.DataFrame:
    """Rows ordered tick-major, machine-minor. Columns: tick, machine + AI4I_FEATURES."""
    rng = np.random.default_rng(seed)
    base_rpm = rng.normal(1500, 35, n_machines)
    base_tq = rng.normal(40, 2.0, n_machines)
    base_air = rng.normal(298.3, 0.5, n_machines)
    wear0 = rng.uniform(15, 80, n_machines)
    if fault == "wear":
        wear0[target] = 115.0

    rows = []
    for t in range(n_ticks):
        for m in range(n_machines):
            active = fault is not None and m == target and start <= t < start + duration
            ramp = float(np.clip((t - start) / max(1.0, duration * 0.35), 0, 1)) if active else 0.0

            air = base_air[m] + 0.7 * np.sin(2 * np.pi * t / max(n_ticks, 1)) + rng.normal(0, 0.12)
            tq = base_tq[m] + rng.normal(0, 1.8)
            wear = wear0[m] + 1.0 * t
            gap = 10.3 + rng.normal(0, 0.15)

            if active and fault == "overload":
                tq += 27.0 * ramp
            if active and fault == "cooling":
                gap -= 2.4 * ramp
            if fault == "wear" and m == target and t >= start:
                wear += 1.8 * (t - start)

            rpm = base_rpm[m] - 6.0 * (tq - base_tq[m]) + rng.normal(0, 12)
            if active and fault == "cooling":
                rpm -= 175.0 * ramp

            proc = air + gap + 0.02 * (tq - base_tq[m])
            rows.append(
                {
                    "tick": t,
                    "machine": m,
                    "Type": MACHINE_TYPES[m % len(MACHINE_TYPES)],
                    "Air temperature": float(np.clip(air, 271, 329)),
                    "Process temperature": float(np.clip(proc, 281, 339)),
                    "Rotational speed": float(np.clip(rpm, 501, 2999)),
                    "Torque": float(np.clip(tq, 0.5, 99)),
                    "Tool wear": float(np.clip(wear, 0, 299)),
                }
            )
    return pd.DataFrame(rows)


def build_flow_stream(
    pool: pd.DataFrame,
    feature_names: list[str],
    n_ticks: int,
    n_machines: int,
    seed: int,
    attack: bool = False,
    family: str | None = None,
    target: int = 0,
    start: int = 40,
    duration: int = 40,
    intensity: float = 0.8,
) -> pd.DataFrame:
    """
    One network flow per (tick, machine), drawn from REAL benchmark flows in `pool`
    (columns: Attack_type + model features). Benign traffic is drawn from the Normal classes;
    inside the attack window the target machine's flows are drawn from `family` with
    probability `intensity`. Returns features + true_class + true_attack, aligned with
    simulate_telemetry row order.
    """
    rng = np.random.default_rng(seed + 1000)
    benign = pool[pool["Attack_type"].isin(NORMAL_CLASSES)]
    atk = pool[pool["Attack_type"] == family] if family else pool.iloc[0:0]
    if benign.empty:
        raise ValueError("Flow pool has no Normal-class flows.")
    if attack and atk.empty:
        raise ValueError(f"Flow pool has no flows for attack family {family!r}.")

    picks = []
    for t in range(n_ticks):
        for m in range(n_machines):
            in_window = attack and m == target and start <= t < start + duration
            use_attack = in_window and rng.random() < intensity
            src = atk if use_attack else benign
            picks.append(src.iloc[int(rng.integers(0, len(src)))])
    out = pd.DataFrame(picks).reset_index(drop=True)
    out["true_class"] = out["Attack_type"]
    out["true_attack"] = ~out["Attack_type"].isin(NORMAL_CLASSES)
    return out[[*feature_names, "true_class", "true_attack"]]
