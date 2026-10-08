"""
Cross-signal fusion: combine the machine-health verdict and the network-security verdict
for the SAME asset into one operator-facing level.

The two models are trained on unrelated datasets, so they share no asset key in the data.
Fusion is only meaningful once an asset id links a machine to its network flows (in the
twin that mapping is simulated; in a real plant it would come from the asset inventory).
"""

from __future__ import annotations

LEVELS = ["OK", "WARNING", "NETWORK ALERT", "ALARM", "CRITICAL"]   # ascending severity


def fuse(machine_status: str, network_attack: bool) -> str:
    """machine_status: NORMAL | EARLY WARNING | HIGH RISK | (anything else = unknown)."""
    degraded = machine_status in ("EARLY WARNING", "HIGH RISK")
    if network_attack and degraded:
        return "CRITICAL"
    if network_attack:
        return "NETWORK ALERT"
    if machine_status == "HIGH RISK":
        return "ALARM"
    if machine_status == "EARLY WARNING":
        return "WARNING"
    return "OK"


def explain_fusion(level: str, machine_status: str, failure_p: float, net_flagged: int, window: int) -> str:
    """Human-readable reason, built from the computed values."""
    if level == "CRITICAL":
        return (f"Machine is {machine_status.lower()} ({failure_p:.1%} failure probability) AND "
                f"{net_flagged}/{window} recent flows were classified as attacks: possible "
                "cyber-physical incident.")
    if level == "NETWORK ALERT":
        return f"{net_flagged}/{window} recent flows classified as attacks; machine health is normal."
    if level == "ALARM":
        return f"Failure probability {failure_p:.1%} is above the primary threshold."
    if level == "WARNING":
        return f"Failure probability {failure_p:.1%} is above the early-warning threshold."
    return "No warning from either model."


def severity(level: str) -> int:
    return LEVELS.index(level)
