"""
Streamlit Dashboard

Portfolio-grade front end for the frozen V0.5 models and V0.7 inference layer.
This dashboard is a research/benchmark demonstration, not a live industrial
control system.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Make project-root imports work when this file is launched from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.inference import (  # noqa: E402
    AI4I_EARLY_WARNING_THRESHOLD,
    AI4I_PRIMARY_THRESHOLD,
    explain_ai4i,
    explain_rt_iot2022,
    get_expected_features,
    load_models,
    predict_ai4i,
    predict_rt_iot2022,
)

from src.dashboard.twin import build_twin_html  # noqa: E402
from src.inference.fusion import severity  # noqa: E402
from src.twin.simulator import NORMAL_CLASSES, SCENARIOS  # noqa: E402
from src.twin.timeline import TICK_MINUTES, build_timeline  # noqa: E402


st.set_page_config(
    page_title="SecurePredict",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

FIG_DIR = PROJECT_ROOT / "reports" / "v0_1" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "v0_1" / "tables"


AI4I_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]


def _load_model_results() -> dict:
    """Read frozen test metrics from the saved V0.5 report tables (single source of truth)."""
    ai = pd.read_csv(TABLE_DIR / "12_ai4i_final_test_results.csv")
    ai = ai[ai["operating_mode"] == "balanced_primary"].iloc[0]
    rt = pd.read_csv(TABLE_DIR / "13_rt_iot2022_final_test_results.csv").iloc[0]
    return {
        "AI4I 2020": {
            "ROC-AUC": ai["roc_auc"], "PR-AUC": ai["pr_auc"], "F1": ai["f1"],
            "Recall": ai["recall"], "Precision": ai["precision"],
            "Test rows": int(ai["tn"] + ai["fp"] + ai["fn"] + ai["tp"]),
        },
        "RT-IoT2022": {
            "ROC-AUC": rt["roc_auc"], "PR-AUC": rt["pr_auc"], "F1": rt["f1"],
            "Recall": rt["recall"], "Precision": rt["precision"],
            "Test rows": int(rt["tn"] + rt["fp"] + rt["fn"] + rt["tp"]),
        },
    }


MODEL_RESULTS = _load_model_results()


@st.cache_resource
def initialize_models():
    """Load the frozen final models once per Streamlit process."""
    return load_models()


@st.cache_data
def model_schemas():
    return get_expected_features()


def probability_bar(label: str, probability: float) -> None:
    st.metric(label, f"{probability * 100:.2f}%")
    st.progress(max(0.0, min(1.0, probability)))


def render_input_explanation(explanation: dict, title: str) -> None:
    """Show SHAP contributions for the CURRENT input."""
    st.subheader(title)
    unit = explanation["output_space"]
    df = pd.DataFrame(explanation["contributions"]).rename(
        columns={"feature": "Feature", "value": "Input value", "shap": "Contribution"}
    )
    st.caption(
        f"Contributions are in {unit}. Positive values push toward "
        "failure/attack; negative values push away."
    )
    st.bar_chart(df.set_index("Feature")["Contribution"])
    st.dataframe(df, hide_index=True, width="stretch")


def render_header():
    st.title("🏭 SecurePredict")
    st.caption(
        "Explainable AI for Industrial IoT: predictive maintenance + network threat detection"
    )


def render_overview():
    st.header("System Overview")

    st.markdown(
        """
        **SecurePredict** combines two independent ML monitoring streams under one
        Industrial IoT security concept:

        - **Machine Health:** predicts machine failure risk from the AI4I 2020
          predictive-maintenance benchmark.
        - **Network Security:** classifies network-flow traffic as normal or attack
          using the RT-IoT2022 benchmark.
        - **Explainability:** SHAP analysis shows which features influence the models.

        The models, thresholds, and test-set metrics shown here are frozen from the
        completed V0.5 evaluation. This application is a research/benchmark
        demonstration, not a live factory control system.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("AI4I model", "XGBoost")
    c2.metric("RT-IoT2022 model", "Random Forest")
    c3.metric("Explainability", "SHAP")

    st.subheader("Architecture")

    st.code(
        """Industrial IoT
      │
      ├── Machine telemetry
      │      └── XGBoost → failure probability → thresholds
      │
      └── Network flow telemetry
             └── Random Forest → attack probability → risk level
                         │
                         └── SHAP → model explanations
""",
        language="text",
    )

    st.info(
        "The two datasets are modeled independently. Their rows are not merged into "
        "one training table."
    )


MH_DEFAULTS = {
    "mh_type": "L", "mh_air": 298.2, "mh_proc": 308.5,
    "mh_rpm": 1500.0, "mh_tq": 42.0, "mh_wear": 120.0,
}


def render_machine_health():
    st.header("Machine Health")
    st.write(
        "Enter one machine-telemetry sample. The frozen AI4I XGBoost model will "
        "return a failure probability and two operating decisions."
    )

    prefill = st.session_state.pop("_prefill_mh", None)
    if prefill:
        st.session_state.update(prefill)
        st.info("Loaded from the digital twin: " + prefill.get("_source", "selected reading"))
    for key, default in MH_DEFAULTS.items():
        st.session_state.setdefault(key, default)
    autorun = st.session_state.pop("_autorun_mh", False)

    with st.form("ai4i_form"):
        left, right = st.columns(2)

        with left:
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"], key="mh_type")
            air_temp = st.number_input(
                "Air Temperature (K)", min_value=270.0, max_value=330.0, key="mh_air"
            )
            process_temp = st.number_input(
                "Process Temperature (K)", min_value=280.0, max_value=340.0, key="mh_proc"
            )

        with right:
            rotational_speed = st.number_input(
                "Rotational Speed (rpm)", min_value=500.0, max_value=3000.0, key="mh_rpm"
            )
            torque = st.number_input(
                "Torque (Nm)", min_value=0.0, max_value=100.0, key="mh_tq"
            )
            tool_wear = st.number_input(
                "Tool Wear (min)", min_value=0.0, max_value=300.0, key="mh_wear"
            )

        submitted = st.form_submit_button("Predict Machine Health", type="primary") or autorun

    if not submitted:
        st.info(
            f"Primary threshold: {AI4I_PRIMARY_THRESHOLD:.2f} | "
            f"Early-warning threshold: {AI4I_EARLY_WARNING_THRESHOLD:.2f}"
        )
        return

    sample = {
        "Type": machine_type,
        "Air temperature": air_temp,
        "Process temperature": process_temp,
        "Rotational speed": rotational_speed,
        "Torque": torque,
        "Tool wear": tool_wear,
    }

    try:
        result = predict_ai4i(sample)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader("Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        probability_bar("Failure probability", result["failure_probability"])
    with c2:
        st.metric(
            "Primary decision",
            result["primary_status"],
            f"threshold {result['primary_threshold']:.2f}",
        )
    with c3:
        st.metric(
            "Early warning",
            result["early_warning_status"],
            f"threshold {result['early_warning_threshold']:.2f}",
        )

    if result["primary_failure"]:
        st.error(
            "Primary decision: the predicted failure probability is at or above "
            "the frozen 0.77 operating threshold."
        )
    elif result["early_warning"]:
        st.warning(
            "Early-warning condition: probability is below the primary threshold "
            "but at or above the frozen 0.07 early-warning threshold."
        )
    else:
        st.success("No failure warning under either configured decision threshold.")

    st.caption(
        "Thresholds were selected during validation and then frozen before the "
        "untouched test evaluation."
    )

    try:
        render_input_explanation(
            explain_ai4i(sample), "Why this prediction? (SHAP for your input)"
        )
    except Exception as exc:
        st.warning(f"Could not compute explanation: {exc}")


def render_network_security():
    st.header(" Network Security")
    st.write(
        "The final RT-IoT2022 model expects 78 numeric network-flow features. "
        "For a clean demo, start with the synthetic interface sample or upload a "
        "one-row CSV containing the exact model features."
    )

    schema = model_schemas()["rt_iot2022"]

    modes = ["Synthetic interface sample", "Upload one-row CSV"]
    twin_flow = st.session_state.get("twin_flow")
    if twin_flow is not None:
        modes.append("Flow from digital twin")
    if st.session_state.pop("_select_twin_flow", False):
        st.session_state["ns_mode"] = "Flow from digital twin"
    option = st.radio("Input mode", modes, horizontal=True, key="ns_mode")

    row_df = None

    if option == "Synthetic interface sample":
        st.warning(
            "This is an interface demonstration only. The synthetic row is not a "
            "benchmark sample and should not be used to claim detection performance."
        )
        if st.button("Run synthetic network-flow demo", type="primary"):
            row_df = pd.DataFrame([{feature: 0.0 for feature in schema}])

    elif option == "Upload one-row CSV":
        uploaded = st.file_uploader(
            "Upload one CSV row with the 78 numeric features",
            type=["csv"],
            help="The column names must match the final RT-IoT2022 model schema exactly.",
        )
        if uploaded is not None:
            try:
                candidate = pd.read_csv(uploaded)
                missing = [c for c in schema if c not in candidate.columns]
                extra = [c for c in candidate.columns if c not in schema]

                if len(candidate) != 1:
                    st.error("Please upload a CSV containing exactly one row.")
                elif missing:
                    st.error(
                        f"Missing {len(missing)} required feature(s): "
                        + ", ".join(missing)
                    )
                elif extra:
                    st.error(
                        f"Found {len(extra)} unexpected column(s): "
                        + ", ".join(extra)
                    )
                else:
                    row_df = candidate[schema].copy()
            except Exception as exc:
                st.error(f"Could not read the uploaded CSV: {exc}")

    if option == "Flow from digital twin" and twin_flow is not None:
        st.success("Scoring the exact flow selected in the digital twin: " + st.session_state.get("twin_flow_source", ""))
        row_df = twin_flow[schema].copy()

    if row_df is None:
        st.caption(f"Expected feature count: {len(schema)} numeric features.")
        return

    try:
        result = predict_rt_iot2022(row_df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader("Prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        probability_bar("Attack probability", result["attack_probability"])
    with c2:
        st.metric("Classification", result["classification"])
    with c3:
        st.metric("Risk level", result["risk_level"])

    if result["attack"]:
        st.error("The model classifies this flow as ATTACK.")
    else:
        st.success("The model classifies this flow as NORMAL.")

    with st.expander("Show model input schema"):
        st.dataframe(
            pd.DataFrame({"feature": schema}),
            width="stretch",
            hide_index=True,
        )

    try:
        render_input_explanation(
            explain_rt_iot2022(row_df), "Why this prediction? (SHAP for this flow)"
        )
    except Exception as exc:
        st.warning(f"Could not compute explanation: {exc}")


FLOWS_FILE = TABLE_DIR / "demo_sample_flows.csv"
LEVEL_ICON = {"OK": "OK", "WARNING": "WARNING", "NETWORK ALERT": "NETWORK ALERT", "ALARM": "ALARM", "CRITICAL": "CRITICAL"}


@st.cache_data(show_spinner="Simulating the shift and scoring every reading and flow with the frozen models ...")
def _run_twin(params: dict, pool):
    return build_timeline(params, pool)


def _goto(page: str) -> None:
    st.session_state["_goto"] = page
    st.rerun()


@st.fragment
def _twin_inspector(payload: dict, flows, tel) -> None:
    """Pick a machine + moment, see the models' view of it, and jump to the detailed pages."""
    st.subheader("Inspect a moment and open it in the detailed pages")
    n_ticks, names = len(payload["frames"]), payload["machines"]
    c1, c2 = st.columns([1, 3])
    mi = c1.selectbox("Machine", range(len(names)), format_func=lambda i: names[i],
                      index=int(payload["window"]["target"]))
    worst = max(range(n_ticks), key=lambda t: (severity(payload["frames"][t][mi]["fu"]), payload["frames"][t][mi]["p"]))
    tick = c2.slider("Moment (tick)", 0, n_ticks - 1, worst, key=f"twin_tick_{mi}",
                     help=f"Default = the most severe moment for this machine. 1 tick = {TICK_MINUTES} simulated minutes.")
    f = payload["frames"][tick][mi]
    row = tel.iloc[tick * len(names) + mi]
    sample = {k: (row[k] if k == "Type" else float(row[k])) for k in AI4I_FEATURES}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fused status", f["fu"])
    m2.metric("Failure probability", f"{f['p'] * 100:.2f}%", f["st"].title(), delta_color="off")
    if "np" in f:
        m3.metric("Flow attack probability", f"{f['np'] * 100:.1f}%", "flagged" if f["nf"] else "not flagged", delta_color="off")
        m4.metric("Simulated flow source", f["nt"])
    st.caption(f["wy"])

    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("Open in Machine Health", type="primary", key="to_mh"):
        st.session_state["_prefill_mh"] = {
            "mh_type": sample["Type"], "mh_air": sample["Air temperature"], "mh_proc": sample["Process temperature"],
            "mh_rpm": sample["Rotational speed"], "mh_tq": sample["Torque"], "mh_wear": sample["Tool wear"],
            "_source": f"{names[mi]} at tick {tick}",
        }
        st.session_state["_autorun_mh"] = True
        _goto("Machine Health")
    if flows is not None and b2.button("Open flow in Network Security", key="to_ns"):
        schema = model_schemas()["rt_iot2022"]
        st.session_state["twin_flow"] = flows.iloc[[tick * len(names) + mi]][schema].reset_index(drop=True)
        st.session_state["twin_flow_source"] = f"{names[mi]} at tick {tick} (simulated source: {f['nt']})"
        st.session_state["_select_twin_flow"] = True
        _goto("Network Security")

    render_input_explanation(explain_ai4i(sample), f"Why {names[mi]} has this failure probability at tick {tick}")


def render_digital_twin():
    st.header("Digital Twin - Control Room")
    st.caption(
        "A simulated six-machine cell. Every colour, probability, SHAP bar and alert is computed live by the "
        "frozen XGBoost (machines) and Random Forest (network) models, then fused per asset. Only the "
        "telemetry physics and which flows are injected are simulation inputs. Network flows are real "
        "RT-IoT2022 benchmark flows; the flow-to-machine mapping is simulated (the two datasets share no asset id)."
    )

    pool = pd.read_csv(FLOWS_FILE) if FLOWS_FILE.exists() else None
    families = sorted(set(pool["Attack_type"]) - NORMAL_CLASSES) if pool is not None else []
    scenarios = [k for k, v in SCENARIOS.items() if pool is not None or not v["cyber"]]
    if pool is None:
        st.info("Network feed not loaded: run `python src/data/export_demo_flows.py` once to enable the cyber "
                "scenarios and the network layer. Machine-health scenarios work without it.")

    default = {"scenario": "Cyber-physical attack" if pool is not None else "Overload event", "target": 2,
               "n_ticks": 120, "seed": 7, "start": 40, "duration": 40,
               "family": "ARP_poisioning" if "ARP_poisioning" in families else (families[0] if families else None),
               "intensity": 0.8}
    params = st.session_state.setdefault("twin_params", default)
    if params["scenario"] not in scenarios:
        params = default

    with st.form("twin_form"):
        c1, c2, c3, c4 = st.columns(4)
        scenario = c1.selectbox("Scenario", scenarios, index=scenarios.index(params["scenario"]))
        target = c2.selectbox("Target machine", range(6), index=params["target"], format_func=lambda i: f"CNC-{i + 1}")
        start = c3.slider("Event starts at tick", 10, 100, params["start"])
        duration = c4.slider("Event duration (ticks)", 10, 60, params["duration"])
        d1, d2, d3, d4 = st.columns(4)
        n_ticks = d1.slider("Shift length (ticks)", 60, 180, params["n_ticks"], step=10)
        seed = d2.number_input("Random seed", 0, 9999, params["seed"])
        family = d3.selectbox("Attack family", families or ["(feed not loaded)"],
                              index=families.index(params["family"]) if params["family"] in families else 0,
                              disabled=not families)
        intensity = d4.slider("Attack intensity", 0.2, 1.0, float(params["intensity"]), step=0.1)
        if st.form_submit_button("Run simulation", type="primary"):
            params = {"scenario": scenario, "target": int(target), "n_ticks": int(n_ticks), "seed": int(seed),
                      "start": int(start), "duration": int(duration), "family": family if families else None,
                      "intensity": float(intensity)}
            st.session_state["twin_params"] = params
    params = dict(params, start=min(params["start"], params["n_ticks"] - 10))

    try:
        payload, flows, tel = _run_twin(params, pool)
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        return

    components.html(build_twin_html(payload, height=780), height=790)
    st.caption("Drag to rotate, wheel to zoom, click a machine or a heatmap cell to focus. Space = play/pause, arrows = step.")

    k = payload["kpis"]
    st.subheader("What the models did in this run")
    cols = st.columns(5)
    cols[0].metric("Peak failure probability", f"{k['peak_failure_probability'] * 100:.1f}%")
    if "fault_lead_time_ticks" in k:
        lead = k["fault_lead_time_ticks"]
        cols[1].metric("Model reaction to physical fault", "none" if lead is None else f"{lead * TICK_MINUTES} min after onset")
    if "attack_flows" in k:
        a, t_ = k["attack_flows_flagged"], k["attack_flows"]
        cols[2].metric("Attack flows flagged", f"{a}/{t_}" if t_ else "n/a (no attack injected)")
        cols[3].metric("False alarms (benign flows)", f"{k['benign_flows_flagged']}/{k['benign_flows']}")
        if "detection_delay_ticks" in k:
            d = k["detection_delay_ticks"]
            cols[4].metric("Confirmed network alert", "never" if d is None else f"{d * TICK_MINUTES} min after start")
        else:
            cols[4].metric("CRITICAL machine-ticks", k["critical_ticks"])
    if params["scenario"].startswith("Cooling") and k.get("fault_lead_time_ticks") is None:
        st.warning("Blind spot: the model did not react to this cooling fault. The AI4I model uses raw features only, "
                   "so it cannot see temperature-gap effects. This is a documented limitation, not a UI bug.")

    ev = pd.DataFrame(payload["events"])
    if len(ev):
        ev["time"] = ev["t"].map(lambda t: f"T+{t * TICK_MINUTES // 60:02d}:{t * TICK_MINUTES % 60:02d}")
        ev["machine"] = ev["m"].map(lambda m: payload["machines"][m])
        log = ev[["time", "machine", "lvl", "text"]].rename(columns={"lvl": "level"})
        with st.expander(f"Incident log ({len(log)} events)", expanded=False):
            st.dataframe(log, hide_index=True, width="stretch")
            st.download_button("Download incident log (CSV)", log.to_csv(index=False), "securepredict_incidents.csv", "text/csv")

    _twin_inspector(payload, flows, tel)


def render_explainability():
    st.header("🧠 Explainability")

    st.write(
        "V0.6 generated SHAP explanations using validation data only. The images "
        "below are representative explanations from the completed analysis. "
        "For an explanation of YOUR input, use the Machine Health or Network "
        "Security page."
    )

    ai_global = FIG_DIR / "ai4i_shap_global_importance.png"
    ai_local = FIG_DIR / "ai4i_shap_local_example.png"
    rt_global = FIG_DIR / "rt_iot2022_shap_global_importance.png"
    rt_local = FIG_DIR / "rt_iot2022_shap_local_example.png"

    tabs = st.tabs(["AI4I Machine Health", "RT-IoT2022 Network Security"])

    with tabs[0]:
        if ai_global.exists():
            st.image(
                str(ai_global),
                caption="AI4I global SHAP feature importance",
                width="stretch",
            )
        else:
            st.warning("AI4I global SHAP figure not found.")

        if ai_local.exists():
            st.image(
                str(ai_local),
                caption="AI4I representative local SHAP explanation",
                width="stretch",
            )
        else:
            st.warning("AI4I local SHAP figure not found.")

    with tabs[1]:
        if rt_global.exists():
            st.image(
                str(rt_global),
                caption="RT-IoT2022 global SHAP feature importance",
                width="stretch",
            )
        else:
            st.warning("RT-IoT2022 global SHAP figure not found.")

        if rt_local.exists():
            st.image(
                str(rt_local),
                caption="RT-IoT2022 representative local SHAP explanation",
                width="stretch",
            )
        else:
            st.warning("RT-IoT2022 local SHAP figure not found.")

    st.info(
        "These local SHAP plots explain representative validation examples; they "
        "are not presented as explanations of the current dashboard input."
    )


def render_model_performance():
    st.header("📊 Model Performance")

    st.caption(
        "Frozen V0.5 metrics from untouched test sets. These are benchmark results "
        "for the specified datasets, not guaranteed production performance."
    )

    cols = st.columns(2)

    for col, (name, metrics) in zip(cols, MODEL_RESULTS.items()):
        with col:
            st.subheader(name)
            for metric in ["ROC-AUC", "PR-AUC", "F1", "Recall", "Precision"]:
                st.metric(metric, f"{metrics[metric]:.6f}")
            st.caption(f"Untouched test rows: {metrics['Test rows']:,}")


    st.subheader("Generalization stress test (V1.1)")
    ood_path = TABLE_DIR / "18_rt_iot2022_ood_results.csv"
    if ood_path.exists():
        ood_df = pd.read_csv(ood_path)[
            ["family", "unseen_rows", "ood_recall_rows", "seen_recall_rows", "recall_drop_vs_seen", "ood_fpr_normal"]
        ].rename(
            columns={
                "family": "Attack family held out of training",
                "unseen_rows": "Unseen rows",
                "ood_recall_rows": "Detected when UNSEEN",
                "seen_recall_rows": "Detected when SEEN",
                "recall_drop_vs_seen": "Drop",
                "ood_fpr_normal": "False-alarm rate (normal)",
            }
        )
        st.dataframe(ood_df, hide_index=True, width="stretch")
        st.caption(
            "A fresh model is trained with the whole family removed, then asked to flag it. "
            "Benchmark scores above are in-distribution; this table is the harder test."
        )
    else:
        st.info("Run `python src/data/rt_iot_ood_study.py` to generate the OOD results.")

    st.subheader("AI4I operating thresholds")
    threshold_df = pd.DataFrame(
        [
            {
                "Mode": "Primary",
                "Threshold": AI4I_PRIMARY_THRESHOLD,
                "Purpose": "Primary failure decision",
            },
            {
                "Mode": "Early Warning",
                "Threshold": AI4I_EARLY_WARNING_THRESHOLD,
                "Purpose": "Higher-recall early warning",
            },
        ]
    )
    st.dataframe(threshold_df, width="stretch", hide_index=True)


def render_about():
    st.header("About SecurePredict")

    st.markdown(
        """
        ### Project goal

        SecurePredict demonstrates how two independent Industrial IoT monitoring
        problems can be combined into one explainable AI application:

        **Predictive maintenance**
        - AI4I 2020 benchmark
        - XGBoost classifier
        - validation-based threshold study
        - SHAP explainability

        **Network security**
        - RT-IoT2022 benchmark
        - numeric-flow Random Forest
        - feature ablation
        - SHAP explainability

        ### Important limitations

        This is a research/benchmark demonstration. The results are based on public
        benchmark datasets and should not be interpreted as proof of real-world
        factory or network performance. Production deployment would require external
        validation, monitoring, drift detection, security controls, and domain-specific
        operational testing.

        ### Development status

        - V0.5: final held-out benchmark evaluation 
        - V0.6: SHAP explainability 
        - V0.7: reusable inference layer 
        - V0.8: Streamlit dashboard 
        """
    )


def main():
    initialize_models()
    render_header()

    if "_goto" in st.session_state:
        st.session_state["page"] = st.session_state.pop("_goto")

    with st.sidebar:
        st.markdown("## SecurePredict")
        page = st.radio(
            "Navigate",
            key="page",
            options=[
                "Overview",
                "Machine Health",
                "Network Security",
                "Digital Twin",
                "Explainability",
                "Model Performance",
                "About",
            ],
        )

        st.divider()
        st.caption("V1.0 Dashboard")
        st.caption("Models: V0.5 frozen artifacts")
        st.caption("Inference: V0.7")

    if page == "Overview":
        render_overview()
    elif page == "Machine Health":
        render_machine_health()
    elif page == "Network Security":
        render_network_security()
    elif page == "Digital Twin":
        render_digital_twin()
    elif page == "Explainability":
        render_explainability()
    elif page == "Model Performance":
        render_model_performance()
    else:
        render_about()


if __name__ == "__main__":
    main()
