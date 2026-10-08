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


def render_machine_health():
    st.header("Machine Health")
    st.write(
        "Enter one machine-telemetry sample. The frozen AI4I XGBoost model will "
        "return a failure probability and two operating decisions."
    )

    with st.form("ai4i_form"):
        left, right = st.columns(2)

        with left:
            machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
            air_temp = st.number_input(
                "Air Temperature (K)", min_value=270.0, max_value=330.0, value=298.2
            )
            process_temp = st.number_input(
                "Process Temperature (K)",
                min_value=280.0,
                max_value=340.0,
                value=308.5,
            )

        with right:
            rotational_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=500.0,
                max_value=3000.0,
                value=1500.0,
            )
            torque = st.number_input(
                "Torque (Nm)", min_value=0.0, max_value=100.0, value=42.0
            )
            tool_wear = st.number_input(
                "Tool Wear (min)", min_value=0.0, max_value=300.0, value=120.0
            )

        submitted = st.form_submit_button("Predict Machine Health", type="primary")

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

    option = st.radio(
        "Input mode",
        ["Synthetic interface sample", "Upload one-row CSV"],
        horizontal=True,
    )

    row_df = None

    if option == "Synthetic interface sample":
        st.warning(
            "This is an interface demonstration only. The synthetic row is not a "
            "benchmark sample and should not be used to claim detection performance."
        )
        if st.button("Run synthetic network-flow demo", type="primary"):
            row_df = pd.DataFrame([{feature: 0.0 for feature in schema}])

    else:
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

    with st.sidebar:
        st.markdown("## SecurePredict")
        page = st.radio(
            "Navigate",
            [
                "Overview",
                "Machine Health",
                "Network Security",
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
    elif page == "Explainability":
        render_explainability()
    elif page == "Model Performance":
        render_model_performance()
    else:
        render_about()


if __name__ == "__main__":
    main()
