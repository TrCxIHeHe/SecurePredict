#SecurePredict Streamlit shell
import streamlit as st

st.set_page_config(page_title="SecurePredict", page_icon="🏭", layout="wide")
st.title("🏭 SecurePredict")
st.caption("Explainable ML for Secure Industrial IoT")
st.info("V0.1 foundation is ready. Model inference and SHAP explanations will be connected after the dataset audit and baseline experiments.")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Machine Health")
    st.metric("Failure Risk", "—")
    st.write("Predictive-maintenance model: coming next")
with col2:
    st.subheader("Network Security")
    st.metric("Threat Risk", "—")
    st.write("IoT intrusion model: coming next")
