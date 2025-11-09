import streamlit as st
from dcai_clinical_nlp import (
    demonstrate_icd_coding,
    demonstrate_ade_detection,
    demonstrate_fairness_analysis
)

st.set_page_config(page_title="Clinical NLP DCAI Demo", layout="wide")

st.title("🧠 Data-Centric AI for Clinical NLP")
st.markdown("### Explore ICD Coding, ADE Detection, and Fairness Analysis interactively")

# Sidebar options
option = st.sidebar.selectbox(
    "Select Demo",
    ["ICD Coding", "Adverse Drug Event Detection", "Fairness Analysis"]
)

if option == "ICD Coding":
    st.subheader("📘 ICD Code Suggestion")
    st.write("This runs the weak supervision pipeline to assign ICD-10 codes based on discharge summaries.")
    if st.button("Run ICD Coding Demo"):
        with st.spinner("Running ICD coding pipeline..."):
            demonstrate_icd_coding()
        st.success("✅ ICD Coding demo executed successfully!")

elif option == "Adverse Drug Event Detection":
    st.subheader("💊 Adverse Drug Event (ADE) Detection")
    st.write("This demo detects potential drug-related adverse events in clinical notes.")
    if st.button("Run ADE Detection Demo"):
        with st.spinner("Running ADE detection..."):
            demonstrate_ade_detection()
        st.success("✅ ADE detection completed successfully!")

elif option == "Fairness Analysis":
    st.subheader("⚖️ Fairness Analysis")
    st.write("Analyzes performance disparities across patient subgroups.")
    if st.button("Run Fairness Analysis Demo"):
        with st.spinner("Running fairness evaluation..."):
            demonstrate_fairness_analysis()
        st.success("✅ Fairness analysis complete!")
