import streamlit as st
import pandas as pd
import tempfile
from pathlib import Path
from main_agent import DataCleaningAgent  # Make sure this import matches your backend agent

# -------------------------
# Streamlit App Config
# -------------------------
st.set_page_config(
    page_title="AI Data Cleaning & EDA by Yaseen_Neural",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI Data Science Agent by Yaseen_Neural")
st.caption("Upload your dataset and let the agent handle cleaning, preprocessing, EDA, visualizations, and AutoML.")

# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

# Initialize agent
agent = DataCleaningAgent(output_dir="output_streamlit")

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("⚙️ Pipeline Options")
do_clean = st.sidebar.checkbox("Data Cleaning", value=True)
do_preprocess = st.sidebar.checkbox("Preprocessing", value=True)
do_eda = st.sidebar.checkbox("Perform EDA", value=True)
do_visualize = st.sidebar.checkbox("Visualizations", value=True)
do_export = st.sidebar.checkbox("Export Results", value=True)
run_automl = st.sidebar.checkbox("Run AutoML (Optional)", value=False)
automl_target = None
if run_automl:
    automl_target = st.sidebar.text_input("🎯 Target Column (for ML modeling)")

# -------------------------
# Main Pipeline Execution
# -------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_file_path = tmp.name

    st.success(f"✅ File uploaded: {uploaded_file.name}")

    if st.button("🚀 Run Full Pipeline"):
        try:
            with st.spinner("Running pipeline... Please wait ⏳"):
                results = agent.run_full_pipeline(
                    input_path=temp_file_path,
                    clean=do_clean,
                    preprocess=do_preprocess,
                    eda=do_eda,
                    visualize=do_visualize,
                    export=do_export,
                    run_automl=run_automl,
                    automl_target=automl_target
                )

            st.success("🎉 Pipeline completed successfully!")

            # -------------------------
            # Create Tabs for Results
            # -------------------------
            tabs = st.tabs(["🧹 Data Preview", "📊 EDA Report", "📈 Visualizations", "🤖 AutoML Results", "💾 Exports"])

            # --- Data Preview ---
            with tabs[0]:
                if hasattr(agent, "processed_data"):
                    st.subheader("✅ Processed Data")
                    st.dataframe(agent.processed_data.head())
                    st.write(f"**Shape:** {agent.processed_data.shape}")
                elif hasattr(agent, "cleaned_data"):
                    st.subheader("✅ Cleaned Data")
                    st.dataframe(agent.cleaned_data.head())
                    st.write(f"**Shape:** {agent.cleaned_data.shape}")

            # --- EDA Report ---
            with tabs[1]:
                st.subheader("📊 EDA Insights")
                if "eda_report" in results:
                    with st.expander("📘 View Detailed EDA Report", expanded=True):
                        st.text(agent.eda_analyzer.format_report_text(results["eda_report"]))
                else:
                    st.info("Run EDA to generate a report.")

            # --- Visualizations ---
            with tabs[2]:
                st.subheader("📈 Interactive Visualizations")
                if "visualizations" in results and results["visualizations"]:
                    cols = st.columns(2)
                    for i, plot_path in enumerate(results["visualizations"]):
                        with cols[i % 2]:
                            st.image(plot_path, use_column_width=True)
                else:
                    st.warning("No visualizations found. Make sure 'Visualizations' is selected in the sidebar.")

            # --- AutoML Results ---
            with tabs[3]:
                st.subheader("🤖 Machine Learning Summary")
                if "automl" in results:
                    st.json(results["automl"])
                else:
                    st.info("Enable AutoML in the sidebar to train a model.")

            # --- Exports ---
            with tabs[4]:
                st.subheader("💾 Download Exported Files")
                if "exported_files" in results:
                    for key, path in results["exported_files"].items():
                        if isinstance(path, list):
                            for p in path:
                                st.download_button(
                                    f"⬇️ Download {key} plot",
                                    Path(p).read_bytes(),
                                    file_name=Path(p).name
                                )
                        else:
                            st.download_button(
                                f"⬇️ Download {key}",
                                Path(path).read_bytes(),
                                file_name=Path(path).name
                            )
                else:
                    st.info("No exported files available yet.")

        except Exception as e:
            st.error(f"⚠️ Pipeline failed with error: {str(e)}")

else:
    st.info("📥 Please upload a dataset to begin.")



