import uuid
import pandas as pd
import streamlit as st
from sinica import ProfileReport

from services.kernel_manager import get_active_dataframe_info
from services.llm_agent import generate_questions
from utils.file_helpers import generate_jupyter_notebook


def render_sidebar():
    """Renders the entire sidebar interface."""
    kernel = st.session_state["kernel"]
    df = st.session_state["df"]
    api_key = st.session_state["API"]

    st.sidebar.success("🟢 Engine Active")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Summary")

    shape, schema_df = get_active_dataframe_info(kernel)

    if schema_df is not None and not schema_df.empty:
        st.sidebar.caption(
            f"**File:** `{st.session_state['file_name']}` ({shape[0]} rows × {shape[1]} cols)"
        )
        with st.sidebar.expander("Dataset Summary Statistics", expanded=False):
            pr = ProfileReport(df)
            st.dataframe(pr.summary())

        with st.sidebar.expander("Data Alerts", expanded=False):
            st.dataframe(pr.alerts())
    else:
        st.sidebar.warning("No active `df` found in kernel.")

    # Memory Inspection Sidebar
    with st.sidebar.expander("Current Kernel Variables", expanded=False):
        if kernel:
            manifest = kernel.get_kernel_memory_manifest()
            if manifest:
                var_rows = []
                for var_name, var_info in manifest.items():
                    v_type = var_info.get("type", "Unknown")
                    if v_type in ("DataFrame", "Series"):
                        details = f"Shape: {var_info.get('shape')}"
                    else:
                        details = str(var_info.get("value", ""))
                    var_rows.append(
                        {"Variable": var_name, "Type": v_type, "Details": details}
                    )
                st.dataframe(
                    pd.DataFrame(var_rows).set_index("Variable"),
                    use_container_width=True,
                )
            else:
                st.info("No active user variables in kernel memory.")
        else:
            st.info("Kernel not initialized.")

    # Controls Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Controls")

    if st.sidebar.button("Regenerate Suggestions", use_container_width=True):
        if api_key:
            with st.sidebar:
                with st.spinner("Analyzing active kernel memory..."):
                    st.session_state["questions"] = generate_questions(kernel, api_key)
            st.sidebar.success("Updated suggestions based on active kernel state!")
            st.rerun()
        else:
            st.sidebar.error("API Key missing.")

    if st.sidebar.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.sidebar.success("Chat history cleared!")
        st.rerun()

    if st.session_state.messages:
        nb_json = generate_jupyter_notebook(
            st.session_state.messages, st.session_state.get("file_name", "data.csv")
        )
        st.sidebar.download_button(
            label="Download Notebook (.ipynb)",
            data=nb_json,
            file_name=f"analysis_{uuid.uuid4().hex[:6]}.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        )
