import os
import random as rd

import pandas as pd
import plotly.io as pio
import streamlit as st

# Setup default configurations before any rendering
pio.templates.default = "plotly_dark"
st.set_page_config(page_title="Data Analysis Engine", layout="wide")

# Import Modularized Components
from components.chat_ui import execute_and_render, render_buttons
from components.sidebar import render_sidebar
from services.kernel_manager import StreamlitJupyterKernel
from services.llm_agent import generate_questions, get_answer
from utils.formatters import extract_command, format_history_for_prompt

# ---------------------------------------------------------------------------- #
#                             Session State Setup                              #
# ---------------------------------------------------------------------------- #
if "kernel" not in st.session_state:
    st.session_state["kernel"] = None
if "questions" not in st.session_state:
    st.session_state["questions"] = None
if "API" not in st.session_state:
    st.session_state["API"] = st.secrets.get("API", os.getenv("GROQ_API_KEY", ""))
if "df" not in st.session_state:
    st.session_state["df"] = None
if "file_name" not in st.session_state:
    st.session_state["file_name"] = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_input" not in st.session_state:
    st.session_state.user_input = None


# ---------------------------------------------------------------------------- #
#                                Main App Loop                                 #
# ---------------------------------------------------------------------------- #
if st.session_state["df"] is not None:
    # 1. Kernel & LLM Initialization flow
    if st.session_state["kernel"] is None or st.session_state["questions"] is None:
        with st.sidebar:
            with st.status("Initializing Data Context...", expanded=True) as status:
                # Boot Kernel
                if st.session_state["kernel"] is None:
                    st.write("Starting Jupyter Kernel...")
                    kernel_obj = StreamlitJupyterKernel()
                    success, error_msg = kernel_obj.inject_dataframe(
                        st.session_state["df"]
                    )
                    if success:
                        st.session_state["kernel"] = kernel_obj
                        st.write("✓ Interactive Jupyter kernel ready")
                    else:
                        st.error("Failed to load DataFrame into Kernel:")
                        st.code(error_msg)
                        st.stop()

                # Boot LLM Questions
                if st.session_state["questions"] is None and st.session_state["API"]:
                    try:
                        st.write("Generating schema-aware questions...")
                        st.session_state["questions"] = generate_questions(
                            st.session_state["kernel"], st.session_state["API"]
                        )
                        st.write("✓ Generated exploration questions")
                    except Exception as e:
                        st.write(f"Could not generate questions: {e}")

                status.update(label="System Ready!", state="complete", expanded=False)

    # 2. Render Sidebar
    render_sidebar()

    # 3. Render Historical Chat Interface
    st.markdown(
        """<div align="center">

# Zeno is ready!


##### **Try asking:** 
*"Summarize this dataset"*<br>
*"Are there missing values or outliers?"*<br>
*"Perform correlation analysis"* <br>

</div>

""",
        unsafe_allow_html=True,
    )
    st.image("assets/chat_intro.png", use_container_width=True)
    st.markdown(
        """<div align = 'center'>
Or select a recommendation below: <div>
                 """,
        unsafe_allow_html=True,
    )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.expander("View Executed Code"):
                    st.code(message["content"], language="python")
                execute_and_render(message["content"], st.session_state["kernel"])

    st.divider()
    render_buttons()

    # 4. Input Handling
    chat_box_input = st.chat_input("Ask a question or type / python code...")
    if chat_box_input:
        st.session_state.user_input = chat_box_input

    if st.session_state.user_input:
        prompt = st.session_state.user_input
        st.session_state.user_input = None

        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        # Branch A: Direct Slash Command Execution
        if prompt.strip().startswith("/"):
            raw_code = extract_command(prompt)
            formatted_response = f"```python\n{raw_code}\n```"

            with st.chat_message("assistant"):
                st.caption("⚡ *Executed directly in Jupyter Kernel (LLM Bypassed)*")
                with st.expander("View Executed Code"):
                    st.code(raw_code, language="python")

            st.session_state.messages.append(
                {"role": "assistant", "content": formatted_response}
            )
            execute_and_render(formatted_response, st.session_state["kernel"])

        # Branch B: Standard LLM Query Processing
        else:
            if not st.session_state["API"]:
                st.error("API Key missing. Please set your Groq API Key.")
            else:
                with st.sidebar:
                    sidebar_status = st.status("Engine is thinking..", expanded=True)

                with st.chat_message("assistant"):
                    kernel_manifest = st.session_state[
                        "kernel"
                    ].get_kernel_memory_manifest()
                    history_str = format_history_for_prompt(st.session_state.messages)

                    stream = get_answer(
                        prompt, kernel_manifest, history_str, st.session_state["API"]
                    )
                    full_response = st.write_stream(stream)

                sidebar_status.update(
                    label="⚡ Code Executing in Kernel...", state="running"
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

                execute_and_render(full_response, st.session_state["kernel"])
                sidebar_status.update(
                    label="✓ Execution Complete", state="complete", expanded=False
                )

        # Shuffle suggestion buttons on completion
        if st.session_state["questions"]:
            rd.shuffle(st.session_state["questions"])

        st.rerun()

else:
    # 5. Onboarding / File Upload UI
    st.markdown("""
    # Data Analysis Engine
    Ask **natural language questions** or execute **direct python commands** with full state persistence inside a live Jupyter Kernel.
    """)

    st.sidebar.info("Waiting for dataset upload...")

    with st.form("Start"):
        st.subheader("Upload Dataset to Begin")
        file = st.file_uploader("Upload CSV data file", type=["csv"])
        submit = st.form_submit_button("Start Engine")

        if submit:
            if file is not None:
                st.session_state["file_name"] = file.name
                st.session_state["df"] = pd.read_csv(file)
                st.session_state["questions"] = None

                if st.session_state["kernel"] is not None and hasattr(
                    st.session_state["kernel"], "stop"
                ):
                    st.session_state["kernel"].stop()

                st.session_state["kernel"] = None
                st.rerun()
            else:
                st.error("Please select a valid CSV file before submitting.")
