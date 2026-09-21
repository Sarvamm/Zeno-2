# ---------------------------------------------------------------------------- #
#                                   Imports                                    #
# ---------------------------------------------------------------------------- #
import json
import os
import random as rd
import re
import tempfile
import uuid
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from jupyter_client import KernelManager

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

import plotly.io as pio

pio.templates.default = "plotly_dark"


# ---------------------------------------------------------------------------- #
#                              Jupyter Kernel Manager                          #
# ---------------------------------------------------------------------------- #
class StreamlitJupyterKernel:
    """Manages an active ipykernel session per Streamlit session state."""

    def __init__(self):
        # 1. Ensure 'python3' kernel spec exists in environment
        ksm = KernelSpecManager()
        if "python3" not in ksm.find_kernel_specs():
            import subprocess
            import sys

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ipykernel",
                    "install",
                    "--user",
                    "--name",
                    "python3",
                ],
                check=True,
            )

        # 2. Start kernel
        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=10)

    def execute_code(self, code: str) -> list[dict]:
        """Executes code inside active kernel and collects raw iopub messages."""
        msg_id = self.kc.execute(code)
        outputs = []

        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=10)
            except Exception:
                break

            if msg["parent_header"].get("msg_id") != msg_id:
                continue

            msg_type = msg["header"]["msg_type"]
            content = msg["content"]

            if msg_type == "stream":
                outputs.append({"type": "stream", "text": content["text"]})
            elif msg_type in ("display_data", "execute_result"):
                outputs.append({"type": "display_data", "data": content["data"]})
            elif msg_type == "error":
                outputs.append({"type": "error", "traceback": content["traceback"]})
            elif msg_type == "status" and content["execution_state"] == "idle":
                break

        return outputs

    def inject_dataframe(self, df: pd.DataFrame, var_name: str = "df"):
        """Loads uploaded CSV DataFrame into kernel global memory."""
        temp_dir = tempfile.gettempdir()
        temp_csv_path = os.path.join(temp_dir, f"data_{uuid.uuid4().hex[:8]}.csv")
        clean_path = temp_csv_path.replace("\\", "/")

        df.to_csv(temp_csv_path, index=False)

        init_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_dark"

if not hasattr(px, 'figure'):
    px.figure = go.Figure
if not hasattr(px, 'Figure'):
    px.Figure = go.Figure

pio.renderers.default = 'notebook_connected'
{var_name} = pd.read_csv('{clean_path}')
"""
        self.execute_code(init_code)

    def get_kernel_memory_manifest(self) -> dict:
        """Inspects kernel global namespace to retrieve active variables, DataFrames, and shapes."""
        inspection_code = """
import json
import pandas as pd

_vars_summary = {}
_ignore_keys = {'In', 'Out', 'exit', 'quit', 'get_ipython', 'pd', 'px', 'go', 'pio', 'json'}

for _k, _v in list(globals().items()):
    if not _k.startswith('_') and _k not in _ignore_keys:
        if isinstance(_v, pd.DataFrame):
            _vars_summary[_k] = {
                "type": "DataFrame",
                "shape": list(_v.shape),
                "columns": _v.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in _v.dtypes.items()}
            }
        elif isinstance(_v, pd.Series):
            _vars_summary[_k] = {
                "type": "Series",
                "shape": list(_v.shape),
                "dtype": str(_v.dtype)
            }
        elif type(_v).__name__ in ('int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set'):
            val_str = str(_v)
            _vars_summary[_k] = {
                "type": type(_v).__name__,
                "value": val_str[:100] + "..." if len(val_str) > 100 else val_str
            }

print("MANIFEST_START" + json.dumps(_vars_summary) + "MANIFEST_END")
"""
        outputs = self.execute_code(inspection_code)

        for out in outputs:
            if out["type"] == "stream" and "MANIFEST_START" in out["text"]:
                try:
                    raw_json = (
                        out["text"]
                        .split("MANIFEST_START")[1]
                        .split("MANIFEST_END")[0]
                        .strip()
                    )
                    return json.loads(raw_json)
                except Exception:
                    break
        return {}

    def stop(self):
        """Safely shuts down client channels and kernel process."""
        try:
            if hasattr(self, "kc") and self.kc is not None:
                self.kc.stop_channels()
        except Exception:
            pass

        try:
            if hasattr(self, "km") and self.km is not None:
                self.km.shutdown_kernel(now=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------- #
#                                Session State                                 #
# ---------------------------------------------------------------------------- #
if "kernel" not in st.session_state:
    st.session_state["kernel"] = None

if "questions" not in st.session_state:
    st.session_state["questions"] = None

if "API" not in st.session_state:
    st.session_state["API"] = st.secrets.get("API", "")

if "df" not in st.session_state:
    st.session_state["df"] = None

if "file_name" not in st.session_state:
    st.session_state["file_name"] = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_input" not in st.session_state:
    st.session_state.user_input = None


# ---------------------------------------------------------------------------- #
#                               Schema Definition                              #
# ---------------------------------------------------------------------------- #
class ListFormatter(BaseModel):
    questions: list[str] = Field(description="List of 10 data analysis questions")


# ---------------------------------------------------------------------------- #
#                              LLM Initialization                              #
# ---------------------------------------------------------------------------- #
llm = None
if st.session_state["API"]:
    llm = ChatGroq(
        groq_api_key=st.session_state["API"],
        model_name="qwen/qwen3.8-27b",
        temperature=0.3,
    )


# ---------------------------------------------------------------------------- #
#                                  Functions                                   #
# ---------------------------------------------------------------------------- #
def extract_command(input_text: str) -> str:
    """Strips the leading slash and cleans up markdown formatting."""
    cmd = input_text.lstrip("/").strip()
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", cmd, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cmd


def format_history_for_prompt() -> str:
    """Combines interaction logs so LLM retains conversation context."""
    if not st.session_state.messages:
        return "No prior conversation."

    history = []
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant (Executed Code)"
        history.append(f"{role}:\n{msg['content']}\n")

    return "\n".join(history[-6:])


def get_answer(user_prompt: str):
    kernel_manifest = {}
    if st.session_state["kernel"]:
        kernel_manifest = st.session_state["kernel"].get_kernel_memory_manifest()

    task_prompt_template = """You are an expert Python Data Analyst operating inside a persistent Jupyter Kernel session.

Active Jupyter Kernel Memory State:
{kernel_manifest}

Execution History:
{history}

Requirements & Guidelines:
1. State & Variable Continuity:
   - You MUST utilize active variables, created DataFrames, and modified schemas listed in the Kernel Memory State above.
   - Reuse existing variables directly instead of re-loading or re-calculating them.
   Following import are already done, DO NOT IMPORT THEM AGAIN IN YOUR CODE BLOCKS:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
2. Code Standards:
   - For Plots: Use Plotly only. IN plotly dark mode which is already set
   - Plotly Rules:
     * Use Plotly Express functions directly (`px.scatter()`, `px.bar()`, `px.line()`, `px.histogram()`).
     * Always end visualization scripts with `fig.show()`.
3. Return ONLY valid executable Python code wrapped inside standard ```python ... ``` code blocks.

User Question:
{user_prompt}"""

    task_prompt = PromptTemplate.from_template(task_prompt_template)
    task_chain = task_prompt | llm | StrOutputParser()

    return task_chain.stream(
        {
            "kernel_manifest": json.dumps(kernel_manifest, indent=2),
            "history": format_history_for_prompt(),
            "user_prompt": user_prompt,
        }
    )


def get_active_dataframe_info() -> tuple[list[int] | None, pd.DataFrame | None]:
    """Retrieves current column details, missing counts, unique value counts,
    mean, median, mode, min, and max directly from the active Jupyter Kernel memory state.
    """
    if not st.session_state["kernel"]:
        return None, None

    inspection_code = """
import json
import pandas as pd

if 'df' in globals() and isinstance(df, pd.DataFrame):
    _info = []
    for col in df.columns:
        s = df[col]
        entry = {
            "Column": str(col),
            "Data Type": str(s.dtype),
            "Missing": int(s.isnull().sum()),
            "Unique": int(s.nunique())
        }
        if pd.api.types.is_numeric_dtype(s):
            clean_s = s.dropna()
            entry["Mean"] = round(float(clean_s.mean()), 4) if not clean_s.empty else None
            entry["Median"] = round(float(clean_s.median()), 4) if not clean_s.empty else None
            mode_val = clean_s.mode()
            entry["Mode"] = round(float(mode_val.iloc[0]), 4) if not mode_val.empty else None
            entry["Min"] = round(float(clean_s.min()), 4) if not clean_s.empty else None
            entry["Max"] = round(float(clean_s.max()), 4) if not clean_s.empty else None
        else:
            clean_s = s.dropna()
            mode_val = clean_s.mode()
            entry["Mean"] = "-"
            entry["Median"] = "-"
            entry["Mode"] = str(mode_val.iloc[0]) if not mode_val.empty else "-"
            entry["Min"] = str(clean_s.min()) if not clean_s.empty else "-"
            entry["Max"] = str(clean_s.max()) if not clean_s.empty else "-"
        _info.append(entry)
    print("COL_INFO_START" + json.dumps({"shape": list(df.shape), "columns": _info}) + "COL_INFO_END")
"""
    outputs = st.session_state["kernel"].execute_code(inspection_code)

    for out in outputs:
        if out["type"] == "stream" and "COL_INFO_START" in out["text"]:
            try:
                raw_json = (
                    out["text"]
                    .split("COL_INFO_START")[1]
                    .split("COL_INFO_END")[0]
                    .strip()
                )
                data = json.loads(raw_json)
                return data["shape"], pd.DataFrame(data["columns"])
            except Exception:
                break
    return None, None


def get_questions() -> list[str]:
    """Generates analytical questions by inspecting the live active DataFrame schema inside the Jupyter kernel."""
    if not st.session_state.get("kernel"):
        return []

    # 1. Fetch current shape and column metadata directly from active kernel memory
    shape, schema_df = get_active_dataframe_info()

    if schema_df is None or schema_df.empty:
        cols_summary_str = "No active DataFrame found."
        shape_str = "Unknown"
    else:
        shape_str = f"{shape[0]} rows × {shape[1]} columns"
        cols_summary_str = json.dumps(schema_df.to_dict(orient="records"), indent=2)

    # 2. Define schema-aware prompt template
    question_gen_prompt_template = """You are an expert Data Analyst and Analytics Engineer.

Given the CURRENT active state of the pandas DataFrame 'df' running in memory:
- **Dataset Dimensions**: {shape_str}
- **Active Columns, Dtypes, and Summaries**:
{cols_summary_str}

Generate 10 actionable specific analytical short questions or visualization ideas.

Return ONLY the list of 10 structured questions."""

    question_gen_prompt = PromptTemplate.from_template(question_gen_prompt_template)
    llm_structured = llm.with_structured_output(ListFormatter)
    question_gen_chain = question_gen_prompt | llm_structured

    try:
        result: ListFormatter = question_gen_chain.invoke(
            {
                "shape_str": shape_str,
                "cols_summary_str": cols_summary_str,
            }
        )
        return result.questions
    except Exception as e:
        st.sidebar.error(f"Failed to generate schema-aware questions: {e}")
        return [
            "Show summary statistics for numerical columns.",
            "Plot distributions of key categorical variables.",
            "Check for correlations across the dataset.",
        ]


def execute_and_render(code_response: str):
    """Executes code in active Jupyter Kernel and renders rich outputs."""
    match = re.search(r"```python\s*\n(.*?)```", code_response, re.DOTALL)
    if not match:
        st.write(code_response)
        return

    code = match.group(1)
    if not st.session_state["kernel"]:
        st.error("Jupyter kernel is not running.")
        return

    outputs = st.session_state["kernel"].execute_code(code)

    for out in outputs:
        if out["type"] == "stream":
            st.text(out["text"])

        elif out["type"] == "display_data":
            data = out["data"]

            if "application/vnd.plotly.v1+json" in data:
                fig_dict = data["application/vnd.plotly.v1+json"]
                st.plotly_chart(fig_dict, use_container_width=False)

            elif "text/html" in data:
                html_str = data["text/html"]
                if "plotly-graph-div" in html_str:
                    components.html(html_str, height=500, scrolling=True)
                else:
                    st.markdown(html_str, unsafe_allow_html=True)

            elif "image/png" in data:
                st.image(data["image/png"])

            elif "text/plain" in data:
                st.code(data["text/plain"], language="text")

        elif out["type"] == "error":
            clean_tb = "\n".join(
                [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in out["traceback"]]
            )
            st.error(f"Execution Error:\n```\n{clean_tb}\n```")


def generate_jupyter_notebook() -> str:
    """Converts the conversation chat history into a downloadable .ipynb JSON structure."""
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    # Setup cell loading dataset
    file_name = st.session_state.get("file_name", "data.csv")
    setup_code = f"import pandas as pd\nimport plotly.express as px\n\ndf = pd.read_csv('{file_name}')\ndf.head()"
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in setup_code.split("\n")],
        }
    )

    # Process user queries and generated assistant code
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            notebook["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"### User: {msg['content']}\n"],
                }
            )
        elif msg["role"] == "assistant":
            match = re.search(r"```python\s*\n(.*?)```", msg["content"], re.DOTALL)
            code_str = match.group(1) if match else msg["content"]

            notebook["cells"].append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [line + "\n" for line in code_str.split("\n")],
                }
            )

    return json.dumps(notebook, indent=2)


def render_buttons() -> None:
    """Renders quick-select suggestion buttons above the chat bar."""
    if st.session_state["questions"]:
        q1, q2, q3 = st.session_state["questions"][:3]
        left, mid, right = st.columns([1, 1, 1])

        if left.button(q1, key=f"q1_{q1[:10]}"):
            st.session_state.user_input = q1
            st.rerun()
        if mid.button(q2, key=f"q2_{q2[:10]}"):
            st.session_state.user_input = q2
            st.rerun()
        if right.button(q3, key=f"q3_{q3[:10]}"):
            st.session_state.user_input = q3
            st.rerun()


# ---------------------------------------------------------------------------- #
#                                   UI Loop                                    #
# ---------------------------------------------------------------------------- #


# Persistent System Controls & Column Browser in Sidebar

if st.session_state["df"] is not None and llm is not None:
    # Initialize Kernel and Questions if needed (Status logged in Sidebar)
    if st.session_state["kernel"] is None or st.session_state["questions"] is None:
        with st.sidebar:
            with st.status("Initializing Data Context...", expanded=True) as status:
                if st.session_state["kernel"] is None:
                    st.write(" Starting Jupyter Kernel...")
                    st.session_state["kernel"] = StreamlitJupyterKernel()
                    st.session_state["kernel"].inject_dataframe(st.session_state["df"])
                    st.write("✓ Interactive Jupyter kernel ready")

                if st.session_state["questions"] is None:
                    try:
                        st.write(" Generating schema-aware questions...")
                        st.session_state["questions"] = get_questions()
                        st.write("✓ Generated exploration questions")
                    except Exception as e:
                        st.write(" Could not generate questions")

                status.update(label="System Ready!", state="complete", expanded=False)

    # Idle state indicator in Sidebar
    st.sidebar.success("🟢 Model Ready")

    # ------------------------------------------------------------------------ #
    # FEATURE 1: Live Dataset Explorer & Summary Statistics                    #
    # ------------------------------------------------------------------------ #
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Summary")

    # Recalculates dynamically from kernel memory on every rerun
    shape, schema_df = get_active_dataframe_info()

    if schema_df is not None and not schema_df.empty:
        st.sidebar.caption(
            f"**File:** `{st.session_state['file_name']}` ({shape[0]} rows × {shape[1]} cols)"
        )

        with st.sidebar.expander(" Dataset Summary Statistics", expanded=False):
            schema_display = schema_df.set_index("Column")
            st.dataframe(schema_display, use_container_width=True)
    else:
        st.sidebar.warning("No active `df` found in kernel.")

    # ------------------------------------------------------------------------ #
    # FEATURE 2: Active Kernel Variables Inspection                            #
    # ------------------------------------------------------------------------ #
    with st.sidebar.expander("Current Kernel Variables", expanded=False):
        if st.session_state["kernel"]:
            manifest = st.session_state["kernel"].get_kernel_memory_manifest()
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

    # ------------------------------------------------------------------------ #
    # FEATURE 3: Chat Controls & Jupyter Notebook Download                     #
    # ------------------------------------------------------------------------ #
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session Controls")

    # Refresh Questions Button (Triggers kernel inspection & LLM call)
    if st.sidebar.button("Regenerate Suggestions", use_container_width=True):
        with st.sidebar:
            with st.spinner("Analyzing active kernel memory..."):
                st.session_state["questions"] = get_questions()
        st.sidebar.success("Updated suggestions based on active kernel state!")
        st.rerun()

    # Clear Chat History Button
    if st.sidebar.button(" Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.sidebar.success("Chat history cleared!")
        st.rerun()

    # Download Generated Code as Jupyter Notebook (.ipynb)
    if st.session_state.messages:
        nb_json = generate_jupyter_notebook()
        st.sidebar.download_button(
            label="Download Notebook (.ipynb)",
            data=nb_json,
            file_name=f"analysis_{uuid.uuid4().hex[:6]}.ipynb",
            mime="application/x-ipynb+json",
            use_container_width=True,
        )

    # ------------------------------------------------------------------------ #
    # Main Chat Interface                                                      #
    # ------------------------------------------------------------------------ #
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.expander("View Executed Code"):
                    st.code(message["content"], language="python")
                execute_and_render(message["content"])

    st.divider()
    render_buttons()

    # Capture chat input
    chat_box_input = st.chat_input(
        "Ask a question or type / code to execute directly..."
    )
    if chat_box_input:
        st.session_state.user_input = chat_box_input

    # Process pending input
    if st.session_state.user_input:
        prompt = st.session_state.user_input
        st.session_state.user_input = None

        # Render User Message
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        # Slash Command Logic (Direct Jupyter Kernel Execution)
        if prompt.strip().startswith("/"):
            raw_code = extract_command(prompt)
            formatted_response = f"```python\n{raw_code}\n```"

            with st.sidebar:
                st.spinner("⚡ Executing direct command in Jupyter Kernel...")

            with st.chat_message("assistant"):
                st.caption("⚡ *Executed directly in Jupyter Kernel (LLM Bypassed)*")
                with st.expander("View Executed Code"):
                    st.code(raw_code, language="python")

            st.session_state.messages.append(
                {"role": "assistant", "content": formatted_response}
            )

            execute_and_render(formatted_response)

        # Natural Language Query (Groq LLM Chain)
        else:
            with st.sidebar:
                sidebar_status = st.status("Zeno is thinking..", expanded=True)

            with st.chat_message("assistant"):
                stream = get_answer(prompt)
                full_response = st.write_stream(stream)

            sidebar_status.update(
                label="⚡ Code Executing in Kernel...", state="running"
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

            execute_and_render(full_response)
            sidebar_status.update(
                label="✓ Execution Complete", state="complete", expanded=False
            )

        # Shuffle sample questions
        if st.session_state["questions"]:
            rd.shuffle(st.session_state["questions"])

        st.rerun()

else:
    # Main Page File Upload & Setup Form
    st.markdown("""
    # Hmmmmm....

    Ask **natural language questions** or execute **direct python commands** with full conversation memory inside a live Jupyter Kernel!
    """)

    st.sidebar.info("Waiting for data file upload...")

    # Main Page Form for File Upload
    with st.form("Start"):
        st.subheader("Upload Dataset to Begin")
        file = st.file_uploader("Upload CSV data file", type=["csv"])
        submit = st.form_submit_button("Start Engine")

        if submit:
            if file is not None:
                st.session_state["file_name"] = file.name
                st.session_state["df"] = pd.read_csv(file)
                st.session_state["questions"] = None

                if (
                    st.session_state["kernel"] is not None
                    and hasattr(st.session_state["kernel"], "stop")
                    and callable(st.session_state["kernel"].stop)
                ):
                    st.session_state["kernel"].stop()

                st.session_state["kernel"] = None
                st.rerun()
            else:
                st.error("Please select a valid CSV file before submitting.")
