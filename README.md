# Zeno 2

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A Streamlit application that automates data analysis by translating natural language into executable Python code. The application maintains an active Jupyter kernel in the background to preserve session state, manage dataset memory, and render interactive visualizations natively within the UI.

## Core Features

* **Natural Language to Code:** Utilizes a Groq-powered LLM to interpret user queries and generate pandas and Plotly data analysis code.
* **Persistent Jupyter Kernel:** Runs a live `ipykernel` session directly bound to the runtime environment. Variables, DataFrames, and library imports persist across all interactions.
* **Direct Command Execution:** Supports manual Python execution, bypassing the LLM entirely, by prefixing inputs with a forward slash (`/`).
* **Interactive Visualizations:** Automatically handles and renders Plotly graphs (defaulting to the `plotly_dark` theme), HTML components, standard PNG images, and text streams.
* **Jupyter Notebook Export:** Compiles the entire session history—user queries as Markdown cells and executed responses as code cells—into a downloadable `.ipynb` file.
* **Context-Aware Suggestions:** Inspects the active DataFrame schema (dimensions, data types, missing values, and descriptive statistics) to generate dynamic, schema-aware analysis questions.

## Architecture & Implementation

### 1. Jupyter Kernel Integration (`StreamlitJupyterKernel`)
The application bypasses standard `exec()` or `eval()` calls. Instead, it uses `jupyter_client.KernelManager` to initialize a dedicated Jupyter kernel tied to the active Python environment.
* **Data Injection:** When a CSV is uploaded, it is saved to a temporary directory and injected into the global kernel memory as a pandas DataFrame (`df`). Common libraries (`pandas`, `numpy`, `matplotlib.pyplot`, `plotly.express`, `plotly.graph_objects`, `plotly.io`) are pre-imported.
* **Kernel Memory Manifest:** The application inspects the global variables inside the kernel (ignoring standard libraries) to provide the LLM with a real-time JSON manifest of active DataFrames, Series, and basic variable types.
* **Execution Polling:** The `execute_code` method communicates with the kernel via ZMQ channels, polling the `iopub` channel to capture streams, display data, execution results, and tracebacks.

### 2. LLM Integration
* **Model:** Uses `ChatGroq` via LangChain, specifically targeting the `qwen/qwen3.8-27b` model with a low temperature (0.3) for deterministic, reliable code generation.
* **Contextual Prompting:** The prompt provides the LLM with the active Jupyter Kernel Memory State and the last six messages of execution history. It is explicitly instructed to reuse existing variables and return only executable Python code formatted in markdown blocks.
* **Structured Outputs:** Uses Pydantic (`ListFormatter`) to force the LLM to generate exactly 10 short, actionable analytical questions based on the active dataset's shape and column summary.

### 3. Rendering Pipeline
When code is executed, the backend parses the Jupyter message types and maps them to Streamlit components:
* `stream`: Rendered as plain text.
* `application/vnd.plotly.v1+json`: Rendered as native, interactive Streamlit Plotly charts.
* `text/html`: Rendered via `components.html` or `st.markdown` (depending on whether it contains a Plotly div).
* `image/png`: Rendered natively via `st.image`.
* `error`: Extracts the traceback, strips ANSI escape codes, and displays it via `st.error`.

## Interface Layout

* **Initialization Screen:** Prompts the user to upload a CSV file. The engine will not start until data is provided.
* **Sidebar Controls:**
  * **Dataset Summary:** Displays the active file name, dimensions, and an expandable table of column-level statistics (Data Type, Missing, Unique, Mean, Median, Mode, Min, Max) calculated directly inside the kernel.
  * **Current Kernel Variables:** Lists custom variables currently held in the kernel's memory, showing their types and shapes.
  * **Action Buttons:** Includes options to Regenerate Suggestions, Clear Chat History, and Download Notebook (.ipynb).
* **Main Chat Interface:**
  * Displays the conversational history.
  * Executed code blocks are tucked inside collapsible expanders.
  * Visual and textual outputs are rendered directly beneath the code.
  * Three dynamic quick-action buttons sit above the text input to jump-start the analysis.

## Configuration & Requirements

* **API Key:** Requires a Groq API key to function. The application looks for this key in Streamlit secrets (`st.secrets["API"]`) or the system environment variable `GROQ_API_KEY`.
* **Core Dependencies:** `streamlit`, `pandas`, `jupyter_client`, `langchain_core`, `langchain_groq`, `pydantic`, `plotly`.
