<div align="center">
  <img src="assets/logo.png" alt="alt text" height="200" />
  <h1>Zeno 2</h1>
</div>

<!-- Tech Stack & System Badges -->
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter_Kernel-Active_State-F37626?style=for-the-badge&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Groq API](https://img.shields.io/badge/Groq_API-OSS_120B-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Sinica](https://img.shields.io/badge/Profiling-Sinica-8E44AD?style=for-the-badge)](https://github.com/sarvamm/zeno-2)

<!-- Repository & Status Badges -->
[![License: NPOSL-3.0](https://img.shields.io/badge/License-NPOSL--3.0-blue.svg?style=for-the-badge)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/sarvamm/zeno-2?style=for-the-badge&color=yellow)](https://github.com/sarvamm/zeno-2/stargazers)

**Zeno 2** is an intelligent, conversational data analysis platform built with Streamlit, LangChain, and an isolated Jupyter Kernel background runtime. By bridging open-source 120B LLMs hosted on Groq with live stateful execution and automated data profiling, Zeno 2 allows users to explore datasets through natural language queries, direct Python commands, and interactive visual outputs.

---

## Key Features

* **Automated Data Profiling (`sinica`)**: Instant data summaries, schema detection, and automated data alert generation upon file upload using the custom `sinica` library.
* **Stateful Jupyter Kernel Runtime**: Background kernel managed via `jupyter_client` maintains full in-memory execution context (DataFrames, variables, plots, and models) across conversation turns.
* **Context-Aware LLM Agent**: Powered by OSS 120B models via Groq API and LangChain. The agent inspects kernel variable memory manifests and schema metadata to write precise, context-accurate executable code.
* **Interactive Chat Interface**: Renders streamed code responses, execution logs, formatted tables, and interactive Plotly visualizations directly inside Streamlit.
* **Direct Kernel Commands (`/` Slash Mode)**: Bypass the LLM to execute raw Python code directly inside the active Jupyter kernel space (e.g., `/ df.describe()`).
* **Dynamic Question Recommendations**: Schema-aware analysis prompts automatically generated upon dataset initialization and shuffled dynamically to guide user exploration.
* **Interactive & Exportable Visualizations**: High-contrast, interactive Plotly charts natively supporting panning, zooming, and direct image download.
* **Full Notebook Export**: Export the entire analysis session, executed code blocks, and output history directly into a standard `.ipynb` Jupyter Notebook.

---

## Architecture Overview

```mermaid
flowchart TD
    %% Nodes
    UI["💻 Streamlit UI"]
    LLM["🧠 LLM Agent (Groq / OSS 120B)"]
    Kernel["⚙️ StreamlitJupyterKernel (Isolated Process)"]
    Sinica[" Sinica (Data Profiling)"]

    %% Connections
    UI -- "1. Uploads CSV Dataset" --> Kernel
    Kernel -- "2. Auto-Profiles Data" --> Sinica
    Sinica -. "3. Memory Manifest & Schema" .-> LLM
    UI -- "4. Natural Language Query" --> LLM
    LLM -- "5. Generates Executable Code" --> Kernel
    UI -- "Direct /Slash Command" --> Kernel
    Kernel -- "6. Streams Outputs, Tables, & Plotly Charts" --> UI

    %% Styling
    classDef primary fill:#1E88E5,stroke:#022b4d,stroke-width:2px,color:#fff
    classDef ai fill:#43A047,stroke:#113a13,stroke-width:2px,color:#fff
    classDef engine fill:#E53935,stroke:#5c100e,stroke-width:2px,color:#fff
    classDef data fill:#FB8C00,stroke:#663800,stroke-width:2px,color:#fff

    class UI primary
    class LLM ai
    class Kernel engine
    class Sinica data
```

---


When a CSV dataset is uploaded:

1. **Kernel Initialization**: A background Python environment (`StreamlitJupyterKernel`) is spawned using `ipykernel`.
2. **Data Injection & Profiling**: The DataFrame is saved to disk and loaded into kernel memory. `sinica.ProfileReport` executes automatically to construct data summary tables and alert lists.
3. **Question Generation**: The schema and initial data summary are passed to the Groq API (OSS 120B model) to output tailored exploratory questions.
4. **Execution & Feedback**: User queries generate code blocks that run inside the kernel. Standard output, errors, and Plotly graphics are captured via `iopub` channels and rendered in Streamlit.

---

## Project Structure

```
zeno-2/
├── app.py                      # Main Streamlit application entry point
├── components/
│   ├── chat_ui.py              # Chat rendering engine and output handlers
│   └── sidebar.py              # Data overview, alerts, and export options
├── services/
│   ├── kernel_manager.py       # StreamlitJupyterKernel manager (ipykernel bridge)
│   └── llm_agent.py            # LangChain & Groq API agent logic
├── utils/
│   └── formatters.py           # Command extractors and prompt history formatters
├── assets/
│   └── chat_intro.png          # UI banner assets
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation

```

---

## Tech Stack

| Component | Technology |
| --- | --- |
| **Frontend UI** | [Streamlit](https://streamlit.io/?utm_source=gemini) |
| **Kernel Manager** | `jupyter_client`, `ipykernel` |
| **Data Profiling** | `sinica` (Custom Profiling Library), `pandas` |
| **LLM Provider** | [Groq API](https://groq.com/?utm_source=gemini) (OSS 120B Model) |
| **Agent Framework** | [LangChain](https://www.langchain.com/?utm_source=gemini) |
| **Visualization** | [Plotly Express & Graph Objects](https://plotly.com/python/?utm_source=gemini) |

---

## Getting Started

### Prerequisites

* **Python 3.10+** installed.
* A **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/?utm_source=gemini)).

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/sarvamm/zeno-2.git
cd zeno-2

```


2. **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt
pip install sinica

```


4. **Register the Jupyter Kernel**
Ensure `ipykernel` is installed in your active environment:
```bash
python -m ipykernel install --user --name streamlit_env_3_10 --display-name "Streamlit Environment Python"

```



---

## Environment Configuration

Create a `.env` file or configure Streamlit secrets (`.streamlit/secrets.toml`) with your Groq API Key:

```env
GROQ_API_KEY="your_groq_api_key_here"

```

Alternatively, set it via Streamlit secrets (`.streamlit/secrets.toml`):

```toml
API = "your_groq_api_key_here"

```

---

## Running the Application

Launch the Streamlit app:

```bash
streamlit run app.py

```

1. Open your browser at `http://localhost:8501`.
2. Upload any tabular dataset (`.csv`).
3. View automated alerts and summary metrics in the sidebar.
4. Begin asking questions in natural language, selecting generated recommendations, or running direct commands using `/ <python_code>`.

---

## Usage Guide

### Natural Language Queries

Type questions like:

* *"What is the correlation between numerical columns?"*
* *"Build a scatter plot of feature A vs feature B colored by target."*
* *"Identify missing values and clean them."*

### Direct Slash Commands (`/`)

Bypass the LLM to execute Python code instantly in the active Jupyter Kernel:

* `/ df.head(10)`
* `/ df.info()`
* `/ plt.hist(df['column_name'])`

### Exporting Session

At any point during or after analysis, click **Download Jupyter Notebook** in the sidebar to retrieve an `.ipynb` file containing all generated code snippets and execution history.