import json
import re


def generate_jupyter_notebook(messages: list, file_name: str) -> str:
    """Converts conversation history to downloadable .ipynb format."""
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

    setup_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sinica import ProfileReport

pio.templates.default = "plotly_dark"
pio.renderers.default = 'notebook_connected'

df = pd.read_csv('{file_name}')

pr = ProfileReport(df)

summary_df = pr.summary() 
alerts_df = pr.alerts()   
"""
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in setup_code.split("\n")],
        }
    )

    for msg in messages:
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
