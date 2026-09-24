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
