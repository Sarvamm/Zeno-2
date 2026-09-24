import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
import pandas as pd
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


class StreamlitJupyterKernel:
    """Manages an active ipykernel session bound directly to the current runtime environment."""

    def __init__(self):
        self.kernel_name = self._ensure_env_kernelspec()
        self.km = KernelManager(kernel_name=self.kernel_name)
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=15)

    def _ensure_env_kernelspec(self) -> str:
        ksm = KernelSpecManager()
        kernel_name = f"streamlit_env_{sys.version_info.major}_{sys.version_info.minor}"
        if kernel_name not in ksm.get_all_specs():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "ipykernel",
                    "install",
                    "--user",
                    "--name",
                    kernel_name,
                    "--display-name",
                    "Streamlit Environment Python",
                ],
                check=True,
            )
        return kernel_name

    def execute_code(self, code: str, timeout: int = 20) -> list[dict]:
        msg_id = self.kc.execute(code)
        outputs = []
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=timeout)
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
            elif msg_type == "status" and content.get("execution_state") == "idle":
                break
        return outputs

    def inject_dataframe(
        self, df: pd.DataFrame, var_name: str = "df"
    ) -> tuple[bool, str | None]:
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
pio.renderers.default = 'notebook_connected'
{var_name} = pd.read_csv('{clean_path}')
"""
        outputs = self.execute_code(init_code, timeout=30)
        for out in outputs:
            if out.get("type") == "error":
                clean_tb = "\n".join(
                    [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in out["traceback"]]
                )
                return False, clean_tb
        return True, None

    def get_kernel_memory_manifest(self) -> dict:
        inspection_code = """
import json
import pandas as pd

_vars_summary = {}
_ignore_keys = {'In', 'Out', 'exit', 'quit', 'get_ipython', 'pd', 'px', 'go', 'pio', 'json', 'np', 'plt'}

for _k, _v in list(globals().items()):
    if not _k.startswith('_') and _k not in _ignore_keys:
        if isinstance(_v, pd.DataFrame):
            _vars_summary[_k] = {
                "type": "DataFrame", "shape": list(_v.shape), 
                "columns": _v.columns.tolist(), "dtypes": {col: str(dtype) for col, dtype in _v.dtypes.items()}
            }
        elif isinstance(_v, pd.Series):
            _vars_summary[_k] = {"type": "Series", "shape": list(_v.shape), "dtype": str(_v.dtype)}
        elif type(_v).__name__ in ('int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set'):
            val_str = str(_v)
            _vars_summary[_k] = {"type": type(_v).__name__, "value": val_str[:100] + "..." if len(val_str) > 100 else val_str}
print("MANIFEST_START" + json.dumps(_vars_summary) + "MANIFEST_END")
"""
        outputs = self.execute_code(inspection_code)
        for out in outputs:
            if out["type"] == "stream" and "MANIFEST_START" in out["text"]:
                try:
                    match = re.search(
                        r"MANIFEST_START(.*?)MANIFEST_END", out["text"], re.DOTALL
                    )
                    if match:
                        return json.loads(match.group(1).strip())
                except Exception:
                    break
        return {}

    def stop(self):
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


def get_active_dataframe_info(
    kernel: StreamlitJupyterKernel,
) -> tuple[list[int] | None, pd.DataFrame | None]:
    if not kernel:
        return None, None
    inspection_code = """
import json
import pandas as pd

if 'df' in globals() and isinstance(df, pd.DataFrame):
    _info = []
    for col in df.columns:
        s = df[col]
        entry = {"Column": str(col), "Data Type": str(s.dtype), "Missing": int(s.isnull().sum()), "Unique": int(s.nunique())}
        clean_s = s.dropna()
        if pd.api.types.is_numeric_dtype(s):
            entry["Mean"] = round(float(clean_s.mean()), 4) if not clean_s.empty else None
            entry["Median"] = round(float(clean_s.median()), 4) if not clean_s.empty else None
            entry["Mode"] = round(float(clean_s.mode().iloc[0]), 4) if not clean_s.mode().empty else None
            entry["Min"] = round(float(clean_s.min()), 4) if not clean_s.empty else None
            entry["Max"] = round(float(clean_s.max()), 4) if not clean_s.empty else None
        else:
            entry["Mean"] = "-"
            entry["Median"] = "-"
            entry["Mode"] = str(clean_s.mode().iloc[0]) if not clean_s.mode().empty else "-"
            entry["Min"] = str(clean_s.min()) if not clean_s.empty else "-"
            entry["Max"] = str(clean_s.max()) if not clean_s.empty else "-"
        _info.append(entry)
    print("COL_INFO_START" + json.dumps({"shape": list(df.shape), "columns": _info}) + "COL_INFO_END")
"""
    outputs = kernel.execute_code(inspection_code)
    for out in outputs:
        if out["type"] == "stream" and "COL_INFO_START" in out["text"]:
            try:
                match = re.search(
                    r"COL_INFO_START(.*?)COL_INFO_END", out["text"], re.DOTALL
                )
                if match:
                    data = json.loads(match.group(1).strip())
                    return data["shape"], pd.DataFrame(data["columns"])
            except Exception:
                break
    return None, None
