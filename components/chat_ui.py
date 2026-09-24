import re
import streamlit as st
import streamlit.components.v1 as components
from services.kernel_manager import StreamlitJupyterKernel


def execute_and_render(code_response: str, kernel: StreamlitJupyterKernel):
    """Executes Python code block in kernel and renders output objects visually."""
    match = re.search(r"```python\s*\n(.*?)```", code_response, re.DOTALL)
    code = match.group(1) if match else code_response.strip()

    if not kernel:
        st.error("Jupyter kernel is not running.")
        return

    outputs = kernel.execute_code(code)

    for out in outputs:
        if out["type"] == "stream":
            st.text(out["text"])

        elif out["type"] == "display_data":
            data = out["data"]
            if "application/vnd.plotly.v1+json" in data:
                fig_dict = data["application/vnd.plotly.v1+json"]
                st.plotly_chart(fig_dict, use_container_width=True)
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


def render_buttons():
    """Renders dynamic quick query buttons."""
    if st.session_state.get("questions"):
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
