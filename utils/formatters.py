import re


def extract_command(input_text: str) -> str:
    """Strips leading slashes and extracts raw python blocks."""
    cmd = input_text.lstrip("/").strip()
    match = re.search(r"```(?:python)?\s*\n?(.*?)```", cmd, re.DOTALL)
    if match:
        return match.group(1).strip()
    return cmd


def format_history_for_prompt(messages: list) -> str:
    """Formats prior chat messages for the LLM prompt context."""
    if not messages:
        return "No prior conversation."

    history = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant (Executed Code)"
        history.append(f"{role}:\n{msg['content']}\n")

    return "\n".join(history[-6:])
