import json

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from config.prompts import QUESTION_GEN_PROMPT_TEMPLATE, TASK_PROMPT_TEMPLATE
from services.kernel_manager import StreamlitJupyterKernel, get_active_dataframe_info


class ListFormatter(BaseModel):
    questions: list[str] = Field(
        description="List of 10 actionable data analysis questions"
    )


def get_llm(api_key: str):
    return ChatGroq(
        groq_api_key=api_key,  # type: ignore
        model_name="openai/gpt-oss-120b",  # type: ignore
        temperature=0.3,
    )


def get_answer(user_prompt: str, kernel_manifest: dict, history_str: str, api_key: str):
    llm = get_llm(api_key)
    task_prompt = PromptTemplate.from_template(TASK_PROMPT_TEMPLATE)
    task_chain = task_prompt | llm | StrOutputParser()

    return task_chain.stream(
        {
            "kernel_manifest": json.dumps(kernel_manifest, indent=2),
            "history": history_str,
            "user_prompt": user_prompt,
        }
    )


def generate_questions(kernel: StreamlitJupyterKernel, api_key: str) -> list[str]:
    llm = get_llm(api_key)
    schema_df = get_active_dataframe_info(kernel)[1]

    if schema_df is None or schema_df.empty:
        cols_summary_str = "No active DataFrame found."
    else:
        cols_summary_str = json.dumps(
            schema_df.to_dict(orient="records"), indent=2, ensure_ascii=False
        )

    # Use JsonOutputParser bound to ListFormatter schema
    parser = JsonOutputParser(pydantic_object=ListFormatter)

    prompt = PromptTemplate(
        template=QUESTION_GEN_PROMPT_TEMPLATE
        + "\n\nReturn a JSON object containing a 'questions' key:\n{format_instructions}",
        input_variables=["cols_summary_str"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    try:
        result = chain.invoke({"cols_summary_str": cols_summary_str})

        if isinstance(result, dict):
            return result.get("questions", [])
        return getattr(result, "questions", [])

    except Exception as e:
        safe_e = str(e).encode("utf-8", errors="backslashreplace").decode("utf-8")
        print(f"LLM Generation Error: {safe_e}")
        return [
            f"LLM Generation Error: {safe_e}.",
            "Plot distributions of key categorical variables.",
            "Check for correlations across the dataset.",
        ]
