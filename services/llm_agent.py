import json
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config.prompts import TASK_PROMPT_TEMPLATE, QUESTION_GEN_PROMPT_TEMPLATE
from services.kernel_manager import get_active_dataframe_info, StreamlitJupyterKernel


class ListFormatter(BaseModel):
    questions: list[str] = Field(
        description="List of 10 actionable data analysis questions"
    )


def get_llm(api_key: str):
    return ChatGroq(
        groq_api_key=api_key,
        model_name="qwen/qwen3.8-27b",
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
    shape, schema_df = get_active_dataframe_info(kernel)

    if schema_df is None or schema_df.empty:
        cols_summary_str = "No active DataFrame found."
        shape_str = "Unknown"
    else:
        shape_str = f"{shape[0]} rows × {shape[1]} columns"
        cols_summary_str = json.dumps(schema_df.to_dict(orient="records"), indent=2)

    prompt = PromptTemplate.from_template(QUESTION_GEN_PROMPT_TEMPLATE)
    chain = prompt | llm.with_structured_output(ListFormatter)

    try:
        result: ListFormatter = chain.invoke(
            {
                "shape_str": shape_str,
                "cols_summary_str": cols_summary_str,
            }
        )
        return result.questions
    except Exception as e:
        print(f"LLM Generation Error: {e}")
        return [
            "Show summary statistics for numerical columns.",
            "Plot distributions of key categorical variables.",
            "Check for correlations across the dataset.",
        ]
