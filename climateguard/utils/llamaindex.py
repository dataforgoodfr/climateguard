from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel


def get_pydantic_program(llm, output_cls: BaseModel, prompt_template_str: str):
    return LLMTextCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(output_cls=output_cls),
        prompt_template_str=prompt_template_str,
        verbose=True,
        llm=llm
    )