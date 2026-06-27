import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional
from pydantic import BaseModel, Field
from RAG_mapper.src.llm_model import get_llm

class Translation(BaseModel):
    translation: Optional[str] = Field(None, description="Translation of the description of the variable")


def create_translator_chain(model_name: str = 'gpt-oss:latest'):
    # prompt
    with open('preprocessing/prompts/system_instruction_translator.md', 'r') as instructions_file:
        with open('preprocessing/prompts/human_prompt_translator.md', 'r') as human_prompt_file:
            prompt_template = ChatPromptTemplate([
                ('system', instructions_file.read()),
                ('user', human_prompt_file.read()),
            ])
    # llm
    llm = get_llm(ollama_model=model_name)

    # parser
    parser = PydanticOutputParser(pydantic_object= Translation)

    # chain
    return prompt_template | llm | parser

def translate_var(description: str, model_name: str = 'gpt-oss:latest'):
    "given a variable name, the function picks the ontology vocabulary"
    chain = create_translator_chain(model_name=model_name)
    print(f"Invoke filtering model for variable {description}...\n")
    if description:
        result = chain.invoke({'description': description})
        print(f"{model_name} output: {result}\n")
        return result.translation
    else:
        print(f"No description found")
        return ''