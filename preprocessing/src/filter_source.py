import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from typing import Optional
from pydantic import BaseModel, Field

from RAG_mapper.src.llm_model import OllamaWrapper
from Evaluation.OntoMapping_benchmark.src.OM_pipeline import setup_logging

logger = setup_logging()

class Onto(BaseModel):
    onto: Optional[str] = Field(None, description="The vocabulary source picked by the llm.")


def create_source_chain(llm_model='gpt-oss:20b'):

    with open('preprocessing/prompts/system_instruction_source-2.md') as f:
        system_prompt = f.read()

    with open('preprocessing/prompts/human_prompt_source.md') as f:
        human_template = f.read()

    parser = PydanticOutputParser(pydantic_object=Onto)

    ollama = OllamaWrapper(timeout=80)

    def run(variable, var_desc):
        prompt = human_template.format(
            variable=variable,
            var_desc= var_desc
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = ollama.chat(model=llm_model, messages=messages)

        text = response.message.content
        parsed = parser.parse(text)

        return parsed

    return run



def pick_source(query: str, var_desc: str, model_name: str = 'gpt-oss:latest'):
    "given a variable name, the function picks the ontology vocabulary"
    chain = create_source_chain(llm_model=model_name)
    print(f"Invoke filtering model for variable {query}...\n")
    result = chain(variable=query, var_desc = var_desc)
    print(f"{model_name} output: {result}\n")
    return result

def filter_candidates(query: str, var_desc: str, candidates: pd.DataFrame, model_name: str = 'gpt-oss:latest'):
    try:
        source = pick_source(query, var_desc, model_name=model_name)
        return candidates.loc[candidates['source'] == source]
    except Exception as e:
        print(f"No source found for {query}: {e}. \n Default vocabulary: SNOMEDCT")
        source = 'SNOMEDCT_US'
        return candidates.loc[candidates['source'] == source]

def pick_umls_source(query: str, var_desc: str, model_name: str = 'gpt-oss:20b'):
    try:
        source = pick_source(query,  var_desc = var_desc, model_name=model_name)
        source = source.onto
        return source
    except Exception as e:
        logger.error(f'Error picking the umls source vocabulary: {e}\n Default vocabulary: SNOMEDCT')
        return 'SNOMEDCT_US'





if __name__ == '__main__':

    chain = create_source_chain()
    var = 'Class'
    var_desc = 'binary classification label that indicates if the patient had a breast cancer recurrence event or not after n years. The time to event is not defined by the dataset, could be 10 years or something else'
    result = pick_umls_source(var, var_desc)
    print(result)
