import pandas as pd
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import time
from RAG_mapper.src.llm_model import CosyChatOllama, SchemaNode
from RAG_mapper.src.RAG_mapper import RAGMapper



class RAGMapperEvaluator(RAGMapper):

    def __init__(self, var_list: List[str], k: int = 10, t: float = 0.75) -> None:

        super().__init__(var_list=var_list, k=k, t=t)

    def create_evaluator_chain(self):
        with open('RAG_mapper/prompts/system_instructions_evaluator.md', 'r') as instructions_file:
            with open('RAG_mapper/prompts/human_prompt_evaluator.md', 'r') as human_prompt_file:
                prompt_template = ChatPromptTemplate([
                    ('system', instructions_file.read()),
                    ('user', human_prompt_file.read()),
                ])
        # llm
        llm = CosyChatOllama()

        # parser
        parser = PydanticOutputParser(pydantic_object= SchemaNode)
        def to_dataframe(schema_node: SchemaNode) -> pd.DataFrame:
            """Convert SchemaNode to single-row DataFrame."""
            return pd.DataFrame([schema_node.model_dump()])
        # chain
        return prompt_template | llm | parser | to_dataframe


    def evaluate(self, query: str) -> pd.DataFrame:

        print(f"Starting evaluation for query: {query}\n")
        res = pd.DataFrame()
        rag_res = self.RAG_map(query)
        candidates = self.map_umls(query)
        candidates_text = ''
        if not candidates.empty:
            candidates_text = "\n".join(
                f"{i + 1}. ontology_id: {row['ontology_id']} | ontology_name: {row['ontology_name']} | confidence: {row['confidence']:.2f}" for i, row in candidates.iterrows())
        try:
            chain = self.create_evaluator_chain()
            res = chain.invoke({'variable': query, 'rag_result': rag_res, 'candidates': candidates_text})
            print(f'Agent output: {res}\n')
        except Exception as e:
            print(f"Error invoking evaluator agent for ({query}): {e}")

        return res



