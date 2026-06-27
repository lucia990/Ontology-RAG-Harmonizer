import pandas as pd
from typing import List
import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from UMLS_mapper.src.umls_search_engine import get_umls_search_engine
from RAG_mapper.src.llm_model import SupervisorOutput, OllamaWrapper, RankedCandidates


class RAGMapper:

    def __init__(self, var_list: List[str], var_desc: List[str],  FAISS_INDEX_PATH:str = "UMLS_mapper/data/processed/faiss_index_25.bin", METADATA_PATH: str = "UMLS_mapper/data/processed/metadata_25.csv", k: int = 5, t: float = 0.8) -> None:
        print(f'Initializing RAG mapper with parameters:\n - Num. nearest neighbors: {k}\n - Similarity threshold: {t}\n')
        self.var_list = var_list
        self.var_desc = var_desc
        self.current_query = None
        self.umls_search_engine = get_umls_search_engine(FAISS_INDEX_PATH, METADATA_PATH)
        self.k = k
        self.t = t
        '''
        self.local_schema = pd.DataFrame({'variable': self.var_list,
                                        'AI_code': [None] * len(self.var_list),
                                        'ontology_name': [None] * len(self.var_list),
                                        'confidence': [None] * len(self.var_list),
                                        #'source': [None] * len(self.var_list)
                                        })
        '''
        print('RAG mapper initialized.')

    def map_umls(self, query_str: str) -> pd.DataFrame:
        """
        Maps a single string query to filtered set of candidate ontology nodes.
        Returns 'Needs_Review' if similarity is low for every candidate or 'Error' on failure.
        """

        try:
            search_query = query_str.strip()
            self.current_query = query_str

            if not isinstance(query_str, str) or not query_str.strip():
                return pd.DataFrame([{'variable': 'Empty input', 'ontology_name': None, 'confidence': 0.0, 'CUI': None}], index= None)

            results_df = self.umls_search_engine.search(search_query, self.k)
            results_df['confidence'] = results_df['confidence'].apply(lambda x: round(x, 2))
            # results_df = filter_umls(search_query, results_df)
            if not results_df.empty:
                print(f"Search engine OUTPUT:\n{results_df}")
                results_df['is_relevant'] = results_df['confidence'] > self.t

                filtered_df = results_df[results_df['is_relevant'] == True].reset_index(drop=True)
                filtered_df = filtered_df.copy()
                filtered_df['variable'] = [query_str] * len(filtered_df)
                if len(filtered_df) != 0:
                    return filtered_df[['variable', 'ontology_name', 'confidence', 'CUI', 'semantic_type']] # UMLS_release 'ontology_name', 'confidence', 'CUI', 'source', 'se_CODE'
                else:
                    print(f'No significant match found for query {search_query}\n')
                    return pd.DataFrame([{'variable': search_query, 'ontology_name': None,'confidence': 0.0, 'CUI': None}], index=None)
            else:
                print(f'No match found for query {search_query}\n')
                return pd.DataFrame([{'variable': search_query, 'ontology_name': None, 'confidence': 0.0, 'CUI': None}], index = None)


        except Exception as e:
            print(f"Error mapping query ({query_str}): {e}")
            return pd.DataFrame([{'variable': query_str,  'ontology_name': None, 'confidence': 0.0, 'CUI': None}], index = None)

    def create_mapper_chain(self, llm_model='granite4:latest'):

        with open('RAG_mapper/prompts/system_instructions_mapper-2.md') as f:
            system_prompt = f.read()

        with open('RAG_mapper/prompts/human_prompt_mapper.md') as f:
            human_template = f.read()

        parser = PydanticOutputParser(pydantic_object=RankedCandidates)

        ollama = OllamaWrapper(timeout=80)

        def run(variable, candidates, var_desc):
            sys_prompt = system_prompt.format(
                variable=variable,
                candidates=candidates,
                var_desc=var_desc,
            )
            prompt = human_template.format(
                variable=variable,
                candidates=candidates,
                var_desc = var_desc
            )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt},
            ]

            response = ollama.chat(model=llm_model, messages=messages)

            text = response.message.content

            # Strip markdown fences if model wraps output
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            # Parse into SchemaNode
            parsed = parser.parse(text)

            rows = []
            for candidate in parsed.candidates or []:
                rows.append({
                    "variable": parsed.variable,
                    "rank": candidate.rank,
                    "AI_code": candidate.AI_code,
                    "AI_name": candidate.AI_name,
                    "confidence": candidate.confidence,
                    "CUI": candidate.CUI
                })

            # If parsing yielded nothing, return a single Needs_Review row
            if not rows:
                rows.append({
                    "variable": parsed.variable,
                    "rank": 1,
                    "AI_code": "Needs_Review",
                    "AI_name": None,
                    "confidence": 0.0,
                    "CUI": None
                })

            return pd.DataFrame(rows)

        return run

    def format_schema(self) -> str:
        if self.local_schema.empty:
            return "Current schema is empty"
        valid_rows = self.local_schema.dropna(subset=['se_CODE', 'ontology_name'])
        valid_rows = valid_rows[
            (valid_rows['CUI'] != 'Needs_Review') &
            (valid_rows['CUI'] != 'Error') &
            (valid_rows['CUI'] != 'Skipped')
            ]

        if valid_rows.empty:
            return "No unsuccessful ontology mappings yet in the current schema."

        formatted_lines = [
            f"{i + 1}. variable: {row['variable']} | AI_code: {row['AI_code']} | ontology_name: {row['ontology_name']} | confidence: {row['confidence']:.2f}"
            for i, row in valid_rows.iterrows()
        ]

        header = "Previously mapped ontology nodes (DON'T DUPLICATE these):"
        return header + "\n" + "\n".join(formatted_lines)

    def format_candidates(self, candidates: pd.DataFrame) -> str:
        return "\n".join(
            f"{i + 1}. ontology_name: {row['ontology_name']} | confidence: {row['confidence']:.2f} | CUI: {row['CUI']}"
            for i, row in candidates.iterrows()
        )

    def RAG_map(self, query: str, var_desc:str, llm_model:str = 'granite4:latest'):

        print('RAG mapper started... \n')
        res = pd.DataFrame()
        candidates = self.map_umls(query)
        print(f"list of candidates:\n{candidates}\n\n")
        candidates_text = 'No candidates found'
        if not candidates.empty:
            candidates_text = self.format_candidates(candidates)
            try:
                chain = self.create_mapper_chain(llm_model)
                res = chain(variable=query, var_desc = var_desc,  candidates=candidates_text)
                print(f'Agent output: {res}\n')

            except Exception as e:
                print(f"Error invoking mapper agent for ({query}): {e}")

            return candidates, res

        else:
            return candidates_text, 'no mapping available'
        #local_schema = self.format_schema()

    def create_evaluator_chain(self, llm_model):

        with open('RAG_mapper/prompts/system_instructions_evaluator-3.md') as f:
            system_prompt = f.read()

        with open('RAG_mapper/prompts/human_prompt_evaluator.md') as f:
            human_template = f.read()

        parser = PydanticOutputParser(pydantic_object=SupervisorOutput)
        ollama = OllamaWrapper(timeout=80)

        def run(variable, var_desc, rag_result, candidates):
            prompt = human_template.format(
                variable=variable,
                var_desc = var_desc,
                rag_result=rag_result,
                candidates=candidates
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            response = ollama.chat(model=llm_model, messages=messages)

            text = response.message.content
            parsed = parser.parse(text)

            return pd.DataFrame([parsed.model_dump()])

        return run

    def evaluate(self, query: str, var_desc:str, llm_model: str= 'gpt-oss:20b'):

        print(f"Starting evaluation for query: {query}\n")
        res = pd.DataFrame()
        candidates, rag_res = self.RAG_map(query, var_desc)
        candidates_text = ''
        if not candidates.empty:
            candidates_text = "\n".join(
                f"{i + 1}. ontology_name: {row['ontology_name']} | confidence: {row['confidence']:.2f} | CUI: {row['CUI']}" for i, row in candidates.iterrows())
        try:
            rag_res_text = "\n".join(
                f"{i + 1}. AI_code: {row['AI_code']} | AI_name: {row['AI_name']} | confidence: {round(row['confidence'], 2)} | CUI: {row['CUI']}"
                for i, row in rag_res.iterrows()
            )
            chain = self.create_evaluator_chain(llm_model)
            res = chain(variable=query, var_desc= var_desc, rag_result=rag_res_text, candidates=candidates_text)
            print(f'Evaluator agent output: {res}\n')
        except Exception as e:
            print(f"Error invoking evaluator agent for ({query}): {e}")

        return candidates, res

if __name__ == "__main__":
    rag_mapper = RAGMapper([''], [''], FAISS_INDEX_PATH = f"UMLS_mapper/data/processed/faiss_index_76.bin", METADATA_PATH= f"UMLS_mapper/data/processed/metadata_76.csv")
    test_query = input("Enter the query to run: ")
    var_desc = input("Enter the variable description: ")
    res = rag_mapper.evaluate(test_query, var_desc)
    print(res)