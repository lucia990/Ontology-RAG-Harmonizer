import pandas as pd
import numpy as np
from typing import List
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
import time

from UMLS_mapper.src.umls_search_engine import get_umls_search_engine
from RAG_mapper.src.llm_model import CosyChatOllama, SchemaNode

class RAGMapper:

    def __init__(self, var_list: List[str], k: int = 10, t: float = 0.8) -> None:
        print(f'Initializing RAG mapper with parameters:\n - Num. nearest neighbors: {k}\n - Similarity threshold: {t}\n')
        self.var_list = var_list
        self.current_query = None
        self.umls_search_engine = get_umls_search_engine()
        self.k = k
        self.t = t
        self.local_schema = pd.DataFrame({'variable': self.var_list,
                                        'ontology_id': [None] * len(self.var_list),
                                        'ontology_name': [None] * len(self.var_list),
                                        'confidence': [None] * len(self.var_list),
                                        })
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
                return pd.DataFrame([{'variable': 'Empty input', 'ontology_id': 'No_Query', 'ontology_name': None, 'confidence': 0.0}], index= None)

            results_df = self.umls_search_engine.search(search_query, self.k)[['ontology_id', 'ontology_name', 'confidence']]

            if not results_df.empty:
                results_df['is_relevant'] = results_df['confidence'] > self.t

                filtered_df = results_df[results_df['is_relevant'] == True].reset_index(drop=True)
                filtered_df = filtered_df.copy()
                filtered_df['variable'] = [query_str] * len(filtered_df)
                if len(filtered_df) != 0:
                    return filtered_df[['variable', 'ontology_id', 'ontology_name', 'confidence']]
                else:
                    print(f'No significant match found for query {search_query}\n')
                    return pd.DataFrame([{'variable': search_query, 'ontology_id': 'Needs_Review', 'ontology_name': None,'confidence': 0.0}], index=None)
            else:
                print(f'No match found for query {search_query}\n')
                return pd.DataFrame([{'variable': search_query, 'ontology_id': 'Needs_Review', 'ontology_name': None, 'confidence': 0.0}], index = None)


        except Exception as e:
            print(f"Error mapping query ({query_str}): {e}")
            return pd.DataFrame([{'variable': query_str, 'ontology_id': 'Error',  'ontology_name': None, 'confidence': 0.0}], index = None)

    def create_mapper_chain(self):
        # prompt
        with open('RAG_mapper/prompts/system_instructions_mapper.md', 'r') as instructions_file:
            with open('RAG_mapper/prompts/human_prompt_mapper.md', 'r') as human_prompt_file:
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

    def format_schema(self) -> str:
        if self.local_schema.empty:
            return "Current schema is empty"
        valid_rows = self.local_schema.dropna(subset=['ontology_id', 'ontology_name'])
        valid_rows = valid_rows[
            (valid_rows['ontology_id'] != 'Needs_Review') &
            (valid_rows['ontology_id'] != 'Error') &
            (valid_rows['ontology_id'] != 'Skipped')
            ]

        if valid_rows.empty:
            return "No unsuccessful ontology mappings yet in the current schema."

        formatted_lines = [
            f"{i + 1}. variable: {row['variable']} | ontology_id: {row['ontology_id']} | ontology_name: {row['ontology_name']} | confidence: {row['confidence']:.2f}"
            for i, row in valid_rows.iterrows()
        ]

        header = "Previously mapped ontology nodes (DON'T DUPLICATE these):"
        return header + "\n" + "\n".join(formatted_lines)

    def format_candidates(self, candidates: pd.DataFrame) -> str:
        return "\n".join(f"{i+1}. ontology_id: {row['ontology_id']} | ontology_name: {row['ontology_name']} | confidence: {row['confidence']:.2f}" for i, row in candidates.iterrows())


    def RAG_map(self, query: str) -> pd.DataFrame:

        print('RAG mapper started... \n')
        res = pd.DataFrame()
        candidates = self.map_umls(query)
        print(f"list of candidates:\n{candidates}")
        candidates_text = 'No candidates found'
        if not candidates.empty:
            candidates_text = self.format_candidates(candidates)
        local_schema = self.format_schema()
        try:
            chain = self.create_mapper_chain()
            res = chain.invoke({'variable': query, 'current_schema': local_schema, 'candidates': candidates_text})
            print(f'Agent output: {res}\n')


        except Exception as e:
            print(f"Error invoking mapper agent for ({query}): {e}")

        return res














    def interactive_map_review(self) -> pd.DataFrame:
        """
        Iteratively prompts the user to input a candidate for 'Needs_Review' rows.
        """
        review_indices = self.local_schema[self.local_schema['ontology_id'] == 'Needs_Review'].index

        if review_indices.empty:
            return self.local_schema

        print(f"\n--- Starting interactive review for ({len(review_indices)} rows) ---")

        source_col_name = self.local_schema.columns[0]

        for idx in review_indices:
            original_query = str(self.local_schema.loc[idx, source_col_name]).strip()
            self.current_query = original_query
            current_name = self.local_schema.loc[idx, 'variable']

            print(f"\nRow Index {idx} (Original Query: '{original_query}')")
            print(f"Last Unsuccessful Query: '{current_name}'")

            while True:
                new_candidate = input('Suggest new search candidate (or type "SKIP" to keep as Needs_Review/Skipped): ')

                if new_candidate.upper() == 'SKIP':
                    self.local_schema.loc[idx, 'ontology_id'] = 'Skipped'
                    self.local_schema.loc[idx, 'ontology_name'] = f"Skipped by User (Variable name: {current_name})"
                    break

                self.RAG_map(new_candidate)

        return  self.local_schema






if __name__ == '__main__':
    try:
        import numpy as np
        import os
        def preprocess_var_series(row: pd.Series):
            """
            Generate the variable string from the given row
            """
            try:
                search_query = ' '.join(row.fillna('').apply(str))
                if not search_query.strip():
                    return np.nan
                else:
                    return search_query
            except Exception as e:
                print(f"Error processing row ({row}): {e}")
                return np.nan
        FILE = 'Benchmark_MicrobAIome.xlsx'

        # Retrieve data (do it outside the preprocess function because it is too dataframe specific)
        data = pd.read_excel(FILE)
        irish_var = data.iloc[:, 0:2]
        french_var = data.iloc[:, 2:4]

        # Preprocess
        A_irish = irish_var.apply(preprocess_var_series, axis = 1).dropna().to_list()
        A_french = french_var.apply(preprocess_var_series, axis = 1).dropna().to_list()
        os.environ["LANGSMITH_API_KEY"] = "REDACTED_LANGSMITH_KEY"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "RAG_mapper"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

        # Run Baseline
        local_irish_schema = baseline(A_irish)  # 7 unmapped rows
        local_french_schema = baseline(A_french) # 8 unmapped rows

        # Save results
        local_irish_schema.to_csv('local_irish_baseline.csv')
        local_french_schema.to_csv('local_french_baseline.csv')

        print("\n*** FINAL RESULTS SUMMARY ***")
        print(f"Number of irish variables : {len(A_irish)}\nNumber of french variables : {len(A_french)}\n")
        print(f"Mapped variables\n IRISH (N = {len(local_irish_schema)}): {local_irish_schema.head()}\nFRENCH: (N = {len(local_french_schema)}): {local_french_schema.head()}\n")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

