import pandas as pd
from typing import List
import time
from tqdm import tqdm

from RAG_mapper.src.map_evaluator import RAGMapperEvaluator


class SchemaBuilder(RAGMapperEvaluator):

    def __init__(self, var_list: List[str], k: int = 10, t: float = 0.75) -> None:

        super().__init__(var_list=var_list, k=k, t=t)

    def update_schema(self, query:str) -> None:
        res = self.evaluate(query)
        idx_to_update = self.local_schema.index[self.local_schema['variable'] == query].tolist()
        if idx_to_update:
            idx = idx_to_update[0]

            self.local_schema.loc[idx, 'ontology_id'] = res['ontology_id'].iloc[0]
            self.local_schema.loc[idx, 'ontology_name'] = res['ontology_name'].iloc[0]
            self.local_schema.loc[idx, 'confidence'] = res['confidence'].iloc[0]

            print(f"Successfully mapped and updated schema for variable: '{query}'")
        else:
            print(f"Warning: Variable '{query}' not found in local_schema to update.")


    def create_schema(self) -> pd.DataFrame:
        start = time.time()
        print('Extract schema from list of variables... \n')
        for query in tqdm(self.var_list):
            print(f'Extracting schema for variable: {query}... \n')
            current_row = self.local_schema[self.local_schema['variable'] == query]
            if current_row['ontology_id'].iloc[0] is not None:
                print(f"Skipping '{query}'. Already mapped.")
                continue
            self.update_schema(query)
        print('Schema extraction complete.\n')
        print(f'Time elapsed: {time.time() - start} seconds.\n')

        return self.local_schema

    def save_schema(self, out_file: str = 'retrieved_schema.xlsx') -> None:
        self.local_schema.to_excel(out_file)
        print(f'Schema saved to {out_file}.')