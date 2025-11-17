import pandas as pd
from typing import List, Optional
from RAG_mapper.src.RAG_mapper import RAGMapper
from RAG_mapper.src.schema_builder import SchemaBuilder



class InteractiveReview(RAGMapper):

    def __init__(self, var_list: List[str], k: int = 10, review_threshold: float = 0.7) -> None:

        super().__init__(var_list=var_list, k=k, t=review_threshold)

        self.review_queue = []
        self.review_log = []
        self.local_schema = None
        print(f'Interactive Review initialized with review threshold {self.t}')

    def load_schema(self, schema_builder: Optional[SchemaBuilder] = None) -> None:

        if schema_builder is None:
            print("No SchemaBuilder instance provided. Running full pipeline...")
            schema_builder = SchemaBuilder(self.var_list)
        else:
            print("Loading existing schema from provided SchemaBuilder instance...")

        self.local_schema = schema_builder.local_schema.copy()

        mask = self.local_schema['ontology_id'].isin(['Needs_Review', 'Error'])
        self.review_queue = self.local_schema[mask]['variable'].tolist()

        self.local_schema = schema_builder.create_schema().copy()
        mask = self.local_schema['ontology_id'].isin(['Needs_Review', 'Error'])
        self.review_queue = self.local_schema[mask]['variable'].tolist()
        print(f'Loaded schema for review. Found {len(self.review_queue)} variables needing review.')

    def filter_candidates(self, variable: str) -> pd.DataFrame:
        candidates = self.map_umls(variable)
        # candidates_text = self.format_candidates(candidates)
        return candidates

    def interactive_query(self, variable: str, candidates) -> pd.Series:
        """
        Present candidates for manual or automated review.
        """
        print(f"\nVariable: {variable}")
        print(f"Candidates:\n{candidates}")
        choice = input("Select best match [number] or press Enter to skip: ")


        if not choice:
            return pd.Series({'variable': variable, 'ontology_id': 'SKIPPED',  'ontology_name': None, 'confidence': 0.0})
        try:
            choice_idx = int(choice)
        except ValueError:
            print("Invalid input! Please enter a number or press Enter to skip.")
            return self.interactive_query(variable, candidates)  # retry

        if choice_idx < 0 or choice_idx >= len(candidates):
            print("Choice out of range! Retry.")
            return self.interactive_query(variable, candidates)  # retry


        return candidates.loc[int(choice), :]

    def update_schema(self, variable: str, choice: pd.Series) -> None:

        #row = choice.iloc[0]
        idx = self.local_schema[self.local_schema["variable"] == variable].index[0]
        self.local_schema.loc[idx, "ontology_id"] = choice["ontology_id"]
        self.local_schema.loc[idx, "ontology_name"] = choice["ontology_name"]
        self.local_schema.loc[idx, "confidence"] = round(float(choice['confidence']),2)
        self.review_log.append({
            "variable": variable,
            "selected_node":choice["ontology_name"],
            "confidence": round(float(choice['confidence']),2)
        })
        print(f"Updated schema entry for '{variable}'.")

    def run_review(self):
        for variable in self.review_queue:
            candidates = self.filter_candidates(variable)
            print(f"list of candidates: \n{candidates}")
            print('before')
            selected = self.interactive_query(variable, candidates)
            print('after')
            print(f"selected candidate: \n{selected}")
            if not selected.empty:
                self.update_schema(variable, selected)
        print("Review process completed.")
        return self.local_schema


# Example usage

if __name__ == '__main__':
    try:
        import numpy as np
        import os

        # PRE-PROCESSING STEP:
        # This is specific to the simplified (no description) MicrobAIome benchmark dataset:
        ## For categorical variables the feature (query for the system) is the base value. This means that for any categorical variable there are as many queries as allowed (categorical) value.
        def pick_feature(row: pd.Series) -> Optional[str]:
            if pd.isna(row.iloc[1]):
                return row.iloc[0]
            else:
                return row.iloc[1]

        def preprocess_var_series(row: pd.Series, add_description: bool =  True) -> pd.Series :
            try:
                cleaned = row.fillna('')
                search_query = pick_feature(row)
                description = cleaned.iloc[2]
                print(description)
                if search_query:

                    search_query = search_query + ' ' + description
                #search_query = ' '.join(row.fillna('').apply(str))
                    return search_query
                else:
                    return np.nan

            except Exception as e:
                print(f"Error processing row ({row}): {e}")
                return np.nan
        FILE = 'Benchmark_MicrobAIome.xlsx'

        # Retrieve data (do it outside the preprocess function because it is too dataframe specific)
        data = pd.read_excel(FILE)
        irish_var = data.iloc[:, 0:3].dropna(how = 'all')
        french_var = data.iloc[:, 3:6].dropna(how = 'all')

        '''
        os.environ["LANGSMITH_API_KEY"] = "REDACTED_LANGSMITH_KEY"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "RAG_mapper"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        '''
        # Preprocess
        A_irish = irish_var.apply(preprocess_var_series, axis=1).dropna().to_list()
        A_french = french_var.apply(preprocess_var_series, axis=1).dropna().to_list()


        # Create 'orchestrator' instance
        reviewer_irish = InteractiveReview(A_irish)
        reviewer_french = InteractiveReview(A_french)

        ####  End-to-end
        # Internally generate schema
        reviewer_irish.load_schema()
        reviewer_french.load_schema()

        # Run full mapping
        local_irish_schema = reviewer_irish.run_review()
        local_french_schema = reviewer_french.run_review()

        print("\n*** FINAL RESULTS SUMMARY (END-TO-END) ***\n")
        print(f"Number of irish variables : {len(A_irish)}\nNumber of french variables : {len(A_french)}\n")
        print(f"Mapped variables\n IRISH (N = {len(local_irish_schema)}): {local_irish_schema.head()}\nFRENCH: (N = {len(local_french_schema)}): {local_french_schema.head()}\n")


        #### MODULAR (existing non-reviewed schema)

        # Create SchemaBuilder instance
        irish_builder = SchemaBuilder(A_irish, t = 0.8)
        french_builder = SchemaBuilder(A_french, t = 0.8)

        # Generate the corresponding non reviewed schema
        irish_builder.create_schema()
        french_builder.create_schema()

        # Internally generate schema
        reviewer_irish.load_schema(irish_builder)
        reviewer_french.load_schema(french_builder)

        # Run the manual review
        irish_local_schema = reviewer_irish.run_review()
        french_local_schema = reviewer_french.run_review()

        # update schema
        irish_builder.local_schema = reviewer_irish.local_schema
        french_builder.local_schema = reviewer_french.local_schema

        # save local schema
        irish_builder.save_schema('results/irish_local_schema_256.xlsx')
        french_builder.save_schema('results/french_local_schema_256.xlsx')

        print("\n*** FINAL RESULTS SUMMARY (MODULAR) ***\n ***")
        print(f"Number of irish variables : {len(A_irish)}\nNumber of french variables : {len(A_french)}\n")
        print(f"Mapped variables\n IRISH (N = {len(local_irish_schema)}): {local_irish_schema.head()}\nFRENCH: (N = {len(local_french_schema)}): {local_french_schema.head()}\n")





    except Exception as e:
        print(f"An unexpected error occurred: {e}")