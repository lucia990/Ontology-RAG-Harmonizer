import pandas as pd
import ast

def read_mapping_table(mapping_file):
    df = pd.read_csv(f"Evaluation/OntoMapping_benchmark/mapped_codes/{mapping_file}", index_col=0)
    list_cols = [col for col in df.columns if df[col].dtype == object]
    df[list_cols] = df[list_cols].map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df


