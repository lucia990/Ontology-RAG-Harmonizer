import pandas as pd
import numpy as np
from typing import Optional

from RAG_mapper.src.RAG_mapper import RAGMapper

def pick_feature(row: pd.Series) -> Optional[str]:
    if pd.isna(row.iloc[1]):
        return row.iloc[0]
    else:
        return row.iloc[1]


def preprocess_var_series(row: pd.Series, add_description: bool = True):
    try:
        cleaned = row.fillna('')
        search_query = pick_feature(row)
        description = cleaned.iloc[2]
        if search_query:

            search_query = search_query + ' ' + description
            # search_query = ' '.join(row.fillna('').apply(str))
            return search_query
        else:
            return np.nan

    except Exception as e:
        print(f"Error processing row ({row}): {e}")
        return np.nan


FILE = 'Benchmark_MicrobAIome.xlsx'
output_file = 'results/LuNikJay_benchmark/Ontology_Benchmark_Mapping.xlsx'

data = pd.read_excel(FILE)
irish_var = data.iloc[:, 0:3].dropna(how='all')
french_var = data.iloc[:, 3:6].dropna(how='all')

A_irish = irish_var.apply(preprocess_var_series, axis=1).dropna().to_list()
A_french = french_var.apply(preprocess_var_series, axis=1).dropna().to_list()

RAG_mapper = RAGMapper(var_list= A_irish + A_french, k = 20)

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:

    for var in A_irish:
        sheet_name = f'irish_{''.join(e for e in var if e.isalnum())[:20]}'
        candidate_df = RAG_mapper.map_umls(var)
        candidate_df['variable'] = var
        candidate_df['curator_rank'] = None
        candidate_df.to_excel(writer, sheet_name=sheet_name, index=False)

    for var in A_french:
        sheet_name = f'french_{''.join(e for e in var if e.isalnum())[:20]}'  # Use a clear, unique sheet name
        candidate_df = RAG_mapper.map_umls(var)
        candidate_df['curator_rank'] = None
        candidate_df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Successfully created '{output_file}' with {len(A_irish) + len(A_french)} sheets.")