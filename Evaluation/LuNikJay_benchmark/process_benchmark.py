import pandas as pd
from pathlib import Path


def process_ontology_dataset(name):
    file_path = Path(f'Evaluation/Ontology_Benchmark_Mapping_{name}.xlsx')
    all_sheets = pd.read_excel(file_path, sheet_name=None)

    results = []

    for sheet_name, df in all_sheets.items():
        # Clean column names to avoid whitespace issues
        df.columns = [c.strip() for c in df.columns]

        all_retrieved_ids = df['ontology_id'].dropna().tolist()
        variable_name = str(df['variable'].iloc[0]).strip()   #.replace(' ', '')
        cuis = []
        status = "success"

        if not df.empty and df['curator_rank'].iloc[0] == 0:
            status = "all_wrong"

        else:
            mask = df['curator_rank'].isin([1, 2, 3, 4, 5, '1', '2', '3', '4', '5'])
            ranked_matches = df[mask].copy()

            if ranked_matches.empty:
                status = "no_mapping"
            else:
                ranked_matches['curator_rank'] = pd.to_numeric(ranked_matches['curator_rank'])
                ranked_matches = ranked_matches.sort_values('curator_rank')
                cuis = ranked_matches['ontology_id'].tolist()

        results.append({
            "variable": variable_name,
            "matched_cuis": cuis,
            "se_output": all_retrieved_ids,
            "status": status
        })

    return pd.DataFrame(results)






