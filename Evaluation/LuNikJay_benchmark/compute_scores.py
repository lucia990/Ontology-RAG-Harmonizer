import pandas as pd
from typing import List, Dict
import ast
from collections import Counter
from Evaluation.process_benchmark import process_ontology_dataset
from RAG_mapper.src.schema_builder import SchemaBuilder


def collect_benchmarks(curators: List):
    benchmarks = {}
    for curator in curators:
        benchmarks[curator] = process_ontology_dataset(curator)
    return benchmarks


def rag_map(var_to_map:List ) -> pd.DataFrame:
    rag_mapper = SchemaBuilder(var_to_map)
    rag_output = rag_mapper.create_schema()
    return rag_output

def build_comparison_matrix(rag_output: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    # concatenate expected and retrieved output
    y_df = pd.concat([benchmark[['matched_cuis', 'status']], rag_output['ontology_id']], axis=1)
    # add the column of matches found
    y_df['is_found'] = y_df.apply(
        lambda r: int(
            isinstance(r['matched_cuis'], (list, set, tuple)) and
            r['ontology_id'] in r['matched_cuis']
        ),
        axis=1
    )
    return y_df


################################ curator specif metrics #########################
def compute_agreement(curators: List, y_dfs: Dict) -> pd.DataFrame:
    agreement_column = pd.concat([df['is_found'] for df in y_dfs.values()], axis=1).sum(axis=1)
    return agreement_column*len(curators)

def se_performance(y_df):
    return (y_df["status"] == "success").mean()

def hallucination_rate(y_df):
    failed = y_df[y_df["status"].isin(["no_mapping", "all_wrong"])]

    if len(failed) == 0:
        return 0.0

    # Hallucination = should abstain but outputs an ontology ID
    hallucinated = ~failed["ontology_id"].isin([None, "Needs_Review"])

    return hallucinated.mean()

def retrieval_conditioned_accuracy(df):
    retrieved = df[df["status"].isin(["success", "all_wrong"])]
    if len(retrieved) == 0:
        return 0.0
    return retrieved["is_found"].mean()

def unconditioned_accuracy(df):
    return df["is_found"].mean()

def status_distribution(df):
    return df["status"].value_counts(normalize=True)

def abstention_rate(df):
    return (df["ontology_id"].isna()).mean()

def compute_metrics(dfs, output_file):
    rows = []

    for curator, df in dfs.items():

        rows.append({
            "curator": curator,
            "search_engine_success_rate": round(se_performance(df),2),
            "retrieval_conditioned_agreement": round(retrieval_conditioned_accuracy(df),2),
            "unconditioned_agreement": round(unconditioned_accuracy(df),2),
            "hallucination_rate": round(hallucination_rate(df), 2),
            "status_distribution": round(status_distribution(df), 2),
            "abstention_rate": round(abstention_rate(df), 2),
            "n_samples": len(df)
        })

    df = pd.DataFrame(rows).set_index("curator")
    df.to_excel(output_file)

    return df



def stack_curator_data(dfs):
    frames = []

    for curator, df in dfs.items():
        tmp = df[["is_found"]].copy()
        tmp["curator"] = curator
        tmp["variable"] = tmp.index
        frames.append(tmp)

    return pd.concat(frames, ignore_index=True)


def variable_agreement_score(dfs, curator_weights=None):
    """
    Computes weighted agreement per variable using DataFrame index.
    """
    data = stack_curator_data(dfs)

    # Default equal weights
    if curator_weights is None:
        curator_weights = {c: 1.0 for c in data["curator"].unique()}

    data["weight"] = data["curator"].map(curator_weights)

    agreement = (
        data
        .groupby("variable")
        .apply(
            lambda x: (x["is_found"] * x["weight"]).sum() / x["weight"].sum()
        )
        .rename("agreement_score")
        .reset_index()
    )

    return agreement

# ########################### Aggregated benchmarks #########################
def read_comparison_matrices(max_length: int = 72):
    y_dfs = {}
    for phd in curators:
        y_dfs[phd] = pd.read_excel(f'Evaluation/comparison_matrix_{max_length}_{phd}.xlsx', index_col=0)
        y_dfs[phd]['matched_cuis'] = y_dfs[phd]['matched_cuis'].apply(ast.literal_eval)
    return y_dfs

def convert_to_set(benchmarks):
    matched_sets = {
        curator: df["matched_cuis"].apply(set)
        for curator, df in benchmarks.items()
    }
    return pd.DataFrame(matched_sets)

def count_overlapping_cuis(row):
    all_cuis = list(row.dropna())
    from collections import Counter
    counts = Counter(cui for s in all_cuis for cui in s)
    return sum(v > 1 for v in counts.values())

def normalized_agreement(row):
    sets = [s for s in row.dropna()]
    counts = Counter(cui for s in sets for cui in s)
    union = set.union(*sets)
    if not union:
        return 1.0
    overlap =  sum(v > 1 for v in counts.values())
    union = len(set.union(*sets))

    return overlap / union if union else 1


def system_accuracy(row, ontology_id):
    return sum(ontology_id in s for s in row) / len(row)

if __name__ == '__main__':

    y_dfs = {}

    # collect benchmark data
    curators = ["Nik", "Julian", "Lu"]
    benchmarks = collect_benchmarks(curators)

    # run RAG mechanism on benchmark variables
    var_to_map = benchmarks['Lu']['variable'].to_list()
    rag_mapper = SchemaBuilder(var_to_map, k = 50)
    rag_output = rag_mapper.create_schema()

    for curator in curators:
        y_dfs[curator] = build_comparison_matrix(rag_output, benchmarks[curator])
        y_dfs[curator].to_excel(f"Evaluation/comparison_matrix_{curator}.xlsx")
    agreement = compute_agreement(curators, y_dfs)

    res_matrix = compute_metrics(y_dfs, 'Evaluation/metrics.xlsx')
    agreement_df = variable_agreement_score(y_dfs)

    print(agreement_df.head())
'''
# Run RAG on benchmark data
curators_ds = ["Nik", "Julian", "Lu"]
benchmark = {}
rag_outputs = {}
rag_dict = {}
y_df = {}
for phd in curators_ds:
    print(f"-------------------------------------- Processing {phd}...-------------------------------------------------")
    benchmark[phd] = process_ontology_dataset(phd)

    var_to_map = benchmark[phd]['variable'].to_list()
    rag_mapper = SchemaBuilder(var_to_map)

    rag_outputs[phd] = rag_mapper.create_schema()

    rag_dict[phd] = rag_outputs[phd].to_dict(orient='records')

    mask = rag_outputs[phd]['variable'].isin(benchmark[phd]['variable'])

    rag_output = rag_outputs[phd][mask]
    single_y_df = pd.concat([benchmark[phd][['matched_cuis', 'status']], rag_outputs[phd]['ontology_id']], axis=1)

    single_y_df['is_found'] = single_y_df.apply(
        lambda r: int(
            isinstance(r['matched_cuis'], (list, set, tuple)) and
            r['ontology_id'] in r['matched_cuis']
        ),
        axis=1
    )
    y_df[phd] = single_y_df

    print(f"search engine performance: {len(single_y_df[single_y_df['status'] == 'success']) / len(single_y_df)}")
    print(f"Rate of absolut correctness for {phd}: {single_y_df.is_found.sum() / len(single_y_df)}\n")

    print(
        f"Rate of success vs corretness for {phd}: {len(single_y_df[(single_y_df['status'] == 'success') & (single_y_df['is_found'] == 1)]) / len(single_y_df)}\n")
agreement_column = pd.concat([df['is_found'] for df in y_df.values()], axis=1).sum(axis=1)

with open('Evaluation/rag_outputs_72.json', 'w') as outfile:
    json.dump(rag_outputs, outfile, indent = 4)
print(f"RAG outputs saved to {outfile.name}")
# ... WORK IN PROGRESS ...

# Get precomputed rag-based schema
irish_local_schema =pd.read_excel('results/irish_local_schema_256.xlsx', index_col =0)
french_local_schema =pd.read_excel('results/french_local_schema_256.xlsx', index_col =0)

irish_local_schema['variable'] = irish_local_schema['variable'].apply(lambda x : x.strip().replace(' ', ''))
french_local_schema['variable'] = french_local_schema['variable'].apply(lambda x : x.strip().replace(' ', ''))
rag_output = pd.concat([irish_local_schema, french_local_schema], ignore_index=True)

curators_ds = ["Nik", "Julian", "Lucia"]
benchmark = {}
rag_outputs = {}
y_df = {}
for phd in curators_ds:
    print(f"Processing {phd}...")
    benchmark[phd] = process_ontology_dataset(phd)

    mask = rag_output['variable'].isin(benchmark[phd]['variable'])

    rag_outputs[phd] = rag_output[mask]

    single_y_df = pd.concat([benchmark[phd][['matched_cuis', 'status']], rag_outputs[phd]['ontology_id']], axis = 1)

    single_y_df['is_found'] = single_y_df.apply(
        lambda r: int(
            isinstance(r['matched_cuis'], (list, set, tuple)) and
            r['ontology_id'] in r['matched_cuis']
        ),
        axis=1
    )
    #y_df[phd] = single_y_df

    print(f"search engine performance: {len(single_y_df[single_y_df['status'] == 'success'])/len(single_y_df)}")
    print(f"Rate of absolut correctness for {phd}: {single_y_df.is_found.sum()/len(single_y_df)}\n")


    print(f"Rate of success vs corretness for {phd}: {len(single_y_df[(single_y_df['status'] == 'success') & (single_y_df['is_found']==1)])/len(single_y_df)}\n")

'''


