import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from typing import List

from RAG_mapper.src.RAG_mapper import RAGMapper

OUTPUT_DIR = 'results/OntoMapping_benchmark/ht_validation/mel_validation_25'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_FILE = f"{OUTPUT_DIR}/benchmark_checkpoints.csv"
MAPPING_LOG_FILE = f"{OUTPUT_DIR}/granite_production_mappings.csv" # "best" performing selector LLM

def save_checkpoint(query: str, model: str, k: int, t: float, true_rank: int, is_valid: bool):

    '''
    OUTPUT COLUMNS DESCRIPTION:

    query_variable: The literal string query input being mapped

    model: The specific LLM string tag currently running inference

    k: The capacity limit parameter applied to filter the candidate list for this run.

    t: The semantic similarity threshold parameter applied to filter candidates for this run.

    true_rank: This is the heart of the evaluation engine:

        1: The LLM successfully selected the ground-truth CUI as its absolute first choice.
        2 or 3: The LLM kept it in its short list, but put it a few ranks down.
        0: The ground-truth CUI was either never surfaced by the retriever under those (k,t) parameters, OR the LLM omitted it completely from its final selection.

    is_valid: A structural sanity checker flag:

        1: The model ran successfully, and its output correctly parsed into your Pydantic schema (RankedCandidates).

        0: The model failed (e.g., it suffered an Ollama network timeout, or it outputted malformed/hallucinated JSON text that broke the Pydantic parser). This is what computes your Invalid/Fail Rate (%).
    '''

    file_exists = os.path.isfile(CHECKPOINT_FILE)
    df_row = pd.DataFrame([{
        "query_variable": query,
        "model": model,
        "k": k,
        "t": t,
        "true_rank": true_rank,
        "is_valid": int(is_valid)
    }])
    df_row.to_csv(CHECKPOINT_FILE, mode='a', header=not file_exists, index=False)


def save_detailed_mapping(
        query: str,
        model: str,
        k: int,
        t: float,
        retrieved_df: pd.DataFrame,
        llm_output_df: pd.DataFrame,
        true_rank: int,
        is_valid: bool
):
    """
    Saves the absolute state of the RAG mapping step, logging full lists
    as serialized JSON strings inside the CSV row.
    """
    file_exists = os.path.isfile(MAPPING_LOG_FILE)

    # Structure retrieved candidates safely
    retrieved_candidates = []
    if not retrieved_df.empty:
        retrieved_candidates = retrieved_df[['CUI', 'ontology_name', 'confidence']].to_dict(orient='records')

    # Structure the LLM's selected short-list safely
    llm_selection_list = []
    if is_valid and not llm_output_df.empty:
        # Grabs all columns returned by your Pydantic schema (e.g., CUI, reasons, etc.)
        llm_selection_list = llm_output_df.to_dict(orient='records')

    df_row = pd.DataFrame([{
        "query_variable": query,
        "model": model,
        "k": k,
        "t": t,
        "retrieved_candidates": json.dumps(retrieved_candidates, ensure_ascii=False),
        "llm_output_shortlist": json.dumps(llm_selection_list, ensure_ascii=False),
        "true_rank": true_rank,
        "is_valid": int(is_valid)
    }])

    df_row.to_csv(MAPPING_LOG_FILE, mode='a', header=not file_exists, index=False)

def load_processed_queries() -> set:
    if not os.path.isfile(CHECKPOINT_FILE):
        return set()
    try:
        df = pd.read_csv(CHECKPOINT_FILE)
        return set(f"{row['model']}|{row['k']}|{row['t']}|{row['query_variable']}" for _, row in df.iterrows())
    except Exception as e:
        print(f"Warning: Could not read checkpoint file ({e}). Starting fresh.")
        return set()


def run_llm_benchmark_efficient(csv_path: str, benchmark_config: dict):
    """
    Highly efficient evaluation loop mapping specific configurations
    to their designated models to prevent redundant VRAM usage.
    """
    print(f"Loading candidate dataset from {csv_path}...")
    df_all = pd.read_csv(csv_path)

    # ... (Your standard text data sanitization steps here) ...

    grouped = df_all.groupby('query_variable')
    processed_registry = load_processed_queries()

    mapper = RAGMapper(
        var_list=[''], var_desc=[''],
        FAISS_INDEX_PATH="UMLS_mapper/data/processed/faiss_index_50.bin",
        METADATA_PATH="UMLS_mapper/data/processed/metadata_50.csv"
    )

    # 1. Iterate over models from our custom dictionary
    for model, configurations in benchmark_config.items():
        print(f"\n=======================================================")
        print(f" LOADING MODEL: {model} ({len(configurations)} config(s) to run)")
        print(f"=======================================================")

        try:
            chain = mapper.create_mapper_chain(llm_model=model)
        except Exception as e:
            print(f"Failed initialization for model {model}: {e}. Skipping.")
            continue

        # 2. Iterate through queries while the current model is warm in VRAM
        for query, group in tqdm(grouped, desc=f"Inference on {model}"):
            ground_truth_cui = group['ground_truth'].iloc[0]

            # 3. Only loop through the targeted configurations for this specific model
            for config in configurations:
                k_val = config['k']
                t_val = config['t']

                composite_key = f"{model}|{k_val}|{t_val}|{query}"
                if composite_key in processed_registry:
                    continue

                # Dynamic candidate generation matching (k, t) rules
                filtered_candidates = group[group['confidence'] >= t_val].copy()
                filtered_candidates = filtered_candidates.sort_values(by='confidence', ascending=False)
                filtered_candidates = filtered_candidates.head(k_val).reset_index(drop=True)

                if filtered_candidates.empty:
                    save_checkpoint(query, model, k_val, t_val, true_rank=0, is_valid=True)
                    continue

                candidates_text = "\n".join(
                    f"{i + 1}. ontology_name: {row['ontology_name']} | confidence: {row['confidence']:.2f} | CUI: {row['CUI']}"
                    for i, row in filtered_candidates.iterrows()
                )

                try:
                    # Execute inference payload
                    agent_output_df = chain(variable=query, var_desc="", candidates=candidates_text)

                    if agent_output_df.empty or "CUI" not in agent_output_df.columns:
                        save_checkpoint(query, model, k_val, t_val, true_rank=0, is_valid=False)
                        save_detailed_mapping(query, model, k_val, t_val, filtered_candidates, agent_output_df,
                                              true_rank=0, is_valid=False)
                        continue

                    first_row_cui = str(agent_output_df.iloc[0]['CUI']).strip()
                    if first_row_cui in ["Needs_Review", "None", "nan"]:
                        save_checkpoint(query, model, k_val, t_val, true_rank=0, is_valid=True)
                        save_detailed_mapping(query, model, k_val, t_val, filtered_candidates, agent_output_df,
                                              true_rank=0, is_valid=True)
                        continue

                    agent_output_df['CUI'] = agent_output_df['CUI'].astype(str).str.strip()
                    match_indices = agent_output_df[agent_output_df['CUI'] == ground_truth_cui].index.tolist()

                    if match_indices:
                        true_rank = match_indices[0] + 1

                    else:
                        true_rank = 0
                    save_checkpoint(query, model, k_val, t_val, true_rank, is_valid=True)
                    save_detailed_mapping(query, model, k_val, t_val, filtered_candidates, agent_output_df, true_rank,
                                          is_valid=True)
                except Exception as e:
                    save_checkpoint(query, model, k_val, t_val, true_rank=0, is_valid=False)
                    save_detailed_mapping(query, model, k_val, t_val, filtered_candidates, pd.DataFrame(), true_rank=0, is_valid=False)

def compute_metrics_from_checkpoints(benchmark_config: dict) -> pd.DataFrame:

    OUTPUT_DIR = 'results/OntoMapping_benchmark/ht_validation/mel_validation_25'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isfile(CHECKPOINT_FILE):
        print("No checkpoint file found.")
        return pd.DataFrame()

    df = pd.read_csv(CHECKPOINT_FILE)
    summary_results = []

    for model, configurations in benchmark_config.items():
        for config in configurations:
            k_val = config['k']
            t_val = config['t']

            sub_df = df[(df['model'] == model) & (df['k'] == k_val) & (df['t'] == t_val)]
            if sub_df.empty:
                continue

            total_queries = len(sub_df)
            invalid_failures = len(sub_df[sub_df['is_valid'] == 0])

            reciprocal_ranks = []
            hit_at_1, hit_at_3, hit_at_5 = 0, 0, 0

            for _, row in sub_df.iterrows():
                rank = row['true_rank']
                if rank > 0:
                    reciprocal_ranks.append(1.0 / rank)
                    if rank == 1: hit_at_1 += 1
                    if rank <= 3: hit_at_3 += 1
                    if rank <= 5: hit_at_5 += 1
                else:
                    reciprocal_ranks.append(0.0)

            # Latency can be optionally appended if you track execution times in your checkpoints!
            summary_results.append({
                "LLM Model": model,
                "Retriever Setup": f"k={k_val}, t={t_val}",
                "MRR": round(np.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
                "Hit Rate@1 (%)": round((hit_at_1 / total_queries) * 100, 2) if total_queries > 0 else 0.0,
                "Hit Rate@3 (%)": round((hit_at_3 / total_queries) * 100, 2) if total_queries > 0 else 0.0,
                "Invalid/Fail Rate (%)": round((invalid_failures / total_queries) * 100,
                                               2) if total_queries > 0 else 0.0
            })

    summary_report = pd.DataFrame(summary_results)
    output_csv_path = f"{OUTPUT_DIR}/sel_summary_report.csv"
    summary_report.to_csv(output_csv_path, index=False)

    print(f"\n[SUCCESS] Summary report securely saved to: {output_csv_path}")

    return summary_report


def compute_metrics_from_checkpoints_optimized(benchmark_config: dict) -> pd.DataFrame:

    if not os.path.isfile(CHECKPOINT_FILE):
        print("No checkpoint file found.")
        return pd.DataFrame()

    # 1. Load the data
    df = pd.read_csv(CHECKPOINT_FILE)
    if df.empty:
        print("Checkpoint file is empty.")
        return pd.DataFrame()

    # 2. Build a filtering mask based on your BENCHMARK_CONFIG
    # This ensures we only evaluate the specific (model, k, t) setups you care about
    valid_configs = []
    for model, configs in benchmark_config.items():
        for config in configs:
            valid_configs.append((model, config['k'], config['t']))

    # Convert your config dictionary into a multi-index dataframe for an efficient inner join
    config_df = pd.DataFrame(valid_configs, columns=['model', 'k', 't'])
    df = pd.merge(df, config_df, on=['model', 'k', 't'], how='inner')

    if df.empty:
        print("No matching configurations found in checkpoints.")
        return pd.DataFrame()

    # 3. Vectorized calculations for ranks
    # Pre-calculate Reciprocal Rank (1/rank) for rows where rank > 0, else 0
    df['reciprocal_rank'] = np.where(df['true_rank'] > 0, 1.0 / df['true_rank'], 0.0)

    # Pre-calculate boolean hits (using 1-indexing logic matching your checkpoint storage)
    df['hit_1'] = (df['true_rank'] == 1).astype(int)
    df['hit_3'] = ((df['true_rank'] > 0) & (df['true_rank'] <= 3)).astype(int)
    df['hit_5'] = ((df['true_rank'] > 0) & (df['true_rank'] <= 5)).astype(int)  # This is Recall@5
    df['is_invalid'] = (df['is_valid'] == 0).astype(int)

    # 4. Group by experimental setup and aggregate using fast vectorized means/counts
    summary = df.groupby(['model', 'k', 't']).agg(
        total_queries=('is_valid', 'count'),
        mrr=('reciprocal_rank', 'mean'),
        hit_1_rate=('hit_1', 'mean'),
        hit_3_rate=('hit_3', 'mean'),
        hit_5_rate=('hit_5', 'mean'),
        fail_rate=('is_invalid', 'mean')
    ).reset_index()

    # 5. Format and clean up the final presentation table
    summary_report = pd.DataFrame()
    summary_report['LLM Model'] = summary['model']
    summary_report['Retriever Setup'] = "k=" + summary['k'].astype(str) + ", t=" + summary['t'].astype(str)
    summary_report['MRR'] = summary['mrr'].round(4)
    summary_report['Hit Rate@1 (%)'] = (summary['hit_1_rate'] * 100).round(2)
    summary_report['Hit Rate@3 (%)'] = (summary['hit_3_rate'] * 100).round(2)
    summary_report['Hit Rate@5 (%)'] = (summary['hit_5_rate'] * 100).round(2)  # Added Recall@5 safely
    summary_report['Invalid/Fail Rate (%)'] = (summary['fail_rate'] * 100).round(2)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv_path = f"{OUTPUT_DIR}/sel_summary_report.csv"
    summary_report.to_csv(output_csv_path, index=False)

    print(f"\n[SUCCESS] Optimized summary report securely saved to: {output_csv_path}")
    return summary_report


if __name__ == "__main__":
    csv_path = 'results/OntoMapping_benchmark/ht_validation/mel_validation_50/run_50_candidates.csv'

    BENCHMARK_CONFIG = {
        "gpt-oss:120b": [
            {"k": 50, "t": 0.6}
        ],
        "granite4:latest": [
            {"k": 50, "t": 0.6},
            {"k": 50, "t": 0.7}

        ],
        "qwen3.6:35b": [
            {"k": 50, "t": 0.6}
        ]
    }

    # Isolated production run for your selected config
    PRODUCTION_CONFIG = {
        "granite4:latest": [
            {"k": 20, "t": 0.7}
        ]
    }

    run_llm_benchmark_efficient(csv_path, BENCHMARK_CONFIG)
    summary_report = compute_metrics_from_checkpoints_optimized(BENCHMARK_CONFIG)
    print(summary_report)


