import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from typing import List

from RAG_mapper.src.RAG_mapper import RAGMapper

OUTPUT_DIR = 'results/OntoMapping_benchmark/ht_validation/mel_validation_25'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_FILE = f"{OUTPUT_DIR}/sup_benchmark_checkpoints.csv"
MAPPING_LOG_FILE = f"{OUTPUT_DIR}/sup_evaluators_production_mappings.csv"


def save_checkpoint(query: str, model: str, k: int, t: float, true_rank: int, is_valid: bool):
    file_exists = os.path.isfile(CHECKPOINT_FILE)
    df_row = pd.DataFrame([{
        "query_variable": query,
        "model": model,  # Tracks the active Evaluator LLM
        "k": k,  # Inherited from frozen step
        "t": t,  # Inherited from frozen step
        "true_rank": true_rank,  # Evaluator performance metric
        "is_valid": int(is_valid)
    }])
    df_row.to_csv(CHECKPOINT_FILE, mode='a', header=not file_exists, index=False)


def save_detailed_mapping(
        query: str,
        selector_model: str,
        evaluator_model: str,
        k: int,
        t: float,
        retrieved_candidates_json: str,  # Serialized list from Step 1
        selector_shortlist_json: str,  # Granite's original frozen list
        evaluator_output_list: list,  # New re-ranked normalized list
        true_rank: int,
        is_valid: bool
):
    """
    Saves a comprehensive expanded table preserving the intermediate
    selector run state alongside the brand-new evaluator final run column.
    """
    file_exists = os.path.isfile(MAPPING_LOG_FILE)

    df_row = pd.DataFrame([{
        "query_variable": query,
        "model": selector_model,  # Kept for compatibility with your Step 1 schema
        "evaluator_model": evaluator_model,  # Explicit tracking for the new stage 2 model
        "k": k,
        "t": t,
        "retrieved_candidates": retrieved_candidates_json,
        "llm_output_shortlist": selector_shortlist_json,  # Frozen granite4 results
        "llm_final_output": json.dumps(evaluator_output_list, ensure_ascii=False),  # New Evaluator results
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


def run_evaluator_step(csv_path: str, benchmark_config: dict):
    """
    Executes the second step of the evaluation pipeline by feeding frozen intermediate results
    from SELECTOR_OUTPUT_CSV into the designated evaluator chain, parsing the Supervisor nested structure.
    """
    print(f"Loading frozen intermediate selector dataset from {csv_path}...")
    df_all = pd.read_csv(csv_path)

    processed_registry = load_processed_queries()

    mapper = RAGMapper(
        var_list=[''], var_desc=[''],
        FAISS_INDEX_PATH="UMLS_mapper/data/processed/faiss_index_25.bin",
        METADATA_PATH="UMLS_mapper/data/processed/metadata_25.csv"
    )

    # 1. Iterate over the targeted Evaluator models
    for model, configurations in benchmark_config.items():
        print(f"\n=======================================================")
        print(f" WARMING UP STEP 2 EVALUATOR: {model}")
        print(f"=======================================================")

        try:
            evaluator_chain = mapper.create_evaluator_chain(llm_model=model)
        except Exception as e:
            print(f"Failed initialization for evaluator model {model}: {e}. Skipping.")
            continue

        # 2. Extract baseline runs that match the target hyperparameters
        for config in configurations:
            k_val = config['k']
            t_val = config['t']

            config_mask = (df_all['k'] == k_val) & (df_all['t'] == t_val)
            target_rows = df_all[config_mask]

            if target_rows.empty:
                print(f"No logged source configs found for k={k_val}, t={t_val}. Skipping configuration block.")
                continue

            for _, row in tqdm(target_rows.iterrows(), total=len(target_rows), desc=f"Evaluating via {model}"):
                query = row['query_variable']
                selector_model_name = row['model']

                # Check duplication registry using active evaluator model tag
                composite_key = f"{model}|{k_val}|{t_val}|{query}"
                if composite_key in processed_registry:
                    continue

                # Parse the frozen retrieved candidate list and shortlist
                raw_candidates = row['retrieved_candidates']
                raw_shortlist = row['llm_output_shortlist']

                # Safe list unpacking (handles raw strings or parsed python objects seamlessly)
                candidates_list = json.loads(raw_candidates) if isinstance(raw_candidates, str) else raw_candidates
                shortlist_list = json.loads(raw_shortlist) if isinstance(raw_shortlist, str) else raw_shortlist

                # --- STEP 3: FORMAT PROMPT TEMPLATES FOR THE EVALUATOR CHAIN ---
                candidates_text = "\n".join(
                    f"{i + 1}. ontology_name: {c['ontology_name']} | confidence: {c['confidence']:.2f} | CUI: {c['CUI']}"
                    for i, c in enumerate(candidates_list)
                )

                rag_result_text = "\n".join(
                    f"- CUI: {item.get('CUI', '')} | Name: {item.get('AI_name', item.get('ontology_name', ''))}"
                    for item in shortlist_list
                )

                # Recover the ground truth CUI from context tracking
                ground_truth_cui = row.get('ground_truth', None)
                if pd.isna(ground_truth_cui):
                    ground_truth_cui = shortlist_list[0].get('ground_truth', '') if shortlist_list else ''

                # --- STEP 4: INFERENCE AND NESTED STRUCTURAL PARSING ---
                try:
                    # Execute evaluator chain payload (returns a dataframe containing the dumped parsed Pydantic dict)
                    raw_agent_df = evaluator_chain(
                        variable=query,
                        var_desc="",
                        rag_result=rag_result_text,
                        candidates=candidates_text
                    )

                    if raw_agent_df.empty or 'supervised_ranking' not in raw_agent_df.columns:
                        save_checkpoint(query, model, k_val, t_val, true_rank=0, is_valid=False)
                        save_detailed_mapping(query, selector_model_name, model, k_val, t_val,
                                              json.dumps(candidates_list, ensure_ascii=False),
                                              json.dumps(shortlist_list, ensure_ascii=False), [], true_rank=0,
                                              is_valid=False)
                        continue

                    # Unpack the nested Pydantic structures from the first row record
                    record_dict = raw_agent_df.iloc[0].to_dict()
                    rag_review = record_dict.get('rag_review', {})
                    supervised_ranking = record_dict.get('supervised_ranking', {})

                    # Extract structural data safely regardless of inner dictionary parsing formats
                    review_status = rag_review.get('status', 'VALID')
                    ranked_candidates_list = supervised_ranking.get('candidates', [])

                    if review_status == "Needs_Review" or not ranked_candidates_list:
                        save_checkpoint(query, model, k_val, t_val, true_rank=0, is_valid=True)
                        save_detailed_mapping(query, selector_model_name, model, k_val, t_val,
                                              json.dumps(candidates_list, ensure_ascii=False),
                                              json.dumps(shortlist_list, ensure_ascii=False), [], true_rank=0,
                                              is_valid=True)
                        continue

                    # --- STEP 5: ALIGN EVALUATOR DICT MATRIX TO MATCH STEP 1 SELECTION SCHEMA ---
                    normalized_evaluator_shortlist = []
                    for item in ranked_candidates_list:
                        normalized_evaluator_shortlist.append({
                            "variable": query,
                            "rank": item.get('rank'),
                            "AI_code": item.get('CUI'),
                            "AI_name": item.get('CUI_label'),
                            "confidence": item.get('confidence'),
                            "CUI": item.get('CUI'),
                            "reasoning": item.get('reasoning')  # Supervisor explanation telemetry
                        })

                    # Calculate target validation true rank positions
                    evaluator_cuis = [str(item['CUI']).strip() for item in normalized_evaluator_shortlist]

                    try:
                        true_rank = evaluator_cuis.index(str(ground_truth_cui).strip()) + 1
                    except ValueError:
                        true_rank = 0

                    save_checkpoint(query, model, k_val, t_val, true_rank, is_valid=True)
                    save_detailed_mapping(query, selector_model_name, model, k_val, t_val,
                                          json.dumps(candidates_list, ensure_ascii=False),
                                          json.dumps(shortlist_list, ensure_ascii=False),
                                          normalized_evaluator_shortlist, true_rank, is_valid=True)

                except Exception as e:
                    print(f"Error processing query {query}: {e}")
                    save_checkpoint(query, model, k_val, t_val, true_rank=0, is_valid=False)
                    save_detailed_mapping(query, selector_model_name, model, k_val, t_val,
                                          json.dumps(candidates_list, ensure_ascii=False),
                                          json.dumps(shortlist_list, ensure_ascii=False), [], true_rank=0,
                                          is_valid=False)


def compute_metrics_from_checkpoints_optimized(benchmark_config: dict) -> pd.DataFrame:
    if not os.path.isfile(CHECKPOINT_FILE):
        return pd.DataFrame()

    df = pd.read_csv(CHECKPOINT_FILE)
    if df.empty:
        return pd.DataFrame()

    valid_configs = []
    for model, configs in benchmark_config.items():
        for config in configs:
            valid_configs.append((model, config['k'], config['t']))

    config_df = pd.DataFrame(valid_configs, columns=['model', 'k', 't'])
    df = pd.merge(df, config_df, on=['model', 'k', 't'], how='inner')

    if df.empty:
        return pd.DataFrame()

    df['reciprocal_rank'] = np.where(df['true_rank'] > 0, 1.0 / df['true_rank'], 0.0)
    df['hit_1'] = (df['true_rank'] == 1).astype(int)
    df['hit_3'] = ((df['true_rank'] > 0) & (df['true_rank'] <= 3)).astype(int)
    df['hit_5'] = ((df['true_rank'] > 0) & (df['true_rank'] <= 5)).astype(int)
    df['is_invalid'] = (df['is_valid'] == 0).astype(int)

    summary = df.groupby(['model', 'k', 't']).agg(
        total_queries=('is_valid', 'count'),
        mrr=('reciprocal_rank', 'mean'),
        hit_1_rate=('hit_1', 'mean'),
        hit_3_rate=('hit_3', 'mean'),
        hit_5_rate=('hit_5', 'mean'),
        fail_rate=('is_invalid', 'mean')
    ).reset_index()

    summary_report = pd.DataFrame()
    summary_report['LLM Model (Step 2 Evaluator)'] = summary['model']
    summary_report['Retriever Setup'] = "k=" + summary['k'].astype(str) + ", t=" + summary['t'].astype(str)
    summary_report['MRR'] = summary['mrr'].round(4)
    summary_report['Hit Rate@1 (%)'] = (summary['hit_1_rate'] * 100).round(2)
    summary_report['Hit Rate@3 (%)'] = (summary['hit_3_rate'] * 100).round(2)
    summary_report['Hit Rate@5 (%)'] = (summary['hit_5_rate'] * 100).round(2)
    summary_report['Invalid/Fail Rate (%)'] = (summary['fail_rate'] * 100).round(2)

    output_csv_path = f"{OUTPUT_DIR}/sup_summary_report.csv"
    summary_report.to_csv(output_csv_path, index=False)
    print(f"\n[SUCCESS] Pipeline step summary saved to: {output_csv_path}")
    return summary_report


if __name__ == "__main__":
    # Path to frozen selector mappings dataset
    SELECTOR_OUTPUT_CSV = 'results/OntoMapping_benchmark/ht_validation/mel_validation_25/granite_production_mappings.csv'

    # Configuration for second stage evaluator running loop
    EVALUATOR_CONFIG = {
        "qwen3.6:35b": [
            {"k": 50, "t": 0.6}
        ],
        "gemma4:31b": [
            {"k": 50, "t": 0.6}
        ],
        "gpt-oss:120b": [
            {"k": 50, "t": 0.6}
        ]
    }

    run_evaluator_step(SELECTOR_OUTPUT_CSV, EVALUATOR_CONFIG)
    summary_report = compute_metrics_from_checkpoints_optimized(EVALUATOR_CONFIG)
    print(summary_report)