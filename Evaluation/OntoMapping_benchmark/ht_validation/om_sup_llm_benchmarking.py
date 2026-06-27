from Evaluation.OntoMapping_benchmark.src.OM_pipeline import *
from Evaluation.OntoMapping_benchmark.src.compute_scores import *
import pandas as pd
import time


logger = setup_logging()


def read_res_file(x: str, source_onto: str, target_onto:str) -> pd.DataFrame:
    df = pd.read_csv(f'results/OntoMapping_benchmark/llms_benchmarking/selector_output_{x}_25_{source_onto}_{target_onto}.csv')
    df[f'{target_onto}_ids'] = df[f'{target_onto}_ids'].apply(lambda x: parse_to_list_of_strings(x))
    df[f'{source_onto}_names'] = df[f'{source_onto}_names'].apply(lambda x: parse_to_list_of_strings(x))

    df['se_codes'] = df['se_codes'].apply(lambda x: parse_to_list_of_strings(x))
    df['AI_code'] = df['AI_code'].apply(lambda x: str(int(x)) if isinstance(x, float) and not pd.isna(x) else str(x) if not pd.isna(x) else None)

    return df

def get_checkpoint_path_supervisor(results_dir: str, source_onto: str, target_onto: str, MAX_LENGTH: int, llm_model:str) -> str:
    """Return the path for the checkpoint CSV file."""
    return f"{results_dir}/supervisor_output_{llm_model}_{MAX_LENGTH}_{source_onto}_{target_onto}.csv"

def load_checkpoint_supervisor(results_dir: str, source_onto: str, target_onto: str, mapping_table: pd.DataFrame, MAX_LENGTH: int, llm_model:str) -> int:
    """
    Returns the positional offset (iloc position) to resume from,
    by checking how many of the mapping_table's index values are already saved.
    """
    checkpoint_path = get_checkpoint_path_supervisor(results_dir, source_onto, target_onto, MAX_LENGTH, llm_model)
    if os.path.exists(checkpoint_path):
        try:
            df = pd.read_csv(checkpoint_path, index_col=0)
            saved_indices = set(df.index)
            # Find the first mapping_table index not yet saved
            start_index = next(
                (i for i, idx in enumerate(mapping_table.index) if idx not in saved_indices),
                len(mapping_table)  # all rows already processed
            )
            logger.info(f"Checkpoint found — resuming from iloc position {start_index}")
            return start_index
        except Exception as e:
            logger.warning(f"Could not load checkpoint, starting fresh: {e}")
    return 0

def save_chunk(chunk: pd.DataFrame, results_dir: str, source_onto: str, target_onto: str, append: bool, MAX_LENGTH: int, llm_model:str) -> None:
    """
    Save a chunk of results to the CSV checkpoint file.
    Appends if file already exists, writes header only on first write.
    """
    try:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            logger.debug(f"Created directory: {results_dir}")

        out_path = get_checkpoint_path_supervisor(results_dir, source_onto, target_onto, MAX_LENGTH, llm_model)
        chunk.to_csv(out_path, mode='a' if append else 'w', header=not append)
        logger.info(f"Chunk saved ({len(chunk)} rows) → {out_path}")
    except Exception as e:
        logger.error(f"Error saving chunk: {e}", exc_info=True)
        raise

def map_selector(source_onto: str, mapping_table:pd.DataFrame, llm_model: str, sampling_ratio:float = 0.1, k: int = 5, t:float = 0.6):
    source_series = mapping_table[f'{source_onto}_names'].apply(lambda x: max(x, key=len))
    # Sample from mapping table
    N = len(source_series)
    print(f'Number of source variables before sampling: {N}')
    K = int(np.floor(sampling_ratio * N))
    np.random.seed(123)
    sampled_idx = np.random.choice(source_series.index, size=K, replace=False)
    sampled_source = source_series.loc[sampled_idx].to_list()
    logger.info(f"Source variables to map: {len(sampled_source)}")
    with timer(f"Initialising RAGMapper for '{sampled_source}'"):
        rag_mapper = RAGMapper(sampled_source, k=k, t=t)
    with timer(f"Map — {len(sampled_source)} entries"):
        results = [rag_mapper.evaluate(var, llm_model) for var in sampled_source]
    return results, sampled_idx

def supervisor_evaluation(source_onto:str, target_onto:str, mapping_table: pd.DataFrame, MAX_LENGTH: int, k: int, t:float, results_dir: str, llm_model: str = 'gpt-oss:20b', chunk_size: int = 100, sampling_ratio:float=0.5):
    logger.info(f"{'=' * 60}")
    logger.info(f"{llm_model} selector evaluation: {source_onto} → {target_onto}  |  MAX_LENGTH={MAX_LENGTH}")
    logger.info(f"{'=' * 60}")
    # 1. Check for existing checkpoint and determine starting row
    start_index = load_checkpoint_supervisor(results_dir, source_onto, target_onto, mapping_table, MAX_LENGTH, llm_model)

    if start_index >= len(mapping_table):
        logger.info("All rows already processed — skipping evaluation.")
        return

    if start_index > 0:
        logger.info(f"Resuming from row {start_index} / {len(mapping_table)}")
    else:
        logger.info(f"Starting fresh — {len(mapping_table)} rows to process.")

    # 2. Embed target ontology
    embed_target_ontology(target_onto, MAX_LENGTH)
    # 3. Slice mapping table to only unprocessed rows
    remaining_table = mapping_table.iloc[start_index:]
    # 4. Create FAISS index
    emb_path = f"UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{target_onto}.parquet"
    create_target_faiss(emb_path, target_onto, MAX_LENGTH)
    # 5. Process in chunks of `chunk_size`
    for chunk_start in range(0, len(remaining_table), chunk_size):
        chunk_table = remaining_table.iloc[chunk_start: chunk_start + chunk_size]

        absolute_start = start_index + chunk_start
        absolute_end = absolute_start + len(chunk_table)
        logger.info(f"Processing rows {absolute_start}–{absolute_end - 1} …")

        # Run model on this chunk only
        results, sampled_idx = map_selector(source_onto, chunk_table, llm_model=llm_model, sampling_ratio=sampling_ratio, k = k, t = t)

        # Collect results for the chunk
        chunk_results = collect_results(results, chunk_table, sampled_idx)

        logger.debug(f"se_codes sample: {chunk_results['se_codes'].head(3).tolist()}")
        logger.debug(f"AI_code sample:  {chunk_results['AI_code'].head(3).tolist()}")

        # Append to CSV (write header only for the very first chunk)
        is_first_write = (absolute_start == 0)
        save_chunk(chunk_results, results_dir, source_onto, target_onto, append=not is_first_write, MAX_LENGTH = MAX_LENGTH, llm_model=llm_model)

    logger.info(f"supervisor_evaluation with llm {llm_model} complete: {source_onto} → {target_onto} ")



if __name__ == '__main__':
    llms_models = ['gemma4:latest', 'gpt-oss:120b', 'qwen3.5:4b', 'gpt-oss:20b']
    source_onto = 'NCBI'
    target_onto = 'LNC'
    best_selector = 'granite4:latest'
    mapping_table = read_res_file(best_selector, source_onto, target_onto)
    computational_time = {}
    for llm_model in llms_models:
        start_time = time.time()
        supervisor_evaluation(source_onto, target_onto, mapping_table= mapping_table, llm_model= llm_model, k= 5, t=0.6, MAX_LENGTH=25,  sampling_ratio= 1, results_dir= 'results/OntoMapping_benchmark/llms_benchmarking/supervisor_llm')
        computational_time[llm_model] = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        with open('results/OntoMapping_benchmark/llms_benchmarking/supervisor_llm/computational_time_selector.txt', 'a') as file:
            file.write(f'{llm_model}: {computational_time[llm_model]} \n')