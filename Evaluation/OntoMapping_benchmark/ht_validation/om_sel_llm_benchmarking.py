from Evaluation.OntoMapping_benchmark.src.OM_pipeline import *
from Evaluation.OntoMapping_benchmark.src.compute_scores import *
import argparse

logger = setup_logging()

def open_mapping_table(vocabularies: str) -> pd.DataFrame:
    vocabularies_list = vocabularies.split()
    source_onto = vocabularies_list[0]
    target_onto = vocabularies_list[1]
    logging.info("Opening mapping table")
    try:
        res_df = pd.read_csv(f'Evaluation/OntoMapping_benchmark/mapped_codes/mapping_{source_onto}_{target_onto}.csv')
        res_df[f'{target_onto}_ids'] = res_df[f'{target_onto}_ids'].apply(lambda x: parse_to_list_of_strings(x))
        res_df[f'{source_onto}_names'] = res_df[f'{source_onto}_names'].apply(lambda x: parse_to_list_of_strings(x))
        logging.info(f"Mapping table: {res_df.shape} with columns: {res_df.columns}")
        return res_df
    except FileNotFoundError as e:
        logger.error(f"Error loading results: {e}", exc_info=True)
        return pd.DataFrame()



def get_checkpoint_path_selector(results_dir: str, source_onto: str, target_onto: str, MAX_LENGTH: int, llm_model:str) -> str:
    """Return the path for the checkpoint CSV file."""
    return f"{results_dir}/selector_output_prompt_2_{llm_model}_{MAX_LENGTH}_{source_onto}_{target_onto}.csv"

def load_checkpoint_selector(results_dir: str, source_onto: str, target_onto: str, mapping_table: pd.DataFrame, MAX_LENGTH: int, llm_model:str) -> int:
    """
    Returns the positional offset (iloc position) to resume from,
    by checking how many of the mapping_table's index values are already saved.
    """
    checkpoint_path = get_checkpoint_path_selector(results_dir, source_onto, target_onto, MAX_LENGTH, llm_model)
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

        out_path = get_checkpoint_path_selector(results_dir, source_onto, target_onto, MAX_LENGTH, llm_model)
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
        results = [rag_mapper.RAG_map(var, llm_model) for var in sampled_source]
    return results, sampled_idx

def selector_evaluation(source_onto:str, target_onto:str, mapping_table: pd.DataFrame, MAX_LENGTH: int, k: int, t:float, results_dir: str, llm_model: str = 'gpt-oss:20b', chunk_size: int = 100, sampling_ratio:float=0.5):
    logger.info(f"{'=' * 60}")
    logger.info(f"{llm_model} selector evaluation: {source_onto} → {target_onto}  |  MAX_LENGTH={MAX_LENGTH}")
    logger.info(f"{'=' * 60}")
    # 1. Check for existing checkpoint and determine starting row
    start_index = load_checkpoint_selector(results_dir, source_onto, target_onto, mapping_table, MAX_LENGTH, llm_model)

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

    logger.info(f"selector_evaluation with llm {llm_model} complete: {source_onto} → {target_onto} ")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocabularies', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--results_dir', type=str, default='results/OntoMapping_benchmark/llms_benchmarking/')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--t', type=float, default=0.6)
    parser.add_argument('--sampling_ratio', type=float, default=0.5)
    args = parser.parse_args()
    vocabularies = args.vocabularies
    max_length = args.max_length
    results_dir = args.results_dir
    k = args.k
    t = args.t
    sampling_ratio = args.sampling_ratio
    list_vocabularies = vocabularies.split()
    source_onto = list_vocabularies[0]
    target_onto = list_vocabularies[1]
    with timer("Creating mapping table"):
        try:
            mapping_table = open_mapping_table(vocabularies)
            logger.info(f"Mapping table shape: {mapping_table.shape}")
        except Exception as e:
            logger.error(f"Failed to create mapping table: {e}", exc_info=True)
            exit()

    llm_models = ['granite4:latest', 'gpt-oss:120b',  'gpt-oss:20b', 'gemma4:latest']
    computational_time = {}
    for llm_model in llm_models:
        print(f'---- Starting selector evaluation with {llm_model}---- \n')
        start_time = time.time()
        selector_evaluation(source_onto, target_onto, mapping_table, max_length, k, t, results_dir, llm_model=llm_model, sampling_ratio=sampling_ratio)
        computational_time[llm_model] = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

        with open('results/OntoMapping_benchmark/llms_benchmarking/computational_time_selector.txt', 'a') as file:
            file.write(f'{llm_model}: {computational_time[llm_model]} \n')
        print(f'{llm_model}: {computational_time[llm_model]}\n')
        print(f'----- Finish selector evaluation with {llm_model} in {computational_time[llm_model]} seconds  ---- \n')



    print(f'------Finish benchmarking llms {llm_models}')

