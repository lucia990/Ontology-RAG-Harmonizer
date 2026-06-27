import os
import argparse
import pandas as pd
import numpy as np
from typing import List
import logging
import time
from datetime import timedelta
from contextlib import contextmanager

from UMLS_mapper.src.faiss_index import FaissUMLS

from RAG_mapper.src.RAG_mapper import RAGMapper
from Evaluation.OntoMapping_benchmark.src.umls_mapping import create_mapping_table
from UMLS_mapper.scripts.umls_embeddings import umls_sapbert_embeddings

def setup_logging(log_file: str = "evaluation.log") -> logging.Logger:
    """Configure a logger that writes to both console and a log file, avoiding duplicates."""
    logger = logging.getLogger("onto_eval")
    logger.setLevel(logging.DEBUG)

    # Check if handlers already exist before adding new ones
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # File handler (DEBUG level so every detail is captured)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

        # Prevent the log messages from being sent to the root logger
        logger.propagate = False

    return logger

logger = setup_logging()


@contextmanager
def timer(step_name: str):
    """Context manager that logs the elapsed time for a named step."""
    logger.info(f"[START] {step_name}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info(f"[DONE]  {step_name} — elapsed: {timedelta(seconds=round(elapsed))}")

def embed_target_ontology(target_onto, MAX_LENGTH):
    with timer(f"Embedding target ontology: {target_onto}"):
        try:
            umls_sapbert_embeddings(vocabularies=target_onto, MAX_LENGTH=MAX_LENGTH)
        except Exception as e:
            logger.error(f"Failed to embed '{target_onto}': {e}", exc_info=True)

def create_target_faiss(emb_path, target_onto, MAX_LENGTH):

    logger.info(f"Loading embeddings from: {emb_path}")
    with timer(f"Building FAISS index for '{target_onto}'"):
        indexed_umls = FaissUMLS(embeddings_path=emb_path)

    try:
        processed_dir = "UMLS_mapper/data/processed"
        if not os.path.exists(processed_dir):
            os.mkdir(processed_dir)
            logger.debug(f"Created directory: {processed_dir}")

        faiss_path = f"{processed_dir}/faiss_index_{MAX_LENGTH}.bin"
        meta_path = f"{processed_dir}/metadata_{MAX_LENGTH}.csv"

        with timer("Storing FAISS index"):
            indexed_umls._store_faiss_index(faiss_path)
            logger.info(f"FAISS index stored → {faiss_path}")

        with timer("Storing metadata"):
            indexed_umls._map_metadata(meta_path)
            logger.info(f"Metadata stored    → {meta_path}")

    except FileExistsError as e:
        logger.error(f"Error creating UMLS_mapper directory: {e}", exc_info=True)


def map_source(source_onto, mapping_table, k, t, llm_model: str, sampling_ratio:float):

    source_series = mapping_table[f'{source_onto}_names'].apply(lambda x: max(x, key=len))

    # Pick a random sample of the source variable
    N = len(source_series)
    print(f'Number of source variables before sampling: {N}')
    K = int(np.floor(sampling_ratio * N))
    np.random.seed(123)
    sampled_idx = np.random.choice(source_series.index, size=K, replace=False)
    sampled_source = source_series.loc[sampled_idx].to_list()

    logger.info(f"Source variables to map: {len(sampled_source)}")

    with timer(f"Initialising RAGMapper for '{sampled_source}'"):
        rag_mapper = RAGMapper(sampled_source, k=k, t=t)

    with timer(f"RAGMapper.evaluate — {len(sampled_source)} entries"):
        results = [rag_mapper.evaluate(var, llm_model) for var in sampled_source]
    return results, sampled_idx

def collect_results(results: List[pd.DataFrame], mapping_table, sampled_idx):

    mapping_table = mapping_table.loc[sampled_idx].copy()

    idx = mapping_table.index

    def safe_get(res, df_idx, col):
        try:
            return res[df_idx][col].iloc[0]
        except (KeyError, IndexError, TypeError):
            logger.warning(f"Missing '{col}' in result at df_idx={df_idx}, returning None")
            return None

    mapping_table = mapping_table.copy()  # avoid SettingWithCopyWarning

    mapping_table.loc[idx, 'se_codes'] = pd.Series(
        [res[0]['se_CODE'].tolist() if res[0] is not None else None for res in results],
        index=idx
    )
    mapping_table.loc[idx, 'se_names'] = pd.Series(
        [res[0]['ontology_name'].tolist() if res[0] is not None else None for res in results],
        index=idx
    )
    mapping_table.loc[idx, 'AI_code'] = pd.Series(
        [safe_get(res, 1, 'AI_code') for res in results],
        index=idx
    )
    mapping_table.loc[idx, 'AI_name'] = pd.Series(
        [safe_get(res, 1, 'AI_name') for res in results],
        index=idx
    )
    return mapping_table

def save_results(source_onto: str, target_onto: str, final_results: pd.DataFrame, results_dir: str, MAX_LENGTH: int):
    try:
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
            logger.debug(f"Created directory: {results_dir}")

        out_path = f"{results_dir}/rag_output_{MAX_LENGTH}_{source_onto}_{target_onto}.csv"
        final_results.to_csv(out_path)
        logger.info(f"Results saved → {out_path}")

    except FileExistsError as e:
        logger.error(f"Error creating results directory: {e}", exc_info=True)


def get_checkpoint_path(results_dir: str, source_onto: str, target_onto: str, MAX_LENGTH: int) -> str:
    """Return the path for the checkpoint CSV file."""
    return f"{results_dir}/rag_output_{MAX_LENGTH}_{source_onto}_{target_onto}.csv"


def load_checkpoint(results_dir: str, source_onto: str, target_onto: str, mapping_table: pd.DataFrame, MAX_LENGTH: int) -> int:
    """
    Returns the positional offset (iloc position) to resume from,
    by checking how many of the mapping_table's index values are already saved.
    """
    checkpoint_path = get_checkpoint_path(results_dir, source_onto, target_onto, MAX_LENGTH)
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


def save_chunk(chunk: pd.DataFrame, results_dir: str, source_onto: str, target_onto: str, append: bool, MAX_LENGTH: int) -> None:
    """
    Save a chunk of results to the CSV checkpoint file.
    Appends if file already exists, writes header only on first write.
    """
    try:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            logger.debug(f"Created directory: {results_dir}")

        out_path = get_checkpoint_path(results_dir, source_onto, target_onto, MAX_LENGTH)
        chunk.to_csv(out_path, mode='a' if append else 'w', header=not append)
        logger.info(f"Chunk saved ({len(chunk)} rows) → {out_path}")

    except Exception as e:
        logger.error(f"Error saving chunk: {e}", exc_info=True)
        raise

def one_way_evaluation(source_onto:str, target_onto:str, mapping_table: pd.DataFrame, MAX_LENGTH: int, k: int, t:float, results_dir: str, llm_model: str, chunk_size: int = 100, sampling_ratio:float=0.5):
    logger.info(f"{'=' * 60}")
    logger.info(f"one_way_evaluation: {source_onto} → {target_onto}  |  MAX_LENGTH={MAX_LENGTH}")
    logger.info(f"{'=' * 60}")
    # 1. Check for existing checkpoint and determine starting row
    start_index = load_checkpoint(results_dir, source_onto, target_onto, mapping_table, MAX_LENGTH)

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

    # 3. Create FAISS index
    emb_path = f"UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{target_onto}.parquet"
    create_target_faiss(emb_path, target_onto, MAX_LENGTH)

    # 4. Process in chunks of `chunk_size`
    for chunk_start in range(0, len(remaining_table), chunk_size):
        chunk_table = remaining_table.iloc[chunk_start: chunk_start + chunk_size]

        absolute_start = start_index + chunk_start
        absolute_end = absolute_start + len(chunk_table)
        logger.info(f"Processing rows {absolute_start}–{absolute_end - 1} …")

        # Run model on this chunk only
        results, sampled_idx = map_source(source_onto, chunk_table, k, t, llm_model=llm_model, sampling_ratio=sampling_ratio)

        # Collect results for the chunk
        chunk_results = collect_results(results, chunk_table, sampled_idx)

        logger.debug(f"se_codes sample: {chunk_results['se_codes'].head(3).tolist()}")
        logger.debug(f"AI_code sample:  {chunk_results['AI_code'].head(3).tolist()}")

        # Append to CSV (write header only for the very first chunk)
        is_first_write = (absolute_start == 0)
        save_chunk(chunk_results, results_dir, source_onto, target_onto, append=not is_first_write, MAX_LENGTH = MAX_LENGTH)

    logger.info(f"one_way_evaluation complete: {source_onto} → {target_onto}")

    # 4. Run model on source ontology
    # results = map_source(source_onto, mapping_table, k, t, llm_model=llm_model)






if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocabularies', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--results_dir', type=str, default='results/OntoMapping_benchmark/granite_gemma/')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--t', type=float, default=0.6)
    parser.add_argument('--llm_model', type=str, default='gemma4:latest')
    parser.add_argument('--sampling_ratio', type=float, default=0.1)

    args = parser.parse_args()
    vocabularies = args.vocabularies
    max_length = args.max_length
    results_dir = args.results_dir
    k = args.k
    t = args.t
    llm_model = args.llm_model
    sampling_ratio = args.sampling_ratio
    list_vocabularies = vocabularies.split()

    source_onto = list_vocabularies[0]
    target_onto = list_vocabularies[1]

    with timer("Creating mapping table"):
        try:
            mapping_table = create_mapping_table(vocabularies)
            logger.info(f"Mapping table shape: {mapping_table.shape}")
        except Exception as e:
            logger.error(f"Failed to create mapping table: {e}", exc_info=True)
            exit()

    print(f'---- Starting one way evaluation for vocabularies: {vocabularies} ---- \n')
    #two_way_evaluation(vocabularies, MAX_LENGTH=max_length, results_dir=results_dir, k=k, t=t, llm_model=llm_model, sampling_ratio=sampling_ratio)
    one_way_evaluation(source_onto, target_onto, mapping_table, max_length, k, t, results_dir, llm_model=llm_model, sampling_ratio=sampling_ratio)

    print(f'----- Finish one way evaluation for vocabularies: {vocabularies} ---- \n')
