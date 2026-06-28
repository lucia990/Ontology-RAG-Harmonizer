"""
Ontology-to-ontology evaluation.

Loads a pre-computed cross-ontology mapping CSV, samples a subset, embeds
the target ontology from UMLS CONSO, builds a FAISS index, runs the full
RAG pipeline on source concept names, and reports CUI-based Recall@K,
Precision@K, and MRR@K for both FAISS retrieval and the LLM supervisor.

Run from repo root:
  python Evaluation/OntoMapping_benchmark/src/eval_onto_onto.py \\
    --source_onto ICD10 --target_onto LNC --sampling_ratio 0.05
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import UMLS_mapper.src.umls_search_engine as _use_module
from Evaluation.OntoMapping_benchmark.src.compute_scores import ranking_report
from Evaluation.OntoMapping_benchmark.src.OM_pipeline import (
    create_target_faiss, setup_logging, timer,
)
from RAG_mapper.src.RAG_mapper import RAGMapper
from UMLS_mapper.scripts.umls_embeddings import umls_sapbert_embeddings

logger = setup_logging()


def load_mapping_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    list_cols = [c for c in df.columns if df[c].dtype == object]
    df[list_cols] = df[list_cols].map(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df


def normalize_parquet_columns(parquet_path: str) -> None:
    """
    Rename 'Name' -> 'concept_name' and add 'semantic_type' if missing.
    UMLSSearchEngine.search() expects these column names.
    CONSO-based embeddings use 'Name' — this bridges the gap.
    """
    df = pd.read_parquet(parquet_path)
    changed = False
    if "Name" in df.columns and "concept_name" not in df.columns:
        df = df.rename(columns={"Name": "concept_name"})
        changed = True
    if "semantic_type" not in df.columns:
        df["semantic_type"] = None
        changed = True
    if changed:
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Normalized parquet columns → {parquet_path}")


def extract_llm_cuis(eval_df: pd.DataFrame) -> list:
    if eval_df is None or eval_df.empty:
        return []
    try:
        sup = eval_df["supervised_ranking"].iloc[0]
        candidates = sup.get("candidates") or [] if isinstance(sup, dict) else (sup.candidates or [])
        return [
            (c.get("CUI") if isinstance(c, dict) else c.CUI)
            for c in candidates
            if (c.get("CUI") if isinstance(c, dict) else c.CUI)
        ]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Ontology-to-ontology evaluation via RAG pipeline")
    parser.add_argument(
        "--mapping_file",
        default="Evaluation/OntoMapping_benchmark/UMLS_mappings/mapping_ICD10_LNC.csv",
        help="Pre-computed cross-ontology mapping CSV (from UMLS_mappings/)",
    )
    parser.add_argument("--source_onto", default="ICD10", help="Source ontology column prefix")
    parser.add_argument("--target_onto", default="LNC", help="Target ontology to embed and index")
    parser.add_argument("--max_length", type=int, default=25, help="SapBERT tokenisation max length")
    parser.add_argument("--sampling_ratio", type=float, default=0.05, help="Fraction of rows to evaluate")
    parser.add_argument("--k", type=int, default=50, help="FAISS nearest-neighbour count")
    parser.add_argument("--t", type=float, default=0.7, help="Cosine similarity threshold")
    parser.add_argument("--llm_model", default="gpt-oss:20b", help="Evaluator LLM model name")
    parser.add_argument("--results_dir", default="results/OntoMapping_benchmark/onto_onto/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── 1. Load ground-truth mapping ──────────────────────────────────────────
    with timer("Loading mapping table"):
        df = load_mapping_csv(args.mapping_file)
    logger.info(f"Mapping table: {df.shape}  CUI index, cols: {list(df.columns)}")

    # ── 2. Sample ─────────────────────────────────────────────────────────────
    n_sample = max(1, int(np.floor(args.sampling_ratio * len(df))))
    sample_df = df.sample(n=n_sample, random_state=args.seed)
    logger.info(f"Sampled {n_sample} / {len(df)} rows ({args.sampling_ratio * 100:.0f}%)")

    # ── 3. Embed target ontology from CONSO ───────────────────────────────────
    with timer(f"Embedding '{args.target_onto}' from CONSO (MAX_LENGTH={args.max_length})"):
        umls_sapbert_embeddings(vocabularies=args.target_onto, MAX_LENGTH=args.max_length)

    emb_path = f"UMLS_mapper/data/raw/text_embs_{args.max_length}_{args.target_onto}.parquet"

    # Fix column names before indexing: CONSO uses 'Name', search engine expects 'concept_name'
    normalize_parquet_columns(emb_path)

    # ── 4. Build & store FAISS index ──────────────────────────────────────────
    with timer(f"Building FAISS index for '{args.target_onto}'"):
        create_target_faiss(emb_path, args.target_onto, args.max_length)

    # ── 5. Reset singleton so RAGMapper reloads with the new index ────────────
    _use_module._umls_search_engine_instance = None

    faiss_path = f"UMLS_mapper/data/processed/faiss_index_{args.max_length}.bin"
    meta_path = f"UMLS_mapper/data/processed/metadata_{args.max_length}.csv"

    # ── 6. Initialise RAGMapper ───────────────────────────────────────────────
    source_name_col = f"{args.source_onto}_names"
    queries = sample_df[source_name_col].apply(lambda names: max(names, key=len))

    rag = RAGMapper(
        var_list=list(queries),
        var_desc=list(queries),
        FAISS_INDEX_PATH=faiss_path,
        METADATA_PATH=meta_path,
        k=args.k,
        t=args.t,
    )

    # ── 7. Run RAG pipeline ───────────────────────────────────────────────────
    records = []
    for i, (row_idx, row) in enumerate(sample_df.iterrows()):
        query = queries.loc[row_idx]
        gt_cui = row_idx  # CUI is the DataFrame index
        logger.info(f"[{i + 1}/{n_sample}] {query!r}")

        try:
            candidates_df, eval_df = rag.evaluate(query, var_desc=query, llm_model=args.llm_model)
            faiss_cuis = candidates_df["CUI"].tolist() if isinstance(candidates_df, pd.DataFrame) else []
            llm_cuis = extract_llm_cuis(eval_df)
        except Exception as e:
            logger.warning(f"Failed for {query!r}: {e}")
            faiss_cuis, llm_cuis = [], []

        records.append({
            "query": query,
            "gt_cui": gt_cui,
            "gt_cui_list": [gt_cui],
            "faiss_cuis": faiss_cuis,
            "llm_cuis": llm_cuis,
        })

    results = pd.DataFrame(records)

    # ── 8. Save results ───────────────────────────────────────────────────────
    out_path = os.path.join(
        args.results_dir, f"eval_{args.source_onto}_{args.target_onto}.csv"
    )
    results.to_csv(out_path, index=False)
    logger.info(f"Results saved → {out_path}")

    # ── 9. Score (CUI-based) ──────────────────────────────────────────────────
    print("\n=== FAISS Retrieval — Recall / Precision / MRR @K (CUI-based) ===")
    faiss_metrics = ranking_report(
        results, target_col="gt_cui_list", pred_col="faiss_cuis", ks=[1, 2, 3, 4, 5]
    )
    print(faiss_metrics.to_string())

    print("\n=== LLM Supervisor — Recall / Precision / MRR @K (CUI-based) ===")
    llm_metrics = ranking_report(
        results, target_col="gt_cui_list", pred_col="llm_cuis", ks=[1, 2, 3, 4, 5]
    )
    print(llm_metrics.to_string())


if __name__ == "__main__":
    main()