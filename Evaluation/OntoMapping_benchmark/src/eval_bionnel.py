"""
BioNNEL entity linking evaluation.

Loads a sample from the BioNNEL test TSV, downloads and embeds the BioNNE-L
vocabulary from HuggingFace, builds a FAISS index, runs the full RAG pipeline
on each entity mention, and reports CUI-based Recall@K, Precision@K, and
MRR@K for both FAISS retrieval and the LLM supervisor.

Run from repo root:
  python Evaluation/OntoMapping_benchmark/src/eval_bionnel.py --sample_size 50
"""

import argparse
import os
from pathlib import Path

import pandas as pd

import UMLS_mapper.src.umls_search_engine as _use_module
from Evaluation.OntoMapping_benchmark.src.bio_nnel_pipeline import (
    embed_bionnel_umls, load_bionnel_eng_umls,
)
from Evaluation.OntoMapping_benchmark.src.compute_scores import ranking_report
from Evaluation.OntoMapping_benchmark.src.OM_pipeline import (
    create_target_faiss, setup_logging, timer,
)
from RAG_mapper.src.RAG_mapper import RAGMapper

logger = setup_logging()

VOCAB_NAME = "BIO-NNEL"


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
    parser = argparse.ArgumentParser(description="BioNNEL entity linking evaluation via RAG pipeline")
    parser.add_argument(
        "--test_file",
        default="Evaluation/OntoMapping_benchmark/DATA/bionnel_en_test.tsv",
        help="BioNNEL test TSV (columns: document_id, text, entity_type, spans, UMLS_CUI)",
    )
    parser.add_argument("--sample_size", type=int, default=50, help="Number of rows to evaluate")
    parser.add_argument("--max_length", type=int, default=25, help="SapBERT tokenisation max length")
    parser.add_argument("--k", type=int, default=5, help="FAISS nearest-neighbour count")
    parser.add_argument("--t", type=float, default=0.6, help="Cosine similarity threshold")
    parser.add_argument("--llm_model", default="gpt-oss:20b", help="Evaluator LLM model name")
    parser.add_argument("--results_dir", default="results/OntoMapping_benchmark/bionnel/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── 1. Load & sample test data ────────────────────────────────────────────
    with timer("Loading BioNNEL test data"):
        test_df = pd.read_csv(args.test_file, sep="\t")
    logger.info(f"Test set: {test_df.shape}  cols: {list(test_df.columns)}")

    n_sample = min(args.sample_size, len(test_df))
    sample_df = test_df.sample(n=n_sample, random_state=args.seed).reset_index(drop=True)
    logger.info(f"Sample: {n_sample} rows")

    # ── 2. Embed BioNNE-L vocabulary (skips if parquet already exists) ────────
    emb_path = Path(f"UMLS_mapper/data/raw/text_embs_{args.max_length}_{VOCAB_NAME}.parquet")

    if emb_path.exists():
        logger.info(f"BioNNEL embeddings already exist at {emb_path} — skipping embedding step.")
    else:
        with timer("Loading BioNNE-L vocabulary from HuggingFace"):
            umls_df = load_bionnel_eng_umls()
        logger.info(f"BioNNE-L vocabulary: {umls_df.shape}")

        with timer(f"Embedding BioNNE-L vocabulary (MAX_LENGTH={args.max_length})"):
            embed_bionnel_umls(umls_df, MAX_LENGTH=args.max_length, out_dir="UMLS_mapper/data/raw")

    # ── 3. Build & store FAISS index ──────────────────────────────────────────
    # Always rebuild to ensure the index matches the current embeddings file.
    with timer("Building FAISS index for BioNNE-L vocabulary"):
        create_target_faiss(str(emb_path), VOCAB_NAME, args.max_length)

    # ── 4. Reset singleton so RAGMapper reloads with the new index ────────────
    _use_module._umls_search_engine_instance = None

    faiss_path = f"UMLS_mapper/data/processed/faiss_index_{args.max_length}.bin"
    meta_path = f"UMLS_mapper/data/processed/metadata_{args.max_length}.csv"

    # ── 5. Initialise RAGMapper ───────────────────────────────────────────────
    rag = RAGMapper(
        var_list=[],
        var_desc=[],
        FAISS_INDEX_PATH=faiss_path,
        METADATA_PATH=meta_path,
        k=args.k,
        t=args.t,
    )

    # ── 6. Run RAG pipeline ───────────────────────────────────────────────────
    records = []
    for i, row in sample_df.iterrows():
        query = row["text"]
        var_desc = row["entity_type"]
        gt_cui = row["UMLS_CUI"]
        logger.info(f"[{i + 1}/{n_sample}] {query!r}  ({var_desc})")

        try:
            candidates_df, eval_df = rag.evaluate(query, var_desc=var_desc, llm_model=args.llm_model)
            faiss_cuis = candidates_df["CUI"].tolist() if isinstance(candidates_df, pd.DataFrame) else []
            llm_cuis = extract_llm_cuis(eval_df)
        except Exception as e:
            logger.warning(f"Failed for {query!r}: {e}")
            faiss_cuis, llm_cuis = [], []

        records.append({
            "text": query,
            "entity_type": var_desc,
            "gt_cui": gt_cui,
            "gt_cui_list": [gt_cui],
            "faiss_cuis": faiss_cuis,
            "llm_cuis": llm_cuis,
        })

    results = pd.DataFrame(records)

    # ── 7. Save results ───────────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, "eval_bionnel.csv")
    results.to_csv(out_path, index=False)
    logger.info(f"Results saved → {out_path}")

    # ── 8. Score (CUI-based) ──────────────────────────────────────────────────
    print("\n=== FAISS Retrieval — Recall / Precision / MRR @K (CUI-based) ===")
    faiss_metrics = ranking_report(
        results, target_col="gt_cui_list", pred_col="faiss_cuis", ks=[1, 2, 3, 4, 5]
    )
    print(faiss_metrics.to_string())

    print("\n=== LLM Supervisor — Recall / Precision / MRR @K (CUI-based) ===")
    llm_metrics = ranking_report(
        results, target_col="gt_cui_list", pred_col="llm_cuis", ks=[1, 2, 3]
    )
    print(llm_metrics.to_string())


if __name__ == "__main__":
    main()