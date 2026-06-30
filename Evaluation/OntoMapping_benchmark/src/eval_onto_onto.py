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

import matplotlib.pyplot as plt
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


_CB_BLUE       = "#0072B2"
_CB_VERMILLION = "#D55E00"


def plot_onto_onto_results(
    results: pd.DataFrame, ks: list, source_onto: str, target_onto: str, save_dir: str
) -> None:
    """Plot Recall@K and MRR@K curves for FAISS retrieval vs LLM Supervisor."""
    faiss_rep = ranking_report(results, "gt_cui_list", "faiss_cuis", ks=ks)
    llm_rep   = ranking_report(results, "gt_cui_list", "llm_cuis",   ks=ks)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"Ontology mapping: {source_onto} → {target_onto}  (n = {len(results)})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, metric in zip(axes, ["Recall@K", "MRR@K"]):
        fv = [faiss_rep[metric][k] for k in ks]
        lv = [llm_rep[metric][k]   for k in ks]

        ax.plot(ks, fv, marker="o", linewidth=2.5, markersize=8,
                color=_CB_BLUE,       label="SapBERT retrieval")
        ax.plot(ks, lv, marker="s", linewidth=2.5, markersize=8, linestyle="--",
                color=_CB_VERMILLION, label="LLM Supervisor")

        for k, v in zip(ks, fv):
            ax.text(k, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color=_CB_BLUE, fontweight="bold")
        for k, v in zip(ks, lv):
            ax.text(k, v - 0.04, f"{v:.2f}", ha="center", va="top",
                    fontsize=8.5, color=_CB_VERMILLION, fontweight="bold")

        ax.set_title(metric, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("K", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_xticks(ks)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=10, framealpha=0.9, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#aaaaaa")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"onto_onto_{source_onto}_{target_onto}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved → {save_path}")
    plt.show()


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
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Save intermediate results every N rows")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip evaluation; load existing results CSV and plot")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Plot-only shortcut ────────────────────────────────────────────────────
    if args.plot_only:
        out_path = os.path.join(args.results_dir, f"eval_{args.source_onto}_{args.target_onto}.csv")
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"No results file found at {out_path}. Run without --plot_only first.")
        results = pd.read_csv(out_path)
        for col in ("gt_cui_list", "faiss_cuis", "llm_cuis"):
            if col in results.columns:
                results[col] = results[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        logger.info(f"Loaded {len(results)} rows from {out_path}")
        plot_onto_onto_results(results, ks=[1, 2, 3, 4, 5],
                               source_onto=args.source_onto, target_onto=args.target_onto,
                               save_dir=args.results_dir)
        return

    # ── 1. Load ground-truth mapping ──────────────────────────────────────────
    with timer("Loading mapping table"):
        df = load_mapping_csv(args.mapping_file)
    logger.info(f"Mapping table: {df.shape}  CUI index, cols: {list(df.columns)}")

    # ── 2. Sample ─────────────────────────────────────────────────────────────
    n_sample = max(1, int(np.floor(args.sampling_ratio * len(df))))
    sample_df = df.sample(n=n_sample, random_state=args.seed)
    logger.info(f"Sampled {n_sample} / {len(df)} rows ({args.sampling_ratio * 100:.0f}%)")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, f"eval_{args.source_onto}_{args.target_onto}.csv")
    records: list = []
    completed_cuis: set = set()
    if os.path.exists(out_path):
        try:
            ckpt = pd.read_csv(out_path)
            for col in ("gt_cui_list", "faiss_cuis", "llm_cuis"):
                if col in ckpt.columns:
                    ckpt[col] = ckpt[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            records = ckpt.to_dict("records")
            completed_cuis = set(ckpt["gt_cui"].astype(str).tolist())
            logger.info(f"Checkpoint: {len(completed_cuis)} rows already done, resuming.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e} — starting fresh.")
    remaining_sample_df = sample_df[~sample_df.index.astype(str).isin(completed_cuis)]
    logger.info(f"Remaining to process: {len(remaining_sample_df)} / {n_sample}")

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

    if len(remaining_sample_df) > 0:
        rag = RAGMapper(
            var_list=list(queries),
            var_desc=list(queries),
            FAISS_INDEX_PATH=faiss_path,
            METADATA_PATH=meta_path,
            k=args.k,
            t=args.t,
        )

        # ── 7. Run RAG pipeline ───────────────────────────────────────────────────
        for idx, (row_idx, row) in enumerate(remaining_sample_df.iterrows()):
            query = queries.loc[row_idx]
            gt_cui = row_idx  # CUI is the DataFrame index
            logger.info(f"[{len(completed_cuis) + idx + 1}/{n_sample}] {query!r}")

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

            if (idx + 1) % args.checkpoint_every == 0:
                pd.DataFrame(records).to_csv(out_path, index=False)
                logger.info(f"Checkpoint saved ({len(records)} rows) → {out_path}")

    results = pd.DataFrame(records)

    # ── 8. Save results ───────────────────────────────────────────────────────
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

    # ── 10. Plot ──────────────────────────────────────────────────────────────
    plot_onto_onto_results(results, ks=[1, 2, 3, 4, 5],
                           source_onto=args.source_onto, target_onto=args.target_onto,
                           save_dir=args.results_dir)


if __name__ == "__main__":
    main()