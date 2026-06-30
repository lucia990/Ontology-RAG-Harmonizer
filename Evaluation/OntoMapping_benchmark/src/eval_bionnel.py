"""
BioNNEL entity linking evaluation.

Loads a sample from the BioNNEL test TSV, downloads and embeds the BioNNE-L
vocabulary from HuggingFace, builds a FAISS index, runs the full RAG pipeline
on each entity mention, and reports CUI-based Recall@K, Precision@K, and
MRR@K for both SapBERT retrieval and the LLM supervisor.

Run from repo root:
  python Evaluation/OntoMapping_benchmark/src/eval_bionnel.py --sample_size 50
"""

import argparse
import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt
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


_CB_BLUE      = "#0072B2"
_CB_VERMILLION = "#D55E00"


def plot_bionnel_results(results: pd.DataFrame, ks: list, save_dir: str) -> None:
    """Plot Recall@K and MRR@K curves for SapBERT retrieval vs LLM Supervisor."""
    faiss_rep = ranking_report(results, "gt_cui_list", "faiss_cuis", ks=ks)
    llm_rep   = ranking_report(results, "gt_cui_list", "llm_cuis",   ks=ks)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"BioNNEL Entity Linking  (n = {len(results)})",
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
    save_path = os.path.join(save_dir, "bionnel_metrics.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Plot saved → {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="BioNNEL entity linking evaluation via RAG pipeline")
    parser.add_argument(
        "--test_file",
        default="Evaluation/OntoMapping_benchmark/DATA/bionnel_en_test.tsv",
        help="BioNNEL test TSV (columns: document_id, text, entity_type, spans, UMLS_CUI)",
    )
    parser.add_argument("--sample_size", type=int, default=100, help="Number of rows to evaluate")
    parser.add_argument("--max_length", type=int, default=25, help="SapBERT tokenisation max length")
    parser.add_argument("--k", type=int, default=50, help="FAISS nearest-neighbour count")
    parser.add_argument("--t", type=float, default=0.7, help="Cosine similarity threshold")
    parser.add_argument("--llm_model", default="gpt-oss:20b", help="Evaluator LLM model name")
    parser.add_argument("--results_dir", default="results/OntoMapping_benchmark/bionnel/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Save intermediate results every N rows")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip evaluation; load existing results CSV and plot")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Plot-only shortcut ────────────────────────────────────────────────────
    if args.plot_only:
        out_path = os.path.join(args.results_dir, "eval_bionnel.csv")
        if not os.path.exists(out_path):
            raise FileNotFoundError(f"No results file found at {out_path}. Run without --plot_only first.")
        results = pd.read_csv(out_path)
        for col in ("gt_cui_list", "faiss_cuis", "llm_cuis"):
            if col in results.columns:
                results[col] = results[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        logger.info(f"Loaded {len(results)} rows from {out_path}")
        plot_bionnel_results(results, ks=[1, 2, 3, 4, 5], save_dir=args.results_dir)
        return

    # ── 1. Load & sample test data ────────────────────────────────────────────
    with timer("Loading BioNNEL test data"):
        test_df = pd.read_csv(args.test_file, sep="\t")
    logger.info(f"Test set: {test_df.shape}  cols: {list(test_df.columns)}")

    deduped_df = test_df.drop_duplicates(subset="text").sample(frac=1, random_state=args.seed)
    n_sample = min(args.sample_size, len(deduped_df))
    sample_df = deduped_df.iloc[:n_sample].reset_index(drop=True)
    logger.info(f"Sample: {n_sample} unique mentions (from {len(test_df)} total, {len(deduped_df)} unique)")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, "eval_bionnel.csv")
    records: list = []
    completed_texts: set = set()
    if os.path.exists(out_path):
        try:
            ckpt = pd.read_csv(out_path)
            for col in ("gt_cui_list", "faiss_cuis", "llm_cuis"):
                if col in ckpt.columns:
                    ckpt[col] = ckpt[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            records = ckpt.to_dict("records")
            completed_texts = set(ckpt["text"].tolist())
            logger.info(f"Checkpoint: {len(completed_texts)} rows already done, resuming.")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e} — starting fresh.")
    remaining_df = sample_df[~sample_df["text"].isin(completed_texts)].reset_index(drop=True)
    logger.info(f"Remaining to process: {len(remaining_df)} / {n_sample}")

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
    faiss_path = f"UMLS_mapper/data/processed/faiss_index_{args.max_length}.bin"
    faiss_is_fresh = (
        Path(faiss_path).exists()
        and emb_path.exists()
        and Path(faiss_path).stat().st_mtime >= emb_path.stat().st_mtime
    )
    if faiss_is_fresh:
        logger.info(f"FAISS index is up to date — skipping rebuild.")
    else:
        with timer("Building FAISS index for BioNNE-L vocabulary"):
            create_target_faiss(str(emb_path), VOCAB_NAME, args.max_length)

    # ── 4. Reset singleton so RAGMapper reloads with the new index ────────────
    _use_module._umls_search_engine_instance = None

    meta_path = f"UMLS_mapper/data/processed/metadata_{args.max_length}.csv"

    # ── 5. Initialise RAGMapper ───────────────────────────────────────────────
    if len(remaining_df) > 0:
        rag = RAGMapper(
            var_list=[],
            var_desc=[],
            FAISS_INDEX_PATH=faiss_path,
            METADATA_PATH=meta_path,
            k=args.k,
            t=args.t,
        )

        # ── 6. Run RAG pipeline ───────────────────────────────────────────────────
        for idx, row in enumerate(remaining_df.itertuples(index=False)):
            query = row.text
            var_desc = row.entity_type
            gt_cui = row.UMLS_CUI
            logger.info(f"[{len(completed_texts) + idx + 1}/{n_sample}] {query!r}  ({var_desc})")

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

            if (idx + 1) % args.checkpoint_every == 0:
                pd.DataFrame(records).to_csv(out_path, index=False)
                logger.info(f"Checkpoint saved ({len(records)} rows) → {out_path}")

    results = pd.DataFrame(records)

    # ── 7. Save results ───────────────────────────────────────────────────────
    results.to_csv(out_path, index=False)
    logger.info(f"Results saved → {out_path}")

    # ── 8. Score (CUI-based) ──────────────────────────────────────────────────
    print("\n=== SapBERT Retrieval — Recall / Precision / MRR @K (CUI-based) ===")
    faiss_metrics = ranking_report(
        results, target_col="gt_cui_list", pred_col="faiss_cuis", ks=[1, 2, 3, 4, 5]
    )
    print(faiss_metrics.to_string())

    print("\n=== LLM Supervisor — Recall / Precision / MRR @K (CUI-based) ===")
    llm_metrics = ranking_report(
        results, target_col="gt_cui_list", pred_col="llm_cuis", ks=[1, 2, 3, 4, 5]
    )
    print(llm_metrics.to_string())

    # ── 9. Plot ───────────────────────────────────────────────────────────────
    plot_bionnel_results(results, ks=[1, 2, 3, 4, 5], save_dir=args.results_dir)


if __name__ == "__main__":
    main()