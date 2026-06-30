"""
Run eval_onto_onto.py for every mapping_*.csv in UMLS_mappings/ (sequentially),
collect results, and produce comparison and distribution plots.

Usage:
    # Run all evaluations then plot
    python Evaluation/OntoMapping_benchmark/src/run_all_onto_onto.py

    # Skip re-running evals (plot from existing results only)
    python Evaluation/OntoMapping_benchmark/src/run_all_onto_onto.py --skip_eval
"""

import argparse
import ast
import glob
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.OntoMapping_benchmark.src.compute_scores import ranking_report

# Repo root = 4 levels up from this file (Schema_DH/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

MAPPINGS_DIR = "Evaluation/OntoMapping_benchmark/UMLS_mappings"
RESULTS_DIR = "results/OntoMapping_benchmark/onto_onto"
EVAL_SCRIPT = "Evaluation/OntoMapping_benchmark/src/eval_onto_onto.py"

# Vocabulary names that may contain underscores — longest first so greedy prefix match works
KNOWN_VOCABS = sorted(["SNOMEDCT_US", "ICD10", "LNC", "NCBI"], key=len, reverse=True)

FAISS_RETRIEVE_K = 50   # candidates retrieved by SapBERT/FAISS backbone
EVAL_KS = [1, 2, 3, 4, 5]  # cut-offs for Recall/Precision/MRR (both stages)
FAISS_KS = EVAL_KS
LLM_KS   = EVAL_KS

# Okabe-Ito colorblind-safe palette
CB_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#000000"]
_MARKERS   = ["o", "s", "^", "D", "v", "P", "X"]


# ── Discovery ─────────────────────────────────────────────────────────────────

def _parse_mapping_filename(basename: str) -> tuple[str, str] | None:
    name = basename.removeprefix("mapping_").removesuffix(".csv")
    for vocab in KNOWN_VOCABS:  # longest first — avoids "SNOMEDCT" matching before "SNOMEDCT_US"
        if name.startswith(vocab + "_"):
            return vocab, name[len(vocab) + 1:]
    return None


def discover_mappings() -> list[dict]:
    files = sorted(glob.glob(os.path.join(MAPPINGS_DIR, "mapping_*.csv")))
    pairs = []
    for f in files:
        parsed = _parse_mapping_filename(os.path.basename(f))
        if parsed:
            source, target = parsed
            pairs.append({"source": source, "target": target, "file": f})
    return pairs


# ── Evaluation runner ──────────────────────────────────────────────────────────

def run_eval(pair: dict, args: argparse.Namespace) -> None:
    import subprocess
    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--mapping_file", pair["file"],
        "--source_onto", pair["source"],
        "--target_onto", pair["target"],
        "--sampling_ratio", str(args.sampling_ratio),
        "--k", str(FAISS_RETRIEVE_K),
        "--t", str(args.t),
        "--llm_model", args.llm_model,
        "--max_length", str(args.max_length),
        "--results_dir", RESULTS_DIR,
        "--seed", str(args.seed),
    ]
    # Ensure the subprocess can import project modules (UMLS_mapper, RAG_mapper, etc.)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    sep = "=" * 60
    print(f"\n{sep}\nRunning: {pair['source']} → {pair['target']}\n{sep}")
    result = subprocess.run(cmd, check=False, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        print(f"WARNING: eval failed for {pair['source']} → {pair['target']}")


# ── Result loading ─────────────────────────────────────────────────────────────

def load_result(source: str, target: str) -> pd.DataFrame | None:
    path = os.path.join(RESULTS_DIR, f"eval_{source}_{target}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    for col in ("faiss_cuis", "llm_cuis", "gt_cui_list"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
    return df


# ── Per-sample scoring ─────────────────────────────────────────────────────────

def _recall(row: pd.Series, pred_col: str, k: int) -> float:
    relevant = set(row["gt_cui_list"])
    if not relevant:
        return 0.0
    return len(relevant & set(row[pred_col][:k])) / len(relevant)


def _mrr(row: pd.Series, pred_col: str, k: int) -> float:
    relevant = set(row["gt_cui_list"])
    for rank, c in enumerate(row[pred_col][:k], 1):
        if c in relevant:
            return 1.0 / rank
    return 0.0


def compute_per_sample(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    df = df.copy()
    df["faiss_recall"] = df.apply(_recall, pred_col="faiss_cuis", k=k, axis=1)
    df["faiss_mrr"]    = df.apply(_mrr,    pred_col="faiss_cuis", k=k, axis=1)
    df["llm_recall"]   = df.apply(_recall, pred_col="llm_cuis",   k=min(k, max(LLM_KS)), axis=1)
    df["llm_mrr"]      = df.apply(_mrr,    pred_col="llm_cuis",   k=min(k, max(LLM_KS)), axis=1)
    return df


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_summary(all_metrics: dict, save_path: str | None = None) -> None:
    pairs = list(all_metrics.keys())
    x = np.arange(len(pairs))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(pairs) * 1.8), 5))
    fig.suptitle("Onto-to-Onto Benchmark — All Combinations", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, ["Recall@K", "MRR@K"]):
        faiss_k = max(FAISS_KS)
        llm_k   = max(LLM_KS)
        faiss_scores = [all_metrics[p]["faiss"][metric].get(faiss_k, 0) for p in pairs]
        llm_scores   = [all_metrics[p]["llm"][metric].get(llm_k, 0)   for p in pairs]

        bars_f = ax.bar(x - width / 2, faiss_scores, width,
                        label=f"SapBERT @{faiss_k}", color=CB_PALETTE[0],
                        alpha=0.9, edgecolor="white", linewidth=0.6)
        bars_l = ax.bar(x + width / 2, llm_scores, width,
                        label=f"LLM Supervisor @{llm_k}", color=CB_PALETTE[1],
                        alpha=0.9, edgecolor="white", linewidth=0.6)

        for bar in bars_f:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
                    color=CB_PALETTE[0], fontweight="bold")
        for bar in bars_l:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
                    color=CB_PALETTE[1], fontweight="bold")

        ax.set_title(metric, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#aaaaaa")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_curves(all_metrics: dict, save_path: str | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Metric Curves by Vocabulary Pair (SapBERT)", fontsize=13, fontweight="bold")

    for ax, metric in zip(axes, ["Recall@K", "Precision@K", "MRR@K"]):
        for i, (label, metrics) in enumerate(all_metrics.items()):
            ks = sorted(metrics["faiss"][metric].keys())
            ys = [metrics["faiss"][metric][k] for k in ks]
            ax.plot(ks, ys,
                    marker=_MARKERS[i % len(_MARKERS)],
                    markersize=6, linewidth=2,
                    label=label,
                    color=CB_PALETTE[i % len(CB_PALETTE)])

        ax.set_title(metric, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("K", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(FAISS_KS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#aaaaaa")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=min(len(all_metrics), 4),
               fontsize=9, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.14))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_distributions(all_sample_dfs: dict, save_path: str | None = None) -> None:
    labels = list(all_sample_dfs.keys())
    score_cols = [
        ("faiss_recall", f"SapBERT Recall@{max(FAISS_KS)}"),
        ("faiss_mrr",    f"SapBERT MRR@{max(FAISS_KS)}"),
        ("llm_recall",   f"LLM Recall@{max(LLM_KS)}"),
        ("llm_mrr",      f"LLM MRR@{max(LLM_KS)}"),
    ]
    # SapBERT panels use blue; LLM panels use vermillion; medians in orange
    face_colors   = [CB_PALETTE[0], CB_PALETTE[0], CB_PALETTE[1], CB_PALETTE[1]]
    median_colors = [CB_PALETTE[3], CB_PALETTE[3], CB_PALETTE[3], CB_PALETTE[3]]

    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(labels) * 1.6), 9))
    fig.suptitle("Per-Sample Score Distributions across Vocabulary Combinations",
                 fontsize=13, fontweight="bold")

    for (col, title), fcolor, mcolor, ax in zip(score_cols, face_colors, median_colors, axes.flat):
        data = [all_sample_dfs[lbl][col].dropna().values for lbl in labels]
        data = [d if len(d) > 1 else np.array([0.0, 0.0]) for d in data]

        parts = ax.violinplot(data, positions=range(len(labels)),
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(fcolor)
            pc.set_alpha(0.5)
        parts["cmedians"].set_color(mcolor)
        parts["cmedians"].set_linewidth(2)
        for key in ("cbars", "cmins", "cmaxes"):
            parts[key].set_color(fcolor)
            parts[key].set_alpha(0.6)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("Score", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#aaaaaa")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all onto-onto evaluations and plot results")
    parser.add_argument("--skip_eval",      action="store_true",  help="Only plot; skip running evals")
    parser.add_argument("--sampling_ratio", type=float, default=0.05)
    parser.add_argument("--t",              type=float, default=0.7)
    parser.add_argument("--max_length",     type=int,   default=25)
    parser.add_argument("--llm_model",      default="qwen3.6")
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    pairs = discover_mappings()
    print(f"Found {len(pairs)} mapping combination(s):")
    for p in pairs:
        print(f"  {p['source']} → {p['target']}  ({p['file']})")

    if not args.skip_eval:
        for pair in pairs:
            run_eval(pair, args)

    # ── Load results & compute metrics ────────────────────────────────────────
    all_metrics: dict = {}
    all_sample_dfs: dict = {}

    for pair in pairs:
        label = f"{pair['source']}→{pair['target']}"
        df = load_result(pair["source"], pair["target"])
        if df is None:
            print(f"No result file for {label} — skipping.")
            continue

        faiss_report = ranking_report(df, "gt_cui_list", "faiss_cuis", ks=FAISS_KS)
        llm_report   = ranking_report(df, "gt_cui_list", "llm_cuis",   ks=LLM_KS)

        all_metrics[label] = {
            "faiss": {m: faiss_report[m].to_dict() for m in ["Recall@K", "Precision@K", "MRR@K"]},
            "llm":   {m: llm_report[m].to_dict()   for m in ["Recall@K", "Precision@K", "MRR@K"]},
        }
        all_sample_dfs[label] = compute_per_sample(df, k=max(FAISS_KS))

        print(f"\n[{label}] SapBERT @{max(FAISS_KS)} — "
              f"Recall={faiss_report['Recall@K'][max(FAISS_KS)]:.3f}  "
              f"MRR={faiss_report['MRR@K'][max(FAISS_KS)]:.3f}  |  "
              f"LLM @{max(LLM_KS)} — "
              f"Recall={llm_report['Recall@K'][max(LLM_KS)]:.3f}  "
              f"MRR={llm_report['MRR@K'][max(LLM_KS)]:.3f}")

    if not all_metrics:
        print("No results to plot. Run without --skip_eval first.")
        return

    metrics_path = os.path.join(RESULTS_DIR, "all_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAggregated metrics → {metrics_path}")

    plot_summary(all_metrics,    save_path=os.path.join(RESULTS_DIR, "summary_bar.png"))
    plot_curves(all_metrics,     save_path=os.path.join(RESULTS_DIR, "metric_curves.png"))
    plot_distributions(all_sample_dfs, save_path=os.path.join(RESULTS_DIR, "score_distributions.png"))


if __name__ == "__main__":
    main()
