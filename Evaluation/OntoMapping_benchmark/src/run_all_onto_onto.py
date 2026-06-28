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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Evaluation.OntoMapping_benchmark.src.compute_scores import ranking_report

MAPPINGS_DIR = "Evaluation/OntoMapping_benchmark/UMLS_mappings"
RESULTS_DIR = "results/OntoMapping_benchmark/onto_onto"
EVAL_SCRIPT = "Evaluation/OntoMapping_benchmark/src/eval_onto_onto.py"

FAISS_KS = [1, 2, 3, 4, 5]
LLM_KS = [1, 2, 3]


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_mappings() -> list[dict]:
    files = sorted(glob.glob(os.path.join(MAPPINGS_DIR, "mapping_*.csv")))
    pairs = []
    for f in files:
        m = re.match(r"mapping_(.+?)_(.+?)\.csv", os.path.basename(f))
        if m:
            pairs.append({"source": m.group(1), "target": m.group(2), "file": f})
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
        "--k", str(max(FAISS_KS)),
        "--t", str(args.t),
        "--llm_model", args.llm_model,
        "--max_length", str(args.max_length),
        "--results_dir", RESULTS_DIR,
        "--seed", str(args.seed),
    ]
    sep = "=" * 60
    print(f"\n{sep}\nRunning: {pair['source']} → {pair['target']}\n{sep}")
    result = subprocess.run(cmd, check=False)
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

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(pairs) * 1.6), 5))
    fig.suptitle("Onto-to-Onto Benchmark — All Combinations", fontsize=13)

    for ax, metric in zip(axes, ["Recall@K", "MRR@K"]):
        faiss_k = max(FAISS_KS)
        llm_k = max(LLM_KS)
        faiss_scores = [all_metrics[p]["faiss"][metric].get(faiss_k, 0) for p in pairs]
        llm_scores   = [all_metrics[p]["llm"][metric].get(llm_k, 0)   for p in pairs]

        ax.bar(x - width / 2, faiss_scores, width, label=f"FAISS @{faiss_k}",        color="#378ADD")
        ax.bar(x + width / 2, llm_scores,   width, label=f"LLM Supervisor @{llm_k}", color="#D4537E")
        ax.set_title(metric, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_curves(all_metrics: dict, save_path: str | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Metric Curves by Vocabulary Pair (FAISS)", fontsize=13)
    colors = ["#378ADD", "#D4537E", "#1D9E75", "#EF9F27", "#7F77DD", "#E07020", "#888888"]
    metric_keys = ["Recall@K", "Precision@K", "MRR@K"]

    for ax, metric in zip(axes, metric_keys):
        for i, (label, metrics) in enumerate(all_metrics.items()):
            ks = sorted(metrics["faiss"][metric].keys())
            ys = [metrics["faiss"][metric][k] for k in ks]
            ax.plot(ks, ys, marker="o", markersize=5, linewidth=2,
                    label=label, color=colors[i % len(colors)])
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("K")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(FAISS_KS)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(all_metrics), 4),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def plot_distributions(all_sample_dfs: dict, save_path: str | None = None) -> None:
    labels = list(all_sample_dfs.keys())
    score_cols = [
        ("faiss_recall", f"FAISS Recall@{max(FAISS_KS)}"),
        ("faiss_mrr",    f"FAISS MRR@{max(FAISS_KS)}"),
        ("llm_recall",   f"LLM Recall@{max(LLM_KS)}"),
        ("llm_mrr",      f"LLM MRR@{max(LLM_KS)}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(labels) * 1.4), 9))
    fig.suptitle("Per-Sample Score Distributions across Vocabulary Combinations", fontsize=13)

    for (col, title), ax in zip(score_cols, axes.flat):
        data = [all_sample_dfs[lbl][col].dropna().values for lbl in labels]
        data = [d if len(d) > 1 else np.array([0.0, 0.0]) for d in data]  # violinplot needs ≥2 pts
        parts = ax.violinplot(data, positions=range(len(labels)), showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("#7F77DD")
            pc.set_alpha(0.55)
        parts["cmedians"].set_color("#D4537E")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

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
    parser.add_argument("--t",              type=float, default=0.6)
    parser.add_argument("--max_length",     type=int,   default=25)
    parser.add_argument("--llm_model",      default="gpt-oss:20b")
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

        print(f"\n[{label}] FAISS @{max(FAISS_KS)} — "
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
