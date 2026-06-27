import ast
import pandas as pd
import logging
import numpy as np
from typing import Optional, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def setup_logging(log_file: str = "evaluation.log") -> logging.Logger:
    """Configure a logger that writes to both console and a log file."""
    logger = logging.getLogger("compute_scores")
    logger.setLevel(logging.DEBUG)

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
    return logger


logger = setup_logging()

def parse_to_list_of_strings(s):
    result = ast.literal_eval(s)
    return [str(item) for item in result]

def clean_ai_code(x):
    # Case 1: NaN → keep as is (or return None)
    if pd.isna(x):
        return None

    # Case 2: float like 12345.0 → convert to int → string
    if isinstance(x, float):
        if x.is_integer():
            return str(int(x))
        else:
            return str(x)  # keep decimal if meaningful

    # Case 3: already string → return as is
    return str(x)

def open_res_table(vocabularies: str):
    vocabularies_list = vocabularies.split()
    source_onto = vocabularies_list[0]
    target_onto = vocabularies_list[1]
    logging.info("Opening mapping table")
    try:
        res_df = pd.read_csv(f'results/OntoMapping_benchmark/rag_output_25_{source_onto}_{target_onto}.csv')
        res_df[f'{target_onto}_ids'] = res_df[f'{target_onto}_ids'].apply(lambda x: parse_to_list_of_strings(x))
        res_df['se_codes'] = res_df['se_codes'].apply(lambda x: parse_to_list_of_strings(x))
        res_df['AI_code'] =  res_df['AI_code'].apply(lambda x: str(int(x)) if isinstance(x, float) and not pd.isna(x) else str(x) if not pd.isna(x) else None)
        logging.info(f"Mapping table: {res_df.shape} with columns: {res_df.columns}")
        return res_df
    except FileNotFoundError as e:
        logger.error(f"Error loading results: {e}", exc_info=True)
        return None



def recall_at_k(df: pd.DataFrame, target_col: str, pred_col: str, k: int) -> float:
    """
    Compute mean Recall@K across all rows.

    Recall@K = |relevant ∩ top-k predicted| / |relevant|

    Args:
        df:         DataFrame with ground-truth and prediction columns.
        target_col: Column of List[str] ground-truth IDs.
        pred_col:   Column of List[str] ranked candidate IDs.
        k:          Cut-off rank.

    Returns:
        Mean Recall@K over all rows.
    """
    def _row_recall(row):
        relevant = set(row[target_col])
        if not relevant:
            return 0.0
        top_k = set(row[pred_col][:k])
        return len(relevant & top_k) / len(relevant)

    return df.apply(_row_recall, axis=1).mean()


def precision_at_k(df: pd.DataFrame, target_col: str, pred_col: str, k: int) -> float:
    """
    Compute mean Precision@K across all rows.

    Precision@K = |relevant ∩ top-k predicted| / K

    Args:
        df:         DataFrame with ground-truth and prediction columns.
        target_col: Column of List[str] ground-truth IDs.
        pred_col:   Column of List[str] ranked candidate IDs.
        k:          Cut-off rank.

    Returns:
        Mean Precision@K over all rows.
    """
    def _row_precision(row):
        relevant = set(row[target_col])
        top_k = row[pred_col][:k]
        if not top_k:
            return 0.0
        hits = sum(1 for c in top_k if c in relevant)
        return hits / k

    return df.apply(_row_precision, axis=1).mean()


def mrr(df: pd.DataFrame, target_col: str, pred_col: str, k: Optional[int] = None) -> float:
    """
    Compute Mean Reciprocal Rank (MRR), optionally capped at rank K.

    MRR = mean over rows of (1 / rank_of_first_hit),
          where rank_of_first_hit is the 1-based position of the first
          relevant item in the ranked list (0 if none found within k).

    Args:
        df:         DataFrame with ground-truth and prediction columns.
        target_col: Column of List[str] ground-truth IDs.
        pred_col:   Column of List[str] ranked candidate IDs.
        k:          Optional cut-off rank. If None, the full list is used.

    Returns:
        MRR score.
    """
    def _row_rr(row):
        relevant = set(row[target_col])
        candidates = row[pred_col] if k is None else row[pred_col][:k]
        for rank, candidate in enumerate(candidates, start=1):
            if candidate in relevant:
                return 1.0 / rank
        return 0.0

    return df.apply(_row_rr, axis=1).mean()


def ranking_report(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    ks: list[int] = [1, 2, 3, 4, 5],
) -> pd.DataFrame:
    """
    Args:
        df:         DataFrame with ground-truth and prediction columns.
        target_col: Column of List[str] ground-truth IDs.
        pred_col:   Column of List[str] ranked candidate IDs.
        ks:         List of cut-off values.

    Returns:
        DataFrame with columns [K, Recall@K, Precision@K, MRR@K].
    """
    rows = []
    for k in ks:
        rows.append({
            "K":            k,
            "Recall@K":    recall_at_k(df, target_col, pred_col, k),
            "Precision@K": precision_at_k(df, target_col, pred_col, k),
            "MRR@K":       mrr(df, target_col, pred_col, k),
        })
    return pd.DataFrame(rows).set_index("K")


def plot_ranking_metrics(
    metrics: Union[pd.DataFrame, dict],
    vocabularies: str,
    models: list[str] | None = None,
    ks: list[int] | None = None,
    figsize: tuple[float, float] = (13, 4),
    save_path: str | None = None,
) -> None:
    """
    Plot Recall@K, Precision@K, and MRR@K curves from precomputed metrics.

    Parameters
    ----------
    metrics : pd.DataFrame or dict
        Two accepted formats:

        1. DataFrame (output of `ranking_report`):
           Index = K values, columns = ['Recall@K', 'Precision@K', 'MRR@K'].
           Pass a single model's results directly, or a dict of such DataFrames
           keyed by model name via the `models` parameter.

        2. Nested dict  { model_name: { 'Recall@K':    {k: v, ...},
                                        'Precision@K': {k: v, ...},
                                        'MRR@K':       {k: v, ...} } }
           Pass model names via `models` to control order / subset.

    models : list[str], optional
        Subset / ordering of model names when `metrics` is a dict.
        Defaults to all keys in insertion order.

    ks : list[int], optional
        K values to display on the x-axis.
        Defaults to whatever is present in the data.

    figsize : tuple, optional
        Overall figure width × height in inches. Default (13, 4).

    save_path : str, optional
        If provided, saves the figure to this path (e.g. 'metrics.png').
    """
    METRIC_KEYS = ["Recall@K", "Precision@K", "MRR@K"]
    COLORS = ["#378ADD", "#D4537E", "#1D9E75", "#EF9F27", "#7F77DD"]

    # ── Normalise input to { model: { metric: { k: value } } } ──────────────
    if isinstance(metrics, pd.DataFrame):
        # Single model passed as a DataFrame
        model_name = "Model"
        data = {model_name: {m: metrics[m].to_dict() for m in METRIC_KEYS}}
    elif isinstance(metrics, dict):
        first_val = next(iter(metrics.values()))
        if isinstance(first_val, pd.DataFrame):
            # { model_name: DataFrame }
            data = {
                name: {m: df[m].to_dict() for m in METRIC_KEYS}
                for name, df in metrics.items()
            }
        else:
            # Already nested dict
            data = metrics
    else:
        raise TypeError("`metrics` must be a pd.DataFrame or dict.")

    if models is None:
        models = list(data.keys())

    # ── Resolve K values ─────────────────────────────────────────────────────
    if ks is None:
        all_ks: set[int] = set()
        for m_data in data.values():
            for metric_vals in m_data.values():
                all_ks.update(metric_vals.keys())
        ks = sorted(all_ks)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.35)
    fig.suptitle(vocabularies, fontsize=14)
    for ax, metric in zip(axes, METRIC_KEYS):
        for i, model in enumerate(models):
            color = COLORS[i % len(COLORS)]
            y = [data[model][metric].get(k, np.nan) for k in ks]
            ax.plot(ks, y, marker="o", markersize=5, linewidth=2,
                    label=model, color=color)

        ax.set_title(metric, fontsize=12, fontweight="normal", pad=10)
        ax.set_xlabel("K", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(ks)
        ax.xaxis.set_tick_params(labelsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.set_tick_params(labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.5, linestyle="--", alpha=0.5)

    # Single shared legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(models), fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.08))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    vocabularies = 'ICD10 SNOMEDCT_US'
    vocabularies_list = vocabularies.split()
    target_onto = vocabularies_list[1]
    res_df = open_res_table(vocabularies)
    print(f'Number of samples: {len(res_df)}')
    metric_df = ranking_report(res_df, f'{target_onto}_ids', 'se_codes', ks = [1, 2, 3, 4, 5])
    print(metric_df)
    plot_ranking_metrics(metric_df, vocabularies)
