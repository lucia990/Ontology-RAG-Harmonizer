import pandas as pd
import os
import time
from tqdm import tqdm
from Evaluation.OntoMapping_benchmark.src.MEL_pipeline import SemanticMapper


def to_dict(obj):
    """Safely convert Pydantic model or plain dict to dict."""
    if isinstance(obj, dict):
        return obj
    return obj.model_dump()

def save_intermediate_results(result_tuple,
                              candidate_file="results/OntoMapping_benchmark/Clinical_var/Breast_cancer/all_candidates-2.csv",
                              review_file="results/OntoMapping_benchmark/Clinical_var/Breast_cancer/mapping_reviews-2.csv"):

    # Unpack tuple
    candidate_df, review_df = result_tuple

    # -----------------------------
    # 1. Save candidate dataframe
    # -----------------------------
    if not os.path.exists(candidate_file):
        candidate_df.to_csv(candidate_file, index=False)
    else:
        candidate_df.to_csv(candidate_file,
                            mode="a",
                            header=False,
                            index=False)

    # -----------------------------
    # 2. Flatten review dataframe
    # -----------------------------
    rows = []

    for _, row in review_df.iterrows():

        rag_review = row["rag_review"]
        supervised = row["supervised_ranking"]

        variable = supervised.get("variable", None)

        # Save one row per ranked candidate
        for cand in supervised.get("candidates", []):

            rows.append({
                "variable": variable,

                "review_status": rag_review.get("status"),
                "review_explanation": rag_review.get("explanation"),

                "rank": cand.get("rank"),
                "AI_code": cand.get("AI_code"),
                "AI_name": cand.get("AI_name"),
                "confidence": cand.get("confidence"),
                "CUI": cand.get("CUI")
            })

    review_out = pd.DataFrame(rows)

    # -----------------------------
    # 3. Append review results
    # -----------------------------
    if not os.path.exists(review_file):
        review_out.to_csv(review_file, index=False)
    else:
        review_out.to_csv(review_file,
                          mode="a",
                          header=False,
                          index=False)

def save_intermediate_results_cui(
    result_tuple,
    candidate_file="results/OntoMapping_benchmark/Clinical_var/MicrobAIome/all_candidates-3.csv",
    review_file="results/OntoMapping_benchmark/Clinical_var/MicrobAIome/mapping_reviews_cui-3.csv"
):
    candidate_df, review_df = result_tuple

    # -------------------------
    # 1. Save candidate dataframe
    # -------------------------
    if not os.path.exists(candidate_file):
        candidate_df.to_csv(candidate_file, index=False)
    else:
        candidate_df.to_csv(candidate_file, mode="a", header=False, index=False)

    # -------------------------
    # 2. Flatten rows — CUI-centric schema
    # -------------------------
    rows = []

    for _, row in review_df.iterrows():
        rag_review  = to_dict(row["rag_review"])
        supervised  = to_dict(row["supervised_ranking"])

        variable = supervised.get("variable")

        for cand in (supervised.get("candidates") or []):
            cand = to_dict(cand)

            rep_names = cand.get("representative_names") or []
            if isinstance(rep_names, list):
                rep_names = " | ".join(rep_names)

            rows.append({
                "variable":            variable,
                "review_status":       rag_review.get("status"),
                "review_explanation":  rag_review.get("explanation"),
                "rank":                cand.get("rank"),
                "CUI":                 cand.get("CUI"),
                "CUI_label":           cand.get("CUI_label"),
                "representative_names": rep_names,
                "confidence":          cand.get("confidence"),
                "reasoning":           cand.get("reasoning"),
            })

    review_out = pd.DataFrame(rows)

    # -------------------------
    # 3. Append review results
    # -------------------------
    if not os.path.exists(review_file):
        review_out.to_csv(review_file, index=False)
    else:
        review_out.to_csv(review_file, mode="a", header=False, index=False)

if __name__ == "__main__":
    microbaiome = pd.read_csv('Evaluation/OntoMapping_benchmark/DATA/MicrobAIome.csv')[['schemanode_name', 'schemanode_description', 'ontology_id']]

    # output files
    OUTPUT_DIR = "results/OntoMapping_benchmark/Clinical_var/MicrobAIome"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidate_file = f"{OUTPUT_DIR}/all_candidates-2.csv"
    review_file = f"{OUTPUT_DIR}/mapping_reviews-2.csv"
    failure_file = f"{OUTPUT_DIR}/failed_variables-2.txt"

    # Hyperparameters
    k = 10
    t = 0.6
    MAX_LENGTH = 25

    # main loop
    start_time = time.time()
    for _, row in tqdm(microbaiome.iterrows()):
        var = row['schemanode_name']
        var_desc= row['schemanode_description']
        gt = row['ontology_id']

        try:
            sm = SemanticMapper(var= var, var_desc= var_desc, k=k, t=t, MAX_LENGTH=MAX_LENGTH)
            res = sm.map_var()
            save_intermediate_results(res, candidate_file, review_file)
            print(f"[OK] {row['schemanode_name']}")

        except Exception as e:
            print(f"[FAILED] {row['schemanode_name']} -> {e}")

            with open(failure_file, "a") as f:
                f.write(f"{row['schemanode_name']} :: {str(e)}\n")


    print(f'Mapping completed in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')


