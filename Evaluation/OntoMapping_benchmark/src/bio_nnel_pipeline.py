from datasets import load_dataset
import gc
import pandas as pd
import numpy as np
import os
import torch
from pathlib import Path
from tqdm.auto import tqdm
import time
from transformers import AutoTokenizer, AutoModel

from Evaluation.OntoMapping_benchmark.src.OM_pipeline import create_target_faiss
from RAG_mapper.src.RAG_mapper import RAGMapper




###  ~900K TARGET CONCEPTS MISSING!!!
def preprocess_umls(huggingface_path:str = "andorei/BioNNE-L", my_umls_path:str = 'UMLS_mapper/data/raw/filtered_conso_eng.csv'):
    # Load competition vocabulary
    umls = load_dataset(huggingface_path, "Vocabulary", split="train")

    # Keep english rows
    umls_eng = umls.filter(lambda batch: [l == 'ENG' for l in batch['lang']], batched=True)
    umls_df = umls_eng.to_pandas()

    # Load my umls
    my_umls = pd.read_csv(my_umls_path, index_col=False,
                           names=['CUI', 'Language', 'Status', 'LUI', 'string_type', 'SUI', 'atom_status', 'AUI',
                                  'SAUI', 'SCUI', 'SDUI', 'source', 'type', 'CODE', 'Name', 'restriction_level',
                                  'SUPPRESS', 'CVF'], low_memory=False)

    MY_KEY = 'CUI'

    BIONNEL_KEY = 'CUI'

    SEMANTIC_COL = "semantic_type"

    # Merge on CUI
    shared_cuis = set(umls_df[BIONNEL_KEY].unique())

    filtered_umls = my_umls[
    my_umls["CUI"].isin(shared_cuis)
    ].copy()

    semantic_df = (
        umls_df[[BIONNEL_KEY, SEMANTIC_COL]]
        .drop_duplicates()
        .groupby(BIONNEL_KEY)[SEMANTIC_COL]
        .apply(lambda x: "|".join(sorted(set(map(str, x)))))
        .reset_index()
    )

    filtered_umls = filtered_umls.merge(
        semantic_df,
        on=BIONNEL_KEY,
        how="left"
    )


    # Save file
    filtered_umls = filtered_umls[['CUI', 'Language', 'source', 'type', 'CODE', 'Name', 'semantic_type']].copy()
    filtered_umls.to_csv('Evaluation/OntoMapping_benchmark/DATA/filtered_bionnel_eng.csv', index = False)
    print(filtered_umls.shape)
    return filtered_umls


def load_bionnel_eng_umls(huggingface_path: str = "andorei/BioNNE-L" ) -> pd.DataFrame:
    umls = load_dataset(huggingface_path, "Vocabulary", split="train")

    # Keep english rows
    umls_eng = umls.filter(lambda batch: [l == 'ENG' for l in batch['lang']], batched=True)
    umls_df = umls_eng.to_pandas()
    return umls_df

def embed_bionnel_umls(umls_df: pd.DataFrame, MAX_LENGTH:int = 25, BATCH_SIZE: int = 128, out_dir: str = 'UMLS_mapper/data/raw') -> pd.DataFrame:
    torch.cuda.empty_cache()
    print('Load UMLS from Bio-NNEL dataset...')
    file_path = Path(f"{out_dir}/text_embs_{MAX_LENGTH}_BIO-NNEL.parquet")
    if file_path.exists():
        print(f"Embeddings for bio-nnel data with MAX_LENGTH = {MAX_LENGTH} already exist in {file_path}  Skipping...")
        return 0
    else:
        N = len(umls_df)
        print(f"Numbers of nodes to embed: {N}")
        print(f'Load embedding model...\n')
        tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                                          torch_dtype=torch.float16).cuda()

        print(f'*************** Compute embeddings with max Length: {MAX_LENGTH}***************\n')
        all_embs = []
        with torch.no_grad():
            for i in tqdm(np.arange(0, N, BATCH_SIZE)):
                print(f"\nTokenizing batch {i}...\n")
                batch_text = list(umls_df.concept_name)[i:i + BATCH_SIZE]
                toks = tokenizer.batch_encode_plus(
                    batch_text,
                    padding='max_length',
                    max_length=MAX_LENGTH,
                    truncation=True,
                    return_tensors="pt"
                )

                toks_cuda = {}
                print(f'\nEmbedding batch {i}...\n')
                for k, v in toks.items():
                    toks_cuda[k] = v.cuda()
                cls_rep = model(**toks_cuda)[0][:, 0, :]
                print(cls_rep.shape)
                all_embs.append(cls_rep.cpu().detach().numpy())

        all_embs = np.concatenate(all_embs, axis=0)

        umls_df['Text_Embs'] = [emb for emb in all_embs]
        umls_df.to_parquet(file_path,  index=False)
        print(f"------{N} Sapbert embeddings saved in {out_dir}/text_embs_{MAX_LENGTH}_BIO-NNEL.parquet")

        del model, tokenizer, all_embs
        gc.collect()
        torch.cuda.empty_cache()

        create_target_faiss(file_path, 'Bio-NNEL' , MAX_LENGTH)
        print('Embedding and indexing ended successfully!')

def to_dict(obj):
    """Safely convert Pydantic model or plain dict to dict."""
    if isinstance(obj, dict):
        return obj
    return obj.model_dump()

def save_intermediate_results(
    result_tuple,
    candidate_file="results/OntoMapping_benchmark/Clinical_var/Bio-NELL/all_candidates-2.csv",
    review_file="results/OntoMapping_benchmark/Clinical_var/Bio-NELL/mapping_reviews-2.csv"
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
    # 2. Flatten rows — handles both Pydantic models and plain dicts
    # -------------------------
    rows = []

    for _, row in review_df.iterrows():
        rag_review = to_dict(row["rag_review"])
        supervised = to_dict(row["supervised_ranking"])

        variable = supervised.get("variable")

        for cand in (supervised.get("candidates") or []):
            cand = to_dict(cand)  # candidates may also be Pydantic models or dicts
            rows.append({
                "variable":           variable,
                "review_status":      rag_review.get("status"),
                "review_explanation": rag_review.get("explanation"),
                "rank":               cand.get("rank"),
                "AI_code":            cand.get("AI_code"),
                "AI_name":            cand.get("AI_name"),
                "confidence":         cand.get("confidence"),
                "CUI":                cand.get("CUI"),
            })

    review_out = pd.DataFrame(rows)

    # -------------------------
    # 3. Append review results
    # -------------------------
    if not os.path.exists(review_file):
        review_out.to_csv(review_file, index=False)
    else:
        review_out.to_csv(review_file, mode="a", header=False, index=False)

def save_intermediate_results_cui(
    result_tuple,
    candidate_file="results/OntoMapping_benchmark/Clinical_var/Bio-NELL/all_candidates-3.csv",
    review_file="results/OntoMapping_benchmark/Clinical_var/Bio-NELL/mapping_reviews-3.csv"
):
    candidate_df, review_df = result_tuple

    # -------------------------
    # 1. Save candidate dataframe (unchanged)
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







if __name__ == '__main__':

# Embed BioNNE-L vocabulary
    out_dir = 'UMLS_mapper/data/raw'
    MAX_LENGTH = 25
    file_path = Path(f"{out_dir}/text_embs_{MAX_LENGTH}_BIO-NNEL.parquet")
    create_target_faiss(file_path, 'Bio-NNEL' , MAX_LENGTH)
    '''
    umls_df = load_bionnel_eng_umls()
    print('Embed Bio-NNEL UMLS vocabulary.')
    start_time = time.time()
    embed_bionnel_umls(umls_df, MAX_LENGTH=76)
    print(f'Embedding completed in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

    OUTPUT_DIR = "results/OntoMapping_benchmark/Clinical_var/Bio-NELL"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    candidate_file = f"{OUTPUT_DIR}/all_candidates-3.csv"
    review_file    = f"{OUTPUT_DIR}/mapping_reviews-3.csv"
    failure_file   = f"{OUTPUT_DIR}/failed_variables-3.txt"

    k = 20
    t = 0.6
    MAX_LENGTH = 25

    huggingface_path = "andorei/BioNNE-L"
    dev_data = load_dataset(huggingface_path, "English", split="dev")
    dev_df = dev_data.to_pandas().iloc[:100, :]

    rag_mapper = RAGMapper([''], [''], k = k, t = t)
    print(f"Number of variables to map: {len(dev_df)}")
    start_time = time.time()
    for _, row in tqdm(dev_df.iterrows(), total = len(dev_df)):
        var = row['text']
        var_type = row['entity_type']
        try:
            res = rag_mapper.evaluate(var, var_type)
            save_intermediate_results_cui(res, candidate_file, review_file)
            print(f"[OK] {var}")
        except Exception as e:
            print(f"[FAILED] {var} -> {e}")
            with open(failure_file, "a") as f:
                f.write(f"{var} :: {str(e)}\n")
    print(f'Mapping completed in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

'''









