import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import logging
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import AutoTokenizer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def clean_name(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters/punctuation (keeping spaces and alphanumeric characters)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Normalize whitespace (replace multiple spaces with a single space and strip)
    text = " ".join(text.split())
    return text

def preprocess_umls(df, sample_frac=0.0001, random_state=42):
    umls_df = df.dropna(subset= ['Name'])
    umls_df['Name'] = umls_df['Name'].apply(clean_name)
    umls_df = umls_df[umls_df['Name'] != ""]
    umls_df = umls_df.drop_duplicates(subset= ['Name', 'type'], keep='first')
    umls_df["name_length"] = umls_df["Name"].apply(len)
    N = len(umls_df)
    try:
        umls_df["length_strata"] = pd.qcut(
            umls_df["name_length"], q=5, labels=False, duplicates="drop"
        )
    except ValueError:
        # Fallback to simple binning if qcut fails due to lack of variance
        umls_df["length_strata"] = pd.cut(
            umls_df["name_length"], bins=5, labels=False
        )
    actual_strata_count = umls_df["length_strata"].nunique()
    # 4. Dynamically calculate sample size per group
    target_total_samples = int(np.ceil(N * sample_frac))
    samples_per_stratum = max(
        1, int(np.ceil(target_total_samples / actual_strata_count))
    )

    # 5. Perform Stratified Sampling safely (Fixing the include_groups placement)
    sampled_df = umls_df.groupby("length_strata", group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), samples_per_stratum), random_state=random_state
        ),
        include_groups=False,  # <--- It goes here, as an argument to .apply()
    )

    # 6. Shuffle the final subset deterministically
    sampled_df = sampled_df.sample(frac=1, random_state=random_state)

    # 7. Clean up temporary columns and reset index
    sampled_df = sampled_df.drop(
        columns=["name_length", "length_strata"], errors="ignore"
    ).reset_index(drop=True)

    print(
        f"Original Cleaned Rows: {N} -> Sampled Rows: {len(sampled_df)}"
    )
    return sampled_df

def tokenize_table(df, tokenizer, BATCH_SIZE):
    N = len(df)
    logger.info(f"Tokenizing {N} variables...")
    all_token_lengths = []
    print("Analyzing token lengths...")
    for i in tqdm(np.arange(0, N, BATCH_SIZE)):
        batch_text = list(df.variable)[i: i + BATCH_SIZE]

        # Tokenize without padding or truncation to get the real lengths
        toks = tokenizer.batch_encode_plus(
            batch_text,
            padding=False,  # Don't pad so we can see the actual length
            truncation=False,  # Don't truncate so we don't lose data in our stats
            return_attention_mask=False,  # Shaves off a tiny bit of processing time
        )

        # Calculate length for each sequence in the batch and append to our master list
        for input_ids in toks["input_ids"]:
            all_token_lengths.append(len(input_ids))

    # 2. Calculate and display statistics
    all_token_lengths = np.array(all_token_lengths)
    avg_length = np.mean(all_token_lengths)
    max_len_observed = np.max(all_token_lengths)
    p95 = np.percentile(all_token_lengths, 95)
    p99 = np.percentile(all_token_lengths, 99)

    print("\n" + "=" * 40)
    print("TOKEN LENGTH STATISTICS")
    print("=" * 40)
    print(f"Average token count: {avg_length:.2f}")
    print(f"95th percentile:     {p95:.2f} (Covers 95% of your data)")
    print(f"99th percentile:     {p99:.2f} (Covers 99% of your data)")
    print(f"Maximum token count: {max_len_observed}")
    print("=" * 40)
    print(
        f"Recommendation: Set MAX_LENGTH to {int(np.ceil(p99))} or {int(np.ceil(p95))} to balance speed and context."
    )


# Example usage

if __name__ == "__main__":

    logger.info("=" * 50)
    logger.info("Starting Data Preprocessing and Tokenization Pipeline")
    logger.info("=" * 50)
    ###### 1.  Load data
    logger.info("Loading datasets...")
    ## MEL
    '''
    try:
        breast_cancer_df = pd.read_csv(
                'Evaluation/OntoMapping_benchmark/DATA/Breast_cancer.csv')[['schemanode_name', 'schemanode_description', 'ontology_id']]
        logger.info(f"Loading Breast Cancer csv")
        breast_cancer_df['variable'] = (
            breast_cancer_df['schemanode_name'].astype(str) + " " + breast_cancer_df['schemanode_description'].astype(str)
        )
        logger.info(
            f"Successfully loaded Breast Cancer data. Shape: {breast_cancer_df.shape}"
        )
    except Exception as e:
        logger.error(f"Failed to load Breast Cancer data: {e}")
        raise
    try:
        CDC_df = pd.read_excel('Evaluation/OntoMapping_benchmark/DATA/CDC_DHI.xlsx')[['schemanode_name', 'Description']]
        logger.info(f"Loading CDC xlsx")
        CDC_df['variable'] = (
            CDC_df['schemanode_name'].astype(str) + " " + CDC_df['Description'].astype(str)
        )
        logger.info(f"Successfully loaded CDC data. Shape: {CDC_df.shape}")
    except Exception as e:
        logger.error(f"Failed to load CDC data: {e}")
        raise
    '''
    ## OM

    try:
        umls_df = pd.read_csv('UMLS_mapper/data/raw/filtered_conso_eng.csv',  index_col=False, names=['CUI', 'Language', 'Status', 'LUI', 'string_type', 'SUI', 'atom_status', 'AUI',
                                          'SAUI', 'SCUI', 'SDUI', 'source', 'type', 'CODE', 'Name', 'restriction_level',
                                          'SUPPRESS', 'CVF'], low_memory=False)[['Name', 'type']]
        logger.info(
            f"Raw UMLS data loaded. Shape: {umls_df.shape}. Starting preprocessing..."
        )

        umls_df = preprocess_umls(umls_df, sample_frac= 0.001 )
        umls_df["variable"] = (
            umls_df["Name"].astype(str) + " " + umls_df["type"].astype(str)
        )
        logger.info(
            f"Successfully preprocessed UMLS data. New Shape: {umls_df.shape}"
        )
    except Exception as e:
        logger.error(f"Failed during UMLS loading or preprocessing: {e}")
        raise

    ##### 2. Tokenize
    logger.info("Initializing Tokenizer...")
    BATCH_SIZE_OM = 128
    BATCH_SIZE_MEL = 5
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded tokenizer successfully from: '{model_name}'")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    ## MEL
    #logger.info("Tokenize Breast Cancer data...")
    #tokenize_table(breast_cancer_df, tokenizer, BATCH_SIZE_MEL)
    #logger.info("Tokenize Diabetes data...")
    #tokenize_table(CDC_df, tokenizer, BATCH_SIZE_MEL)
    ## OM
    logger.info("Tokenize UMLS data...")
    tokenize_table(umls_df, tokenizer, BATCH_SIZE_OM)


