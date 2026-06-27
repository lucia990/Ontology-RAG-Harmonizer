#from datasets import load_dataset
import pandas as pd
import os
from tqdm.auto import tqdm
import time
from pathlib import Path


from RAG_mapper.src.RAG_mapper import RAGMapper
from Evaluation.OntoMapping_benchmark.src.bio_nnel_pipeline import embed_bionnel_umls, load_bionnel_eng_umls
from Evaluation.OntoMapping_benchmark.src.OM_pipeline import create_target_faiss


class BackboneValidator:
    def __init__(
            self,
            rag_mapper,
            max_length=25,
            ground_truth_col='UMLS_CUI',
            candidate_file='candidate_results.csv',
            failure_file='failures.txt',
            grid_results_file='grid_results.csv',
            output_dir='results/OntoMapping_benchmark/ht_validation/mel_backbone'
    ):
        """
        Initializes the validation library.

        :param rag_mapper: An initialized instance of your RAGMapper.
        :param max_length: Maximum length parameter used for file indexing.
        :param ground_truth_col: Name of the column containing the true labels.
        """

        self.rag_mapper = rag_mapper
        self.max_length = max_length
        self.ground_truth_col = ground_truth_col
        self.candidate_file = candidate_file
        self.failure_file = failure_file
        self.grid_results_file = grid_results_file
        self.output_dir = output_dir

    def preprocess_data(self, df, sample_fraction=1.0, random_seed=42):
        """Cleans, deduplicates, and samples the input dataframe."""
        df = df.copy()
        df['variable'] = df['text'].astype(str) + " " + df['entity_type'].astype(str)

        # Deduplicate based on the combined variable string
        deduped_df = df.drop_duplicates(subset=['variable'], keep='first').copy()

        # Sample if fraction < 1.0
        if sample_fraction < 1.0:
            var_df = deduped_df.sample(frac=sample_fraction, random_state=random_seed).copy()
        else:
            var_df = deduped_df

        return deduped_df, var_df

    def run_inference(self, var_df):
        """Runs the RAG mapper over the dataset and saves raw candidates incrementally."""
        start_time = time.time()
        all_raw_results = []

        # Clear existing tracking files to prevent appending to old runs
        if os.path.exists(self.candidate_file): os.remove(self.candidate_file)
        if os.path.exists(self.failure_file): os.remove(self.failure_file)

        print(f"Starting inference on {len(var_df)} variables...")
        for _, row in tqdm(var_df.iterrows(), total=len(var_df)):
            var = row['variable']
            gt = row[self.ground_truth_col]
            try:
                res_df = self.rag_mapper.map_umls(var)
                if res_df is not None and not res_df.empty:
                    res_df = res_df.copy()
                    res_df['query_variable'] = var
                    res_df['ground_truth'] = gt
                    all_raw_results.append(res_df)

                    # Save raw outputs incrementally
                    header = not os.path.exists(self.candidate_file)
                    res_df.to_csv(f'{self.output_dir}/{self.candidate_file}', mode="a", header=header, index=False)

            except Exception as e:
                print(f"[FAILED] {var} -> {e}")
                with open(self.failure_file, "a") as f:
                    f.write(f"{var} :: {str(e)}\n")

        duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f'Inference completed in {duration}')
        return all_raw_results

    def evaluate_grid(self, all_raw_results, total_queries, grid_combinations, recall_k_vals=[1, 3, 5, 10]):
        """Performs in-memory grid search evaluation for hyperparameter tuning."""
        if not all_raw_results:
            print("No results available to evaluate.")
            return None

        print("\nStarting local grid-search hyperparameter evaluation...")
        master_res_df = pd.concat(all_raw_results, ignore_index=True)
        master_res_df['confidence'] = pd.to_numeric(master_res_df['confidence'])
        master_res_df = master_res_df.sort_values(by=['query_variable', 'confidence'], ascending=[True, False])

        grid_performance = []

        for k_val, t_val in grid_combinations:
            # Filter 1: Apply threshold constraint
            filtered = master_res_df[master_res_df['confidence'] >= t_val].copy()

            # Filter 2: Apply top-k constraint per query
            filtered = filtered.groupby('query_variable').head(k_val).copy()

            # Add rank position (1-indexed)
            filtered['rank'] = filtered.groupby('query_variable').cumcount() + 1
            filtered['is_match'] = filtered['CUI'] == filtered['ground_truth']

            # Find the minimum rank of the correct hit per query
            hits = filtered[filtered['is_match']].groupby('query_variable')['rank'].min().to_dict()

            metrics = {
                'grid_k': k_val,
                'grid_t': t_val,
            }

            for K in recall_k_vals:
                successful_hits = sum(1 for rank in hits.values() if rank <= K)
                recall_at_K = successful_hits / total_queries if total_queries > 0 else 0.0
                metrics[f'Recall@{K}'] = round(recall_at_K, 4)

            grid_performance.append(metrics)

        grid_df = pd.DataFrame(grid_performance)
        grid_df.to_csv(f'{self.output_dir}/{self.grid_results_file}', index=False)

        print("\n--- GRID SEARCH METRICS ---")
        print(grid_df.to_string(index=False))
        return grid_df

    def validate(self, raw_df, grid_combinations, sample_fraction=0.2, random_seed=42, recall_k_vals=[1, 3, 5, 10]):
        """Orchestrates the entire validation pipeline."""
        deduped_df, var_df = self.preprocess_data(raw_df, sample_fraction, random_seed)

        print(f"Total unique variables available: {len(deduped_df)}")
        print(f"Sampling {sample_fraction * 100}% -> Number of variables to map: {len(var_df)}")

        raw_results = self.run_inference(var_df)

        grid_df = self.evaluate_grid(
            all_raw_results=raw_results,
            total_queries=var_df['variable'].nunique(),
            grid_combinations=grid_combinations,
            recall_k_vals=recall_k_vals
        )
        return grid_df



if __name__ == '__main__':
    raw_data = pd.read_csv('Evaluation/OntoMapping_benchmark/DATA/bionnel_en_test.tsv', sep='\t')

    # 1. Set up SapBERT mapper from MAX_LENGTH
    #MAX_LENGTH = 25
    max_length_list = [50, 75]

    for MAX_LENGTH in max_length_list:

        ###### 1.1 Embed UMLS vocabulary
        umls_df = load_bionnel_eng_umls()
        print(f'Embed Bio-NNEL UMLS vocabulary with maximum length {MAX_LENGTH}.')
        start_time = time.time()
        embed_bionnel_umls(umls_df, MAX_LENGTH=MAX_LENGTH)
        print(f'Embedding completed in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

        ###### 1.2 Create FAISS index
        EMBS_DIR = 'UMLS_mapper/data/raw'
        embs_path = Path(f"{EMBS_DIR}/text_embs_{MAX_LENGTH}_BIO-NNEL.parquet")
        create_target_faiss(embs_path, 'Bio-NNEL' , MAX_LENGTH)

        # 2. Instantiate RAG mapper
        OUTPUT_DIR = f"results/OntoMapping_benchmark/ht_validation/mel_validation_{MAX_LENGTH}"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        rag_mapper = RAGMapper(
            [''], [''],
            FAISS_INDEX_PATH=f"UMLS_mapper/data/processed/faiss_index_{MAX_LENGTH}.bin",
            METADATA_PATH=f"UMLS_mapper/data/processed/metadata_{MAX_LENGTH}.csv",
            k=50, t=0.6
        )

        # 3. Instantiate the validator library
        validator = BackboneValidator(
            rag_mapper=rag_mapper,
            max_length=MAX_LENGTH,
            candidate_file=f"run_{MAX_LENGTH}_candidates.csv",
            output_dir= OUTPUT_DIR
        )

        # 4. Run validation
        grid_1 = [(50, 0.7), (50, 0.6), (20, 0.7)]
        results_run_1 = validator.validate(
            raw_df=raw_data,
            grid_combinations=grid_1,
            sample_fraction=0.1,
            random_seed=42
        )










'''
# configuration
OUTPUT_DIR = "results/OntoMapping_benchmark/ht_validation/mel_validation_25"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CANDIDATE_FILE = f"{OUTPUT_DIR}/all_candidates-3.csv"
GRID_RESULTS_FILE = f"{OUTPUT_DIR}/grid_search_metrics.csv"
FAILURE_FILE = f"{OUTPUT_DIR}/failed_variables-3.txt"

MAX_LENGTH = 25
SAMPLE_FRACTION = 0.2
RANDOM_SEED = 42

GRID_COMBINATIONS = [
    (5, 0.7), (5, 0.9),
    (10, 0.7), (10, 0.8), (10, 0.9),
    (20, 0.6), (20, 0.7), (20, 0.8),
    (50, 0.6), (50, 0.7)
]

SUP_GRID_COMBINATION = [(20, 0.7)]
RECALL_K_VALS = [1, 3, 5, 10]

#huggingface_path = "andorei/BioNNE-L"
#data = load_dataset(huggingface_path, "English", split="dev")
#raw_df = data.to_pandas()

raw_df = pd.read_csv('Evaluation/OntoMapping_benchmark/DATA/bionnel_en_test.tsv', sep='\t')
raw_df['variable'] = (
    raw_df['text'].astype(str) + " " + raw_df['entity_type'].astype(str)
)

# 2. Deduplicate based on the input variable string
deduped_df = raw_df.drop_duplicates(subset=['variable'], keep='first').copy()
#var_df = deduped_df.copy()
# 3. Randomly sample a fraction of the dataset
var_df = deduped_df.sample(frac=SAMPLE_FRACTION, random_state=RANDOM_SEED).copy()
#complementary_df = deduped_df.drop(var_df.index).copy() # use to validate supervisor llm

GROUND_TRUTH_COL = 'UMLS_CUI'


# --- INITIALIZE MAPPER WITH UPPER BOUNDS ---
MAX_K = 50
MIN_T = 0.6

rag_mapper = RAGMapper([''], [''], FAISS_INDEX_PATH = f"UMLS_mapper/data/processed/faiss_index_{MAX_LENGTH}.bin", METADATA_PATH= f"UMLS_mapper/data/processed/metadata_{MAX_LENGTH}.csv", k=MAX_K, t=MIN_T)
print(f"Total unique variables available: {len(deduped_df)}")
print(f"Sampling {SAMPLE_FRACTION * 100}% -> Number of variables to map: {len(var_df)}")

start_time = time.time()
all_raw_results = []

# --- STEP 1: MAX EXTRACTION LOOP ---
for _, row in tqdm(var_df.iterrows(), total=len(var_df)):
    var = row['variable']
    gt = row[GROUND_TRUTH_COL]
    try:
        res_df = rag_mapper.map_umls(var)
        if res_df is not None and not res_df.empty:
            res_df['query_variable'] = var
            res_df['ground_truth'] = gt
            all_raw_results.append(res_df)

            # Save raw outputs incrementally
            if not os.path.exists(CANDIDATE_FILE):
                res_df.to_csv(CANDIDATE_FILE, index=False)
            else:
                res_df.to_csv(CANDIDATE_FILE, mode="a", header=False, index=False)
v
    except Exception as e:
        print(f"[FAILED] {var} -> {e}")
        with open(FAILURE_FILE, "a") as f:
            f.write(f"{var} :: {str(e)}\n")

    print(f'Inference completed in {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

# --- STEP 2: IN-MEMORY GRID SEARCH EVALUATION ---
if all_raw_results:
    print("\nStarting local grid-search hyperparameter evaluation...")
    master_res_df = pd.concat(all_raw_results, ignore_index=True)

    master_res_df['confidence'] = pd.to_numeric(master_res_df['confidence'])
    master_res_df = master_res_df.sort_values(by=['query_variable', 'confidence'], ascending=[True, False])

    grid_performance = []

    for k_val, t_val in GRID_COMBINATIONS:
        # Filter 1: Apply strict threshold constraint
        filtered = master_res_df[master_res_df['confidence'] >= t_val].copy()

        # Filter 2: Apply strict top-k constraint per query
        filtered = filtered.groupby('query_variable').head(k_val).copy()

        # Add a rank position (1-indexed)
        filtered['rank'] = filtered.groupby('query_variable').cumcount() + 1
        filtered['is_match'] = filtered['CUI'] == filtered['ground_truth']

        # Find the rank of the correct hit per query
        hits = filtered[filtered['is_match']].groupby('query_variable')['rank'].min().to_dict()
        total_queries = var_df['variable'].nunique()

        metrics = {
            'grid_k': k_val,
            'grid_t': t_val,
        }

        for K in RECALL_K_VALS:
            successful_hits = sum(1 for rank in hits.values() if rank <= K)
            recall_at_K = successful_hits / total_queries if total_queries > 0 else 0.0
            metrics[f'Recall@{K}'] = round(recall_at_K, 4)

        grid_performance.append(metrics)

    grid_df = pd.DataFrame(grid_performance)
    grid_df.to_csv(GRID_RESULTS_FILE, index=False)

    print("\n--- GRID SEARCH METRICS ---")
    print(grid_df.to_string(index=False))
else:
    print("No results were compiled due to prior execution failures.")


'''