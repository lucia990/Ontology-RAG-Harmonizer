from preprocessing.src.filter_source import pick_source
import pandas as pd
import numpy as np
import time
import os
from Evaluation.OntoMapping_benchmark.src.OM_pipeline import setup_logging


logger = setup_logging()

def sample_umls(df, sampling_ratio:float = 0.1):
    N = len(df)
    print(f'Number of source variables before sampling: {N}')
    K = int(np.floor(sampling_ratio * N))
    np.random.seed(123)
    sampled_idx = np.random.choice(df.index, size=K, replace=False)
    sampled_source = df.loc[sampled_idx]
    print(f'Number of source variables after sampling: {len(sampled_source)}')
    return sampled_source


if __name__ == '__main__':

    conso_file = pd.read_csv(f"UMLS_mapper/data/raw/filtered_conso_eng.csv", index_col=False,
                             names=['CUI', 'Language', 'Status', 'LUI', 'string_type', 'SUI', 'atom_status', 'AUI',
                                    'SAUI', 'SCUI', 'SDUI', 'source', 'type', 'CODE', 'Name', 'restriction_level',
                                    'SUPPRESS', 'CVF'], low_memory=False)[['Name', 'source']]

    vocabularies = ['SNOMEDCT_US', 'NCBI', 'ICD10', 'LNC', 'RXNORM']
    llms_models = ['gpt-oss:20b', 'gemma4:latest', 'llama3.3:70b', 'qwen3.5:4b', 'granite4:latest']
    sub_umls = conso_file[conso_file['source'].isin(vocabularies)]
    sampled_umls = sample_umls(sub_umls, 0.0001).copy()
    computational_time = {}
    results_dir= 'results/OntoMapping_benchmark/filter_llm_benchmark/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        logger.debug(f"Created directory: {results_dir}")

    for llm_model in llms_models:
        start_time = time.time()

        sampled_umls[f'{llm_model}_pred'] = sampled_umls['Name'].apply(lambda x: pick_source(x, llm_model)[0])

        computational_time[llm_model] = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))


        with open(f'{results_dir}/computational_time_filter.txt', 'a') as file:
            file.write(f'{llm_model}: {computational_time[llm_model]} \n')

        sampled_umls.to_csv(f'{results_dir}/out_{llm_model}.csv', index=False)
