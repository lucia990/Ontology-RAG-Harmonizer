import argparse
import gc

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import pyarrow.parquet as pq
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import AutoTokenizer, AutoModel
from Evaluation.OntoMapping_benchmark.src.utils import read_mapping_table


def umls_sapbert_embeddings(vocabularies: str = None, mapping_filename: str = None , BATCH_SIZE=128, MAX_LENGTH = 25):
    torch.cuda.empty_cache()
    if mapping_filename is None: # DEFAULT: entire UMLS
        print('Load UMLS from CONSO file...')
        kg_1 = pd.read_csv(f"UMLS_mapper/data/raw/filtered_conso_eng.csv", index_col=False,
                           names=['CUI', 'Language', 'Status', 'LUI', 'string_type', 'SUI', 'atom_status', 'AUI',
                                  'SAUI', 'SCUI', 'SDUI', 'source', 'type', 'CODE', 'Name', 'restriction_level',
                                  'SUPPRESS', 'CVF'], low_memory=False)
        list_vocabularies = ['UMLS']
        if vocabularies: # subset of UMLS
            list_vocabularies = vocabularies.split()
            print(f'Keep {vocabularies} vocabularies...')
            kg_1 = kg_1[kg_1.source.isin(list_vocabularies)]
            kg_1.reset_index(inplace=True)

        kg_1 = kg_1[['Name', 'CUI', 'source', 'CODE', 'type']]

    else: # ontology from mapping file
        kg_1 = read_mapping_table(mapping_filename)
        try:
            list_vocabularies = vocabularies.split()
        except Exception as e:
            print(f'Vocabulary required when embedding from mapping file! Error: {e}\n\n')
            try:
                vocabularies= input(f'Please select source vocabulary from file {mapping_filename}:')
                list_vocabularies = [vocabularies]
            except Exception as e:
                print(f'Error: {e}\n\n')
                exit()
        file_path = Path(f"UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{'_'.join(list_vocabularies)}.parquet")
        if file_path.exists():
            print(f'Embeddings for ontologies {vocabularies} already exist in  {file_path}  Skipping...')
            umls_parquet = pq.ParquetFile(file_path)
            dfs = []
            for batch in tqdm(umls_parquet.iter_batches(batch_size=1000), desc="Loading UMLS embeddings"):
                # umls_embeddings = pd.concat([umls_embeddings, i.to_pandas()], ignore_index=True)
                dfs.append(batch.to_pandas())
            umls_embeddings = pd.concat(dfs, ignore_index=True)

            return umls_embeddings
        kg_1 = kg_1.loc[:, [f'{vocabularies}_names', f'{vocabularies}_ids']]
        kg_1.columns = ['Name', 'CODE']
        kg_1["Name"] = kg_1.iloc[:, 0].apply(lambda x: ' '.join(x))
    # Check if the embedding file already exists
    file_path = Path(f"UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{'_'.join(list_vocabularies)}.parquet")
    if file_path.exists():
        print(f'Embeddings for ontologies {vocabularies} already exist in  {file_path}  Skipping...')
        return 0
    N = len(kg_1)
    print(N)
    print(f"Numbers of nodes to embed: {N}")

    nan_count = kg_1['Name'].isnull().sum()

    print(f"Percentage of NaN 'Name' entries: {(nan_count / N) * 100:.5f}%")

    # replace NaN with empty strings
    kg_1['Name'] = kg_1['Name'].fillna('')
    kg_1['CODE'] = kg_1['CODE'].fillna('')
    kg_1['CODE'] = kg_1['CODE'].astype("string")

    print(f'Load embedding model...\n')
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", torch_dtype=torch.float16).cuda()




    print(f'*************** Compute embeddings with max Length: {MAX_LENGTH}***************\n')
    all_embs = []
    with torch.no_grad():
        for i in tqdm(np.arange(0, N, BATCH_SIZE)):
            print(f"\nTokenizing batch {i}...\n")
            batch_text = list(kg_1.Name)[i:i + BATCH_SIZE]
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

    kg_1['Text_Embs'] = [emb for emb in all_embs]
    kg_1.to_parquet(f"UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{'_'.join(list_vocabularies)}.parquet", index=False)
    print(f"------{N} Sapbert embeddings saved in UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{'_'.join(list_vocabularies)}.parquet")
    del model, tokenizer, all_embs
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_filename', type=str, default=None)
    parser.add_argument('--vocabularies', type=str, required=False, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_length', type=int, default=25)
    args = parser.parse_args()
    list_vocabularies = args.vocabularies
    mapping_filename = args.mapping_filename
    batch_size = args.batch_size
    max_length = args.max_length

    umls_sapbert_embeddings(mapping_filename=mapping_filename, vocabularies = list_vocabularies, BATCH_SIZE=batch_size, MAX_LENGTH=max_length)
