import gc
import os

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class FaissUMLS:
    def __init__(self, embeddings_path: str = 'UMLS_mapper/data/raw/text_embs_25.parquet'):
        print("Initializing FaissUMLS...")

        self.umls_embeddings_df = self._load_and_preprocess_embeddings(embeddings_path)

        # Build index with local arrays and free them immediately after
        embs = np.array(self.umls_embeddings_df['Text_Embs'].values.tolist(), dtype=np.float32)
        embs_normalized = normalize(embs, axis=1, norm='l2')
        del embs

        self.index = self._create_faiss_index(embs_normalized)
        del embs_normalized
        gc.collect()

        # Drop embedding column — only metadata columns needed for _map_metadata
        self.umls_embeddings_df.drop(columns=['Text_Embs'], inplace=True)

        print("FaissUMLS initialized successfully.")

    def _load_and_preprocess_embeddings(self, path: str) -> pd.DataFrame:
        print(f"Loading embeddings from: {path}")
        umls_parquet = pq.ParquetFile(path)
        dfs = []
        for batch in tqdm(umls_parquet.iter_batches(batch_size=1000), desc="Loading UMLS embeddings"):
            dfs.append(batch.to_pandas())
        umls_embeddings = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(umls_embeddings)} embeddings.")
        return umls_embeddings

    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        print(f'FAISS index populated with {index.ntotal} embeddings.')
        return index

    def _store_faiss_index(self, file_path: str):
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"Storing FAISS index into: {file_path}")
        faiss.write_index(self.index, file_path)

    def _map_metadata(self, output):
        # Text_Embs already dropped in __init__
        self.umls_embeddings_df.to_csv(output, index=False)
