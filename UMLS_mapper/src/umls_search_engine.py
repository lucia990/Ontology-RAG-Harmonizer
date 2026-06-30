import pandas as pd

import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import torch
import os



class UMLSSearchEngine:
    def __init__(self, model_path: str, faiss_index_path: str, metadata_path: str):
        """
            Load Embedding model, metadata and FAISS index.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"Loading embedding model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        print("Embedding model loaded.")

        print(f"Loading metadata from: {metadata_path}")
        self.metadata = pd.read_csv(metadata_path).copy()
        print("Metadata loaded.")


        print(f"Loading FAISS index from: {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)
        print("FAISS index loaded.")


    def embed_query(self, query: str) -> np.ndarray:
        """
            Embeds a single query string using SapBERT.
            The output embedding is L2 normalized.
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.tokenizer(query, return_tensors="pt").to(self.device)
            output_tensor = self.model(**input_tensor)[0][:, 0, :]
            output_numpy = output_tensor.cpu().detach().numpy()
            # L2 Normalize the query embedding
            output_numpy_normalized = normalize(output_numpy, axis=1, norm='l2')
        return output_numpy_normalized

    def find_nearest_neighbors(self, query: str, k: int = 5):
        """
            Finds the k nearest neighbors for a given query in the UMLS embeddings.
        """
        print(f'Embedding query: "{query}"')
        query_embs = self.embed_query(query)
        D_cosine, I = self.faiss_index.search(query_embs, k)
        return D_cosine, I



    def search(self, query: str, k: int = 50) -> pd.DataFrame:
        """
            Performs a full search, returning the top k nearest neighbors with their details.
        """
        print(f"search for {query}... ")
        D, I = self.find_nearest_neighbors(query, k)
        nearest_neighbor_indices = I[0]
        results_df = pd.DataFrame({
            'Index': nearest_neighbor_indices,
            'CUI': self.metadata.loc[nearest_neighbor_indices, 'CUI'].values,
            'ontology_name': self.metadata.loc[nearest_neighbor_indices, 'concept_name'].values,
            'confidence': D[0],
            #'source': self.metadata.loc[nearest_neighbor_indices, 'source'].values,
            #'type': self.metadata.loc[nearest_neighbor_indices, 'type'].values,
            #'se_CODE': self.metadata.loc[nearest_neighbor_indices, 'CODE'].values,
            'semantic_type': self.metadata.loc[nearest_neighbor_indices, 'semantic_type'].values
        })
        return results_df



# Global instance of the search engine
# (This will be loaded once when the Django app starts)
_umls_search_engine_instance = None

def get_umls_search_engine(FAISS_INDEX_PATH:str = "UMLS_mapper/data/processed/faiss_index_25.bin", METADATA_PATH: str = "UMLS_mapper/data/processed/metadata_25.csv"):
    global _umls_search_engine_instance
    if _umls_search_engine_instance is None:
        MODEL_PATH = "UMLS_mapper/sapbert_model"

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}. Make sure it's correctly mounted/copied in Docker.")
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index file not found at {FAISS_INDEX_PATH}. Make sure it's correctly mounted/copied in Docker.")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata CSV file not found at {METADATA_PATH}. Make sure it's correctly mounted/copied in Docker.")

        _umls_search_engine_instance = UMLSSearchEngine(MODEL_PATH, FAISS_INDEX_PATH, METADATA_PATH)
    return _umls_search_engine_instance


