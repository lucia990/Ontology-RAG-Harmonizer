import os

from UMLS_mapper.src.faiss_index import FaissUMLS
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == "__main__":
    indexed_umls = FaissUMLS()

    # save indexed uml
    try:
        if not os.path.exists(f'UMLS_mapper/data/processed'):
            os.mkdir(f'UMLS_mapper/data/processed')
        indexed_umls._store_faiss_index(f'UMLS_mapper/data/processed/faiss_index_72.bin')

        # save metadata
        indexed_umls._map_metadata(f'UMLS_mapper/data/processed/metadata_72.csv')

    except FileExistsError as e:
        print(f"Error creating directory for UMLS_mapper: {e}")
