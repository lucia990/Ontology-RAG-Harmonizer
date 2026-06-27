from pydantic import BaseModel

from Evaluation.OntoMapping_benchmark.src.OM_pipeline import setup_logging, timer, embed_target_ontology, create_target_faiss
from RAG_mapper.src.RAG_mapper import RAGMapper
from preprocessing.src.filter_source import pick_umls_source

logger = setup_logging()

# It has to be initialized for every variable
class SemanticMapper(BaseModel):
    var: str
    var_desc: str
    k: int
    t: float
    MAX_LENGTH: int

    def map_var(self):

        # 1. Filter LLM
        onto = pick_umls_source(self.var,'')

        # 2. SaPBERT Semantic mapper
        logger.info(f"Embedding vocabulary: {onto}...")
        embed_target_ontology(onto, self.MAX_LENGTH)

        logger.info("Creating FAISS index...")
        emb_path = f"UMLS_mapper/data/raw/text_embs_{self.MAX_LENGTH}_{onto}.parquet"
        create_target_faiss(emb_path, onto, self.MAX_LENGTH)

        logger.info("Initialising RAG mapper...")
        logger.info(f"Source variables to map: {self.var}...")

        # 3. Adjudicator agent
        with timer(f"... Mapping {self.var}..."):
            rag_mapper = RAGMapper(
                [self.var],
                [self.var_desc],
                k=self.k,
                t=self.t
            )
            res = rag_mapper.evaluate(self.var, self.var_desc,  'gpt-oss:20b')
            return res


# Example usage
if __name__ == "__main__":

    k = 10
    t = 0.6
    MAX_LENGTH = 25

    var_1 = 'Class'
    var_desc_1 = 'binary classification label that indicates if the patient had a breast cancer recurrence event or not after n years. The time to event is not defined by the dataset, could be 10 years or something else'

    var_2 = 'Age'
    var_desc_2 = 'Groups of reported patient age in year at visit time. The years are grouped in timespans of 10 years (10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80-89, 90-99) '

    var_3 = 'menopause'
    var_desc_3 = 'menopausal state of the patient. Can be either menopause before or after 40 years or pre menopause. Is given in three classes (lt40, ge40, premeno)'
    sm = SemanticMapper(var=var_3, var_desc = var_desc_3, k = k, t= t, MAX_LENGTH = MAX_LENGTH)
    res = sm.map_var()
    print(res)
    print(type(res))

