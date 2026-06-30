import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# Disable CUDA cleanly since we are intentionally using the Intel GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import OpenVINO's PyTorch integration backend
import openvino as ov
from Evaluation.OntoMapping_benchmark.src.utils import read_mapping_table


def umls_sapbert_embeddings(
    vocabularies: str = None,
    mapping_filename: str = None,
    BATCH_SIZE=128,
    MAX_LENGTH=25,
):
    if mapping_filename is None:  # DEFAULT: entire UMLS
        print("Load UMLS from CONSO file...")
        kg_1 = pd.read_csv(
            f"UMLS_mapper/data/raw/filtered_conso_eng.csv",
            index_col=False,
            names=[
                "CUI",
                "Language",
                "Status",
                "LUI",
                "string_type",
                "SUI",
                "atom_status",
                "AUI",
                "SAUI",
                "SCUI",
                "SDUI",
                "source",
                "type",
                "CODE",
                "Name",
                "restriction_level",
                "SUPPRESS",
                "CVF",
            ],
            low_memory=False,
        )
        list_vocabularies = ["UMLS"]
        if vocabularies:  # subset of UMLS
            list_vocabularies = vocabularies.split()
            print(f"Keep {vocabularies} vocabularies...")
            kg_1 = kg_1[kg_1.source.isin(list_vocabularies)]
            kg_1.reset_index(inplace=True)

        kg_1 = kg_1[["Name", "CUI", "source", "CODE", "type"]]

    else:  # ontology from mapping file
        kg_1 = read_mapping_table(mapping_filename)
        try:
            list_vocabularies = vocabularies.split()
        except Exception as e:
            print(
                f"Vocabulary required when embedding from mapping file! Error: {e}\n\n"
            )
            try:
                vocabularies = input(
                    f"Please select source vocabulary from file {mapping_filename}:"
                )
                list_vocabularies = [vocabularies]
            except Exception as e:
                print(f"Error: {e}\n\n")
                exit()
        file_path = Path(
            f"UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{'_'.join(list_vocabularies)}.parquet"
        )
        if file_path.exists():
            print(
                f"Embeddings for ontologies {vocabularies} already exist in {file_path} Skipping..."
            )
            umls_parquet = pq.ParquetFile(file_path)
            dfs = []
            for batch in tqdm(
                umls_parquet.iter_batches(batch_size=1000),
                desc="Loading UMLS embeddings",
            ):
                dfs.append(batch.to_pandas())
            umls_embeddings = pd.concat(dfs, ignore_index=True)

            return umls_embeddings
        kg_1 = kg_1.loc[:, [f"{vocabularies}_names", f"{vocabularies}_ids"]]
        kg_1.columns = ["Name", "CODE"]
        kg_1["Name"] = kg_1.iloc[:, 0].apply(lambda x: " ".join(x))

    # Check if the embedding file already exists
    file_path = Path(
        f"UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{'_'.join(list_vocabularies)}.parquet"
    )
    if file_path.exists():
        print(
            f"Embeddings for ontologies {vocabularies} already exist in {file_path} Skipping..."
        )
        return 0
    N = len(kg_1)
    print(N)
    print(f"Numbers of nodes to embed: {N}")

    nan_count = kg_1["Name"].isnull().sum()
    print(f"Percentage of NaN 'Name' entries: {(nan_count / N) * 100:.5f}%")

    # replace NaN with empty strings
    kg_1["Name"] = kg_1["Name"].fillna("")
    kg_1["CODE"] = kg_1["CODE"].fillna("")
    kg_1["CODE"] = kg_1["CODE"].astype("string")

    print(f"Load embedding model...\n")
    tokenizer = AutoTokenizer.from_pretrained(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    model = AutoModel.from_pretrained(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )

    # ------------------ OPENVINO TRANSFORMATION ------------------
    print(f"Compiling model with OpenVINO for Intel iGPU...\n")
    core = ov.Core()

    # Define dummy inputs to trace the structure of the model execution graph
    dummy_input = tokenizer(
        "dummy text",
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    input_shape = dummy_input["input_ids"].shape

    # Convert PyTorch model to OpenVINO IR (Intermediate Representation)
    ov_model = ov.convert_model(
        model,
        example_input=dict(dummy_input),
        input=[
            ("input_ids", input_shape, ov.Type.i64),
            ("attention_mask", input_shape, ov.Type.i64),
            ("token_type_ids", input_shape, ov.Type.i64),
        ],
    )

    # Reshape the compiled model dynamically to accept variable-sized batch inputs
    ov_model.reshape(
        {
            "input_ids": [-1, MAX_LENGTH],
            "attention_mask": [-1, MAX_LENGTH],
            "token_type_ids": [-1, MAX_LENGTH],
        }
    )

    # Load compiled structure strictly onto your Intel Iris Xe Graphics ("GPU")
    compiled_model = core.compile_model(ov_model, device_name="GPU")
    # -------------------------------------------------------------

    print(
        f"*************** Compute embeddings with max Length: {MAX_LENGTH}***************\n"
    )
    all_embs = []
    for i in tqdm(np.arange(0, N, BATCH_SIZE)):
        batch_text = list(kg_1.Name)[i : i + BATCH_SIZE]
        toks = tokenizer.batch_encode_plus(
            batch_text,
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors="np",  # Native NumPy arrays work beautifully with OpenVINO
        )

        # Build input dictionary mapping matching model variable keys
        ov_inputs = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "token_type_ids": toks["token_type_ids"],
        }

        # Run inference using your Iris Xe iGPU
        res = compiled_model(ov_inputs)

        # Pull out output tensor 0 (equivalent to model(...)[0] in PyTorch)
        output_tensor = list(res.values())[0]

        # Extract CLS token representations [:, 0, :]
        cls_rep = output_tensor[:, 0, :]
        all_embs.append(cls_rep)

    all_embs = np.concatenate(all_embs, axis=0)

    kg_1["Text_Embs"] = [emb for emb in all_embs]
    kg_1.to_parquet(
        f"UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{'_'.join(list_vocabularies)}.parquet",
        index=False,
    )
    print(
        f"------{N} Sapbert embeddings saved in UMLS_mapper/data/raw/text_embs_{MAX_LENGTH}_{'_'.join(list_vocabularies)}.parquet"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_filename", type=str, default=None)
    parser.add_argument("--vocabularies", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=25)
    args = parser.parse_args()
    list_vocabularies = args.vocabularies
    mapping_filename = args.mapping_filename
    batch_size = args.batch_size
    max_length = args.max_length

    umls_sapbert_embeddings(
        mapping_filename=mapping_filename,
        vocabularies=list_vocabularies,
        BATCH_SIZE=batch_size,
        MAX_LENGTH=max_length,
    )