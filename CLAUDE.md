# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**Ontology-RAG-Harmonizer** maps clinical variables from source datasets to UMLS ontology codes using a 3-stage pipeline:
1. **SapBERT + FAISS** — embeds query variables and retrieves top-k semantically similar UMLS candidates
2. **Mapper LLM Agent** — ranks candidates using `granite4:latest` (or configurable model)
3. **Evaluator/Supervisor LLM Agent** — validates and reranks with `gpt-oss:20b`

LLMs are hosted on a remote Ollama server (configured via `.env`). All scripts must be run from the repository root because prompts and data paths are relative.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env` (already present, not committed) with:
```
OLLAMA_HOST=https://dev-openwebui.zbh.uni-hamburg.de/ollama
OLLAMA_API_KEY=<key>
```

Before mapping, build the FAISS index from the UMLS CONSO file:
```bash
# Embed a UMLS vocabulary subset (e.g., SNOMEDCT_US) at max_length=25
python UMLS_mapper/scripts/umls_embeddings.py --vocabularies SNOMEDCT_US --max_length 25
# Then build the FAISS index
python UMLS_mapper/src/create_faiss_index.py
# Outputs: UMLS_mapper/data/processed/faiss_index_25.bin + metadata_25.csv
```

## Running the Pipeline

**Single-variable mapping (interactive):**
```bash
python RAG_mapper/src/RAG_mapper.py
```

**End-to-end schema extraction with optional human review:**
```bash
python RAG_mapper/src/interactive_review.py
```

**Ontology benchmark evaluation (argparse CLI):**
```bash
python Evaluation/OntoMapping_benchmark/src/OM_pipeline.py \
  --vocabularies "ICD10 SNOMEDCT_US" \
  --max_length 25 \
  --llm_model gemma4:latest \
  --k 5 --t 0.6 --sampling_ratio 0.1 \
  --results_dir results/OntoMapping_benchmark/run1/
```

**MEL (Medical Entity Linking) pipeline with filter step:**
```bash
python Evaluation/OntoMapping_benchmark/src/MEL_pipeline.py
```

## Architecture

```
UMLS_mapper/
  src/umls_search_engine.py   # UMLSSearchEngine: SapBERT model + FAISS index singleton
  src/faiss_index.py          # FaissUMLS: builds/stores FAISS index from embeddings
  scripts/umls_embeddings.py  # Embeds UMLS CONSO file → Parquet, per vocabulary
  sapbert_model/              # Local SapBERT model weights (must be present at run time)
  data/
    raw/filtered_conso_eng.csv  # UMLS CONSO English subset (large, not in git)
    processed/faiss_index_*.bin # Built FAISS indices
    processed/metadata_*.csv    # CUI/name metadata aligned to FAISS index

RAG_mapper/
  src/llm_model.py            # OllamaWrapper (reads env vars), Pydantic output schemas
  src/RAG_mapper.py           # RAGMapper: FAISS search + mapper LLM chain + evaluator chain
  src/schema_builder.py       # SchemaBuilder(RAGMapperEvaluator): batch variable mapping
  src/interactive_review.py   # InteractiveReview(RAGMapper): human-in-the-loop correction
  src/map_evaluator.py        # RAGMapperEvaluator(RAGMapper): older evaluator chain style
  prompts/                    # Markdown prompt templates (system + human) for mapper & evaluator

preprocessing/
  src/filter_source.py        # pick_umls_source(): LLM picks target UMLS vocabulary for a var
  src/translate_var.py        # translate_var(): translates non-English descriptions
  prompts/                    # Prompt templates for source-selection and translation agents

Evaluation/
  OntoMapping_benchmark/
    src/OM_pipeline.py        # one_way_evaluation(): full benchmark with checkpointing
    src/MEL_pipeline.py       # SemanticMapper: vocabulary-aware single-variable mapping
    src/bio_nnel_pipeline.py  # BioNNEL baseline pipeline
    src/compute_scores.py     # Precision/recall/F1 scoring utilities
    src/umls_mapping.py       # create_mapping_table(): loads UMLS cross-mapping files
    ht_validation/            # Human-in-the-loop validation scripts (mel_backbone, om_sel/sup)
    DATA/                     # Benchmark datasets (MicrobAIome.csv, Breast_cancer.csv, etc.)
    UMLS_mappings/            # Pre-built UMLS cross-vocabulary mapping CSV/TSV files
  LuNikJay_benchmark/         # Human annotator agreement benchmark
```

## Key Class Hierarchy

```
UMLSSearchEngine              (UMLS_mapper/src/umls_search_engine.py)
  └─ singleton via get_umls_search_engine()

RAGMapper                     (RAG_mapper/src/RAG_mapper.py)
  ├─ map_umls()               FAISS nearest-neighbor search
  ├─ RAG_map()                mapper LLM chain → RankedCandidates
  └─ evaluate()               evaluator LLM chain → SupervisorOutput
     ├─ RAGMapperEvaluator    (map_evaluator.py — older LangChain-chain style)
     │    └─ SchemaBuilder    (schema_builder.py — batch loop + save to xlsx)
     └─ InteractiveReview     (interactive_review.py — human correction loop)
```

## Pydantic Output Schemas (llm_model.py)

LLM outputs are parsed via `PydanticOutputParser`. Key schemas:
- `RankedCandidates` — mapper agent output (up to 5 ranked `Candidate` objects)
- `SupervisorOutput` — evaluator output (`RagAudit` verdict + `RankedCUICandidates`)
- `Onto` — source-selection agent output (single vocabulary name)

## Checkpointing

`OM_pipeline.one_way_evaluation()` writes results in chunks and resumes from the last saved row automatically. Checkpoint files follow the pattern:
```
results/<dir>/rag_output_{MAX_LENGTH}_{source_onto}_{target_onto}.csv
```

## Models & Hosts

| Model | Role |
|---|---|
| `granite4:latest` | Mapper LLM agent (candidate ranking) |
| `gpt-oss:20b` | Evaluator/supervisor agent |
| `gemma4:latest` | Alternative mapper (benchmark runs) |
| SapBERT (local) | Embedding model at `UMLS_mapper/sapbert_model/` |

Ollama client is authenticated via Bearer token in `OLLAMA_API_KEY` env var. Connection failures with "401" indicate an expired/wrong key.
