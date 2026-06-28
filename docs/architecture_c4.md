# Ontology-RAG-Harmonizer — C4 Architecture

## Level 1: System Context

```mermaid
C4Context
    title System Context — Ontology-RAG-Harmonizer

    Person(researcher, "Data Harmonization Researcher", "Maps clinical/microbiome variables to standard ontology codes")

    System(harmonizer, "Ontology-RAG-Harmonizer", "3-stage pipeline: FAISS retrieval → Mapper LLM → Evaluator LLM")

    System_Ext(ollama, "Remote Ollama Server", "Hosts LLMs (granite4, gpt-oss:20b, gemma4). Auth via Bearer token.")
    System_Ext(umls, "UMLS Metathesaurus", "Source of ontology concepts (SNOMEDCT_US, ICD10, LOINC, …). Provided as CONSO flat file.")

    Rel(researcher, harmonizer, "Submits variable list / schema")
    Rel(harmonizer, ollama, "Calls Mapper LLM + Evaluator LLM", "HTTPS / Ollama API")
    Rel(harmonizer, umls, "Reads CONSO file at setup", "local CSV")
```

---

## Level 2: Container Diagram

```mermaid
C4Container
    title Containers — Ontology-RAG-Harmonizer

    Person(researcher, "Researcher")

    Container(preproc, "Preprocessing", "Python", "Detects language, translates non-English variables, selects target UMLS vocabulary (Source Selector LLM)")
    Container(umls_mapper, "UMLS Mapper", "Python / SapBERT / FAISS", "Embeds UMLS concepts with SapBERT; builds & queries FAISS index for nearest-neighbour retrieval")
    Container(rag_mapper, "RAG Mapper", "Python / LangChain", "Mapper LLM ranks FAISS candidates; Evaluator LLM validates and reranks; interactive or batch mode")
    Container(evaluation, "Evaluation", "Python", "Ontology-to-ontology benchmark (OM_pipeline), MEL benchmark, HT validation scripts, scoring utilities")

    System_Ext(ollama, "Remote Ollama Server")
    System_Ext(umls, "UMLS CONSO file")

    Rel(researcher, preproc, "Optional: passes raw variables")
    Rel(researcher, rag_mapper, "Runs interactive_review.py or schema_builder.py")
    Rel(preproc, ollama, "Source Selector LLM call", "HTTPS")
    Rel(preproc, rag_mapper, "Normalised variable + vocabulary hint")
    Rel(umls_mapper, umls, "Reads filtered_conso_eng.csv at index build time")
    Rel(rag_mapper, umls_mapper, "Calls UMLSSearchEngine.search()")
    Rel(rag_mapper, ollama, "Mapper LLM + Evaluator LLM calls", "HTTPS")
    Rel(evaluation, umls_mapper, "Uses UMLSSearchEngine for benchmark retrieval")
    Rel(evaluation, ollama, "Benchmark LLM calls", "HTTPS")
```

---

## Level 3: Component Diagram — RAG Mapper

```mermaid
C4Component
    title Components — RAG Mapper container

    Container_Ext(umls_mapper, "UMLS Mapper")
    Container_Ext(ollama, "Remote Ollama Server")

    Component(llm_model, "OllamaWrapper + Pydantic Schemas", "llm_model.py", "Wraps Ollama client with retry logic. Defines output schemas: RankedCandidates, SupervisorOutput.")
    Component(rag_mapper_core, "RAGMapper", "RAG_mapper.py", "Orchestrates FAISS search → Mapper LLM chain → Evaluator LLM chain for a single variable.")
    Component(schema_builder, "SchemaBuilder", "schema_builder.py", "Batch loop over variable list; calls RAGMapperEvaluator per variable; saves results to XLSX with checkpointing.")
    Component(interactive_review, "InteractiveReview", "interactive_review.py", "Human-in-the-loop: presents mapping proposals, accepts corrections, updates schema.")
    Component(map_evaluator, "RAGMapperEvaluator", "map_evaluator.py", "Older LangChain-chain style evaluator, parent class of SchemaBuilder.")

    Rel(rag_mapper_core, llm_model, "Uses OllamaWrapper + parsers")
    Rel(rag_mapper_core, umls_mapper, "map_umls() → FAISS search")
    Rel(rag_mapper_core, ollama, "Mapper & Evaluator LLM calls")
    Rel(schema_builder, map_evaluator, "Inherits from")
    Rel(map_evaluator, rag_mapper_core, "Inherits from RAGMapper")
    Rel(interactive_review, rag_mapper_core, "Inherits from RAGMapper")
```

---

## Level 3: Component Diagram — UMLS Mapper

```mermaid
C4Component
    title Components — UMLS Mapper container

    Component(search_engine, "UMLSSearchEngine", "umls_search_engine.py", "Singleton. Loads SapBERT model + FAISS index. Exposes search(query, k) → DataFrame of candidates with cosine similarity scores.")
    Component(faiss_idx, "FaissUMLS", "faiss_index.py", "Builds FAISS index from SapBERT embeddings. Saves/loads faiss_index_*.bin + metadata_*.csv.")
    Component(embeddings_script, "umls_embeddings.py", "scripts/", "Embeds UMLS CONSO subset (per vocabulary, configurable max_length) → Parquet files.")
    Component(faiss_script, "create_faiss_index.py", "scripts/", "Reads Parquet embeddings, calls FaissUMLS.build(), writes index to disk.")

    System_Ext(sapbert, "SapBERT Model", "Local weights at sapbert_model/")
    System_Ext(umls_conso, "UMLS CONSO CSV", "data/raw/filtered_conso_eng.csv")

    Rel(faiss_script, faiss_idx, "Calls FaissUMLS.build()")
    Rel(faiss_script, embeddings_script, "Reads Parquet output")
    Rel(embeddings_script, umls_conso, "Reads CONSO subset rows")
    Rel(embeddings_script, sapbert, "Encodes concept names")
    Rel(search_engine, sapbert, "Encodes query at runtime")
    Rel(search_engine, faiss_idx, "Loads index, calls search()")
```

---

## Data / Artifact Flow (end-to-end)

```mermaid
flowchart TD
    A[Raw variable list\n.csv / .xlsx] --> B[Preprocessing\nLanguage detection + translation\nVocabulary source selection]
    B --> C[Normalised variable + vocabulary hint]
    C --> D[RAGMapper.map_umls\nFAISS k-NN search]
    D --> E[Top-k UMLS candidates\nwith cosine similarity]
    E --> F[Mapper LLM\ngranite4:latest\nRanks & filters candidates]
    F --> G[RankedCandidates\nJSON-structured]
    G --> H[Evaluator LLM\ngpt-oss:20b\nValidates & reranks]
    H --> I[SupervisorOutput\nRAG audit + final ranking]
    I --> J[Mapped schema\n.xlsx output]

    style D fill:#dbeafe
    style F fill:#fef3c7
    style H fill:#fef3c7
```