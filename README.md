### Simple GraphRAG Pipeline

---
This repo contains scripts for a simple graphRAG pipeline for cybersecurity. It leverages packages from the Neo4J ecosystem. 


---

### Steps

---

1. Go to https://llm-graph-builder.neo4jlabs.com/
2. Register to Neo4J Aura. [Link](https://console-preview.neo4j.io/).
3. Create an Aura instance. Download the credentials.
4. Download PDFs from [here](https://github.com/fractional-ciso/NIST-Cybersecurity-Documents).
5. Upload PDFs to LLM knowledge graph builder. Specifically, I upload the following 5 PDFs to save time and compute:
   - di-nist-sp1800-11a-draft.pdf
   - di-nist-sp1800-11b-draft.pdf
   - cr-mfa-nist-sp1800-17.pdf
   - derived-piv-nist-sp1800-12-v2.pdf
   - abac-nist-sp1800-3-draft.pdf
6. Select GPT-4o model in the lower left. It will be used to generate the knowledge graph.
7. Pull `tazarov/all-minilm-l6-v2-f32` with [Ollama](https:www.ollama.com/tazarov/all-minilm-l6-v2-f32).
8. The `tazarov/all-minilm-l6-v2-f32` model will be used for embedding generation.
9. Run the `aura_db_rag_script_v1.py`.

---

### Scripts

---

The scripts to run are:
- `aura_db_rag_script_v1.py`
  - Main script to run.
  - Leverages online Neo4J Aura instance.
  - Leverages local LLMs.
- `aura_db_rag_script_v2.py`
  - Leverages online Neo4J Aura instance.
  - Attempts to leverage online LLMs.
- `local_rag_script_v2.py`
  - Leverages local Neo4J database instance.
  - Attempts to leverage local LLMs.
  - Does not build very good knowledge graphs.
- `preprocessing.py`
  - Helper script for data preparation.