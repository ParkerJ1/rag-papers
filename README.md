# rag-papers

This is a fully local RAG (Retrieval-Augmented Generation) system for querying research papers. It ingest PDFs, embed them locally, store in a vector database, and query against them via a CLI or browser UI — no cloud APIs, no costs.

This project demonstrates local LLM deployment, vector search, and end-to-end pipeline design.

---

## Stack

| Component | Tool |
|---|---|
| PDF parsing | PyMuPDF |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| LLM inference | LFM2.5-1.2B-Instruct via llama.cpp server |
| UI | Gradio |
| Package management | uv |

---

## Project Structure

```
rag-papers/
  data/
    raw/          # place PDFs here (gitignored)
    processed/    # reserved for future use
  src/
    ingest.py     # PDF loading, chunking, embedding, storing
    query.py      # retrieval, prompt construction, generation
    ui.py         # Gradio interface
    main.py       # CLI entry point
    config.py     # shared configuration
  chroma_db/      # local vector store (gitignored)
```

---

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) pre-built binaries
- LFM2.5-1.2B-Instruct GGUF model (or any GGUF-format model)

### Install dependencies

```bash
uv venv
uv sync
```

### Start the llama.cpp inference server

```bash
path/to/llama-server.exe -m path/to/LFM2.5-1.2B-Instruct-Q4_K_M.gguf
```

The server runs at `http://localhost:8080` by default.

---

## Usage

### CLI

Ingest a single PDF:
```bash
uv run python src/main.py ingest data/raw/paper.pdf
```

Ingest a folder of PDFs (with automatic deduplication):
```bash
uv run python src/main.py ingest data/raw/
```

Query across all ingested papers:
```bash
uv run python src/main.py query "What deep learning methods are used for downscaling?"
```

Query filtered to a specific paper:
```bash
uv run python src/main.py query "What methods are used?" --filter "paper.pdf"
```

### Gradio UI

```bash
uv run python src/ui.py
```

Opens a browser interface at `http://localhost:7860` with tabs for querying and ingesting papers.

---

## Configuration

All key parameters are in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `TOP_K` | 5 | Number of chunks retrieved per query |
| `CHROMA_PATH` | `chroma_db` | Vector store location |
| `LLAMA_SERVER_URL` | `http://localhost:8080/v1/chat/completions` | Inference server endpoint |

---

## Design Decisions

> **Note:** This section reflects deliberate choices made during development.

**Chunking strategy**
<!-- Explain your choice of chunk size (500 chars) and overlap (50 chars). What did you observe? Were chunks too large, too small? Did you experiment? -->

**Embedding model**
<!-- Why all-MiniLM-L6-v2? What tradeoffs did you consider — speed vs quality, model size, etc? -->

**LFM2 over other local models**
<!-- Why LFM2.5-1.2B specifically? What does the Liquid AI architecture offer for RAG tasks? -->

**Deduplication approach**
<!-- Why check by source filename in metadata rather than hashing content? What are the tradeoffs? -->

**Retrieval quality observations**
<!-- What did you notice about the quality of answers? Where does the system do well, where does it struggle? -->

---

## Limitations

- Small LFM2 models are not well-suited for knowledge-intensive or programming tasks — retrieval quality matters more than generation quality at this scale
- Chunking is character-based rather than semantic — a sentence-aware splitter would likely improve retrieval precision
- No re-ranking of retrieved chunks before generation

---

## Future Work

- Sentence-aware chunking
- Re-ranking retrieved chunks by relevance score before generation
- Year and author metadata filtering
- Multi-turn conversational interface