# Simple RAG App

## Demo

> **Add a screenshot here!** Run `streamlit run app.py`, take a screenshot of the UI, save it as `docs/screenshot.png`, and replace this line with:
> > `![App Screenshot](docs/screenshot.png)`

A local Streamlit-based RAG application for uploading documents, embedding them into a vector database, asking document-grounded questions, and evaluating answer quality.

## What It Does

- Upload `PDF` and `TXT` documents
- Split documents with multiple chunking strategies
- Embed chunks with local embedding models
- Store vectors in local databases such as `Chroma`
- Ask questions against the uploaded document set
- Audit answers with a built-in `RAG Evaluator`
- Run automated benchmarks with relevant and irrelevant test questions

## Main Stack

- `Streamlit` for UI
- `LangChain` for retrieval pipeline
- `Chroma` for local vector storage
- `Ollama` or local `GGUF` models for inference
- `sentence-transformers` / Hugging Face embeddings

## Requirements

- Python `3.12` recommended
- `Ollama` installed and running

Install Ollama:

https://ollama.com/download

## How To Run

```bash
git clone <your-repo-url>
cd simple_rag
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
ollama serve
```

In another terminal:

```bash
cd simple_rag
source venv/bin/activate
ollama pull qwen2.5
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## Recommended First Run

Inside the app:

- Inference engine: `Ollama (Native Service)`
- Model: `qwen2.5`
- Database: `Chroma DB (Local Persist)`
- Chunking: `Fixed Size Chunking`

Then:

1. Upload a PDF or TXT file
2. Wait for embedding to finish
3. Ask a document-specific question

## Optional Advanced Features

Some advanced chunking and database modes require extra packages beyond `requirements.txt`.

```bash
pip install langchain-experimental
pip install langchain-qdrant qdrant-client
pip install langchain-milvus "pymilvus[milvus_lite]"
```

Or install all extras at once:

```bash
pip install langchain-experimental langchain-qdrant qdrant-client langchain-milvus "pymilvus[milvus_lite]"
```

## Notes

- `Ollama` is the intended default runtime for this app
- Local GGUF models are also supported through `llama.cpp`
- Benchmark and evaluator tools are built into the UI
