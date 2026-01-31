# RAG PDF Dashboard

A production‑ready, modular RAG (Retrieval‑Augmented Generation) app for chatting with your PDFs. Build multiple indexes, reuse older ones, and ask questions with OpenAI models through a clean Streamlit dashboard.

## Why this matters (relevance)
Most PDF chat demos feel “smart” until they hallucinate. This app keeps answers grounded by retrieving the most relevant chunks from your documents first, then generating responses only from that context. You get:

- **Relevance‑first answers**: retrieval narrows context to the most pertinent passages.
- **Repeatability**: reuse older RAG indexes for consistent results over time.
- **Fast iteration**: build a new index in minutes and compare with prior versions.

## Features
- **Dashboard UI**: choose an existing index or build a new one.
- **Multi‑index storage**: each RAG index is stored with metadata for quick reuse.
- **OpenAI‑based RAG**: OpenAI embeddings + chat model for retrieval and responses.
- **Transparent sources**: inspect the retrieved snippets used to answer.
- **Modular codebase**: clear separation of ingestion, chunking, vector store, and UI.

## Tech Stack
- **Python**
- **Streamlit**
- **LangChain**
- **FAISS**
- **OpenAI API**
- **PyPDF2**

## Project Structure
```
rag_app/
  chunking.py      # text split logic
  llm.py           # OpenAI LLM + embeddings
  pdf_utils.py     # PDF text extraction
  rag.py           # retrieval + answer chain
  settings.py      # defaults
  store.py         # index metadata
  ui.py            # Streamlit dashboard
  vector_store.py  # FAISS load/save
multipdfragapp.py  # app entry point
rag_store/         # local RAG indexes (created at runtime)
```

## Quickstart

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Set your OpenAI API key
```bash
export OPENAI_API_KEY="your_key_here"
```

3) Run the app
```bash
streamlit run multipdfragapp.py
```

## Usage
1. Open the dashboard.
2. In the sidebar, choose **Create new** to build an index from PDFs, or select an existing index.
3. Ask questions in the chat panel.
4. Expand **Sources** to see retrieved snippets that informed the answer.

## Configuration
You can tweak models and retrieval behavior from the sidebar:
- **Chat model**: e.g., `gpt-4o-mini`
- **Embedding model**: e.g., `text-embedding-3-small`
- **Top K**: number of chunks retrieved per query

Defaults live in `rag_app/settings.py`.

## Notes for Production
- Store indexes in a persistent volume if deploying (the app writes to `rag_store/`).
- Consider adding authentication if exposing publicly.
- For larger documents or heavy usage, swap FAISS local storage for a hosted vector DB.

## License
MIT. See `LICENSE`.
