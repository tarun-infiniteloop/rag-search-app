# RAG Search App

A local Retrieval-Augmented Generation project that:

- ingests `CSV`, `XLSX`, and `PDF` files
- creates embeddings with `sentence-transformers`
- stores vectors in `FAISS`
- retrieves relevant results across one file, many files, or a folder
- sends retrieved context to `Gemini` for final answer generation
- provides both a CLI-style entrypoint and a `Streamlit` UI

## Features

- Multi-format ingestion: `csv`, `xlsx`, `pdf`
- Page-aware PDF chunk metadata
- Cached embeddings, units, and FAISS indexes
- Folder-wide and uploaded-file search
- Deduplicated retrieval results
- Source citations in final answers
- Streamlit UI with chat history and expandable source inspection

## Project structure

```text
src/
  app.py                  Streamlit app
  embeddings.py           Embedding model + query embedding
  generation.py           Gemini prompt + answer generation
  ingestion.py            CSV/XLSX/PDF ingestion and chunking
  retrieval.py            FAISS retrieval and metadata helpers
  storage.py              Embedding/unit persistence helpers
  step1_csv_search.py     Early learning step
  step2_pdf_search.py     Early learning step
  step3_combined_search.py Early learning step
  step4_embedding_search.py CLI-style entrypoint
```

## Requirements

- Python 3.11
- A Gemini API key in `.env`

Example `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install streamlit faiss-cpu google-genai python-dotenv
```

If needed, also install:

```bash
pip install sentence-transformers torch torchvision
```

## Run the Streamlit app

```bash
streamlit run src/app.py
```

## Run the CLI-style entrypoint

```bash
python src/step4_embedding_search.py
```

## Supported inputs

- Single file path
- Folder path
- Uploaded files in Streamlit

Supported file types:

- `.csv`
- `.xlsx`
- `.pdf`

## Retrieval pipeline

1. Collect files
2. Load tabular rows or PDF chunks
3. Create embeddings
4. Load or create cached FAISS index
5. Retrieve nearest results
6. Build context
7. Send context to Gemini
8. Return answer with citations

## Notes

- Generated files such as `.npy`, `.index`, and uploaded files are ignored by git.
- The `step1` to `step4` scripts are part of the learning journey and can be kept for reference.
- The modular files in `src/` are the main implementation going forward.
