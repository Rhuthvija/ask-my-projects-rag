# Ask My Projects: a small RAG demo

Ask questions about my own projects (Cell Scope, YOLO smoke detection, Anveshan, FocusIQ, AI Health Record Translator) and get answers taken **only** from their documentation, with sources.

I built this to learn Retrieval-Augmented Generation step by step, without a framework hiding the details. Built as a learning project, with AI assistance.

## How it works

```
Phase 1 (once):    docs/ → load → chunk (120 words, 30 overlap) → embed → vector index
Phase 2 (per question): question → embed → top-3 most similar chunks
                        → prompt: "answer ONLY from this context, cite sources" → LLM → answer
```

| Step | Function in `rag.py` | What it does |
| --- | --- | --- |
| Load | `load_documents` | Reads `.md`, `.txt` and `.pdf` files from `docs/` |
| Chunk | `chunk_text` | Splits text into 120-word chunks; neighbours share 30 words so sentences aren't cut in half |
| Embed | `Embedder` | `all-MiniLM-L6-v2` (sentence-transformers) turns each chunk into a vector that represents its meaning. Falls back to TF-IDF keyword vectors if the model can't be downloaded |
| Search | `VectorIndex.search` | Cosine similarity between the question and every chunk; returns the top 3 |
| Prompt | `build_prompt` | Puts the retrieved chunks in the prompt with a "use only this context" instruction |
| Generate | `generate_answer` | Calls Claude if `ANTHROPIC_API_KEY` is set; otherwise shows the retrieved chunks (retrieval-only mode) |

## Run it

```bash
pip install -r requirements.txt
python rag.py "Which OCR engine reads scanned lab reports?"
python rag.py                      # interactive mode
python evaluate.py                 # retrieval quality check

# optional, to generate full answers:
export ANTHROPIC_API_KEY=...
```

Add your own `.pdf`, `.md` or `.txt` files to `docs/` to ask questions about them.

## Evaluation

`evaluate.py` asks 12 test questions and checks whether the document that contains the answer is among the top 3 retrieved chunks (hit rate @ 3). If retrieval fails, the LLM never sees the right text, so this is the first thing to measure.

With the TF-IDF keyword fallback: **12/12 (100%)**. The questions are easy and each project is very distinct, so this mainly shows the pipeline works.

## Limitations and next steps

- Hit rate is measured per **document**, not per exact passage; a stricter test would check the retrieved chunk contains the answer.
- The vector index is in memory and rebuilt every run. For larger collections I'd use a vector database such as FAISS or Chroma and save the index.
- Fixed-size word chunks can split tables and code blocks; splitting by headings would keep sections together.
- Answer quality (is every claim supported by the retrieved chunks?) isn't evaluated yet.
- In LangChain, the same steps map to a document loader, `RecursiveCharacterTextSplitter`, an embeddings model, a vector store retriever and a prompt + LLM chain.
