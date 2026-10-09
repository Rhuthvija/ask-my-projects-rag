"""
Ask My Projects: a small Retrieval-Augmented Generation (RAG) demo.

It answers questions about my own projects using only their documentation.

Pipeline (each step is one function below):
  1. load_documents   read every .md / .txt / .pdf file in docs/
  2. chunk_text       split each document into small overlapping chunks
  3. Embedder         turn each chunk into a vector that represents its meaning
  4. VectorIndex      store the vectors and find the chunks closest to a question
  5. build_prompt     put the question + retrieved chunks into one prompt
  6. generate_answer  ask the LLM to answer ONLY from those chunks, citing sources

Run:
  python rag.py "What dataset did Cell Scope use?"
  python rag.py              (interactive mode)
"""
import os
import sys
from pathlib import Path

import numpy as np

DOCS_DIR = Path(__file__).parent / "docs"
CHUNK_WORDS = 120     # words per chunk
OVERLAP_WORDS = 30    # words shared between neighbouring chunks
TOP_K = 3             # how many chunks to retrieve per question


# 1. Load ------------------------------------------------------------------
def load_documents(folder=DOCS_DIR):
    """Return a list of (file_name, text)."""
    docs = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in (".md", ".txt"):
            docs.append((path.name, path.read_text(encoding="utf-8", errors="ignore")))
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
            docs.append((path.name, text))
    return docs


# 2. Chunk -----------------------------------------------------------------
def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split text into chunks of `size` words. Neighbouring chunks share
    `overlap` words, so a sentence cut at a boundary still appears whole in one chunk."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


# 3. Embed -----------------------------------------------------------------
class Embedder:
    """Turns text into vectors.

    Default: the sentence-transformers model all-MiniLM-L6-v2, which captures MEANING,
    so "dataset" can match "trained on 27,558 cell images" even with no shared words.
    Fallback (no internet / model not installed): TF-IDF, which only matches shared WORDS.
    """

    def __init__(self, texts):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.kind = "semantic (all-MiniLM-L6-v2)"
        except Exception as err:  # model unavailable -> keyword fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(stop_words="english").fit(texts)
            self.kind = f"keyword fallback (TF-IDF), because: {type(err).__name__}"

    def embed(self, texts):
        if hasattr(self.model, "encode"):
            vectors = self.model.encode(texts)
        else:
            vectors = self.model.transform(texts).toarray()
        # Normalise every vector to length 1, so a dot product = cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


# 4. Store + search --------------------------------------------------------
class VectorIndex:
    """A tiny in-memory vector database. Tools like FAISS or Chroma do the same
    thing, just faster and at much larger scale."""

    def __init__(self, embedder, chunks):
        self.embedder = embedder
        self.chunks = chunks                      # list of {"source", "text"}
        self.vectors = embedder.embed([c["text"] for c in chunks])

    def search(self, question, k=TOP_K):
        q = self.embedder.embed([question])[0]
        scores = self.vectors @ q                 # cosine similarity with every chunk
        best = np.argsort(scores)[::-1][:k]       # indexes of the k highest scores
        return [{**self.chunks[i], "score": float(scores[i])} for i in best]


def build_index(folder=DOCS_DIR):
    chunks = []
    for name, text in load_documents(folder):
        for piece in chunk_text(text):
            chunks.append({"source": name, "text": piece})
    embedder = Embedder([c["text"] for c in chunks])
    return VectorIndex(embedder, chunks)


# 5. Prompt ----------------------------------------------------------------
def build_prompt(question, retrieved):
    context = "\n\n".join(f"[{i + 1}] (from {c['source']})\n{c['text']}"
                          for i, c in enumerate(retrieved))
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say \"I couldn't find that in the documents.\" "
        "Cite the sources you used like [1] or [2].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


# 6. Generate --------------------------------------------------------------
def generate_answer(prompt):
    """Ask Claude if an API key is set; otherwise return None (retrieval-only mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    import anthropic
    client = anthropic.Anthropic()
    reply = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return reply.content[0].text


def ask(index, question):
    retrieved = index.search(question)
    answer = generate_answer(build_prompt(question, retrieved))
    print(f"\nQ: {question}")
    if answer:
        print(f"\nA: {answer}")
    else:
        print("\n(No ANTHROPIC_API_KEY set, so showing the retrieved chunks only.)")
    print("\nRetrieved chunks:")
    for i, c in enumerate(retrieved, 1):
        print(f"  [{i}] {c['source']}  (similarity {c['score']:.2f})  {c['text'][:110]}...")


if __name__ == "__main__":
    index = build_index()
    print(f"Indexed {len(index.chunks)} chunks from {DOCS_DIR.name}/ using {index.embedder.kind}")
    if len(sys.argv) > 1:
        ask(index, " ".join(sys.argv[1:]))
    else:
        while True:
            q = input("\nAsk a question (or press Enter to quit): ").strip()
            if not q:
                break
            ask(index, q)
